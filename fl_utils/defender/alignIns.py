import torch
import numpy as np

def alignins_aggregate(global_model, weight_accumulator_by_client, sampled_participants, helper):
    """
    AlignIns 联邦聚合（基于 AlignIns-master/src/aggregation.py 中的 agg_alignins 思路）。

    说明：
    - 输入输出风格对齐到 avg.py：对 global_model 就地加和更新并返回 True。
    - 本实现内部完成对 `global_model` 的扁平化操作（原版的 flat_global_model）。
    - 所需超参数在函数内写死：与 federated.py 中默认参数保持一致。
    - 客户端编号通过 `sampled_participants` 传入（与 select_aggregator.py 其它方法一致）。

    参数:
    - global_model: 当前全局模型（就地更新）
    - weight_accumulator_by_client: List[Dict[str, Tensor]]，每个元素是一个客户端的逐层更新张量
    - sampled_participants: List[int]，本轮参与的客户端全局编号
    - helper: 带有 config 的辅助对象（仅用于读取设备等通用上下文，不读取 AlignIns 超参）
    """

    # ====================== Step 0: 固定超参数 ======================
    SPARSITY = 0.3   # 对应 args.sparsity 的默认值
    LAMBDA_S = 1.0   # 对应 args.lambda_s 的默认值
    LAMBDA_C = 1.0   # 对应 args.lambda_c 的默认值

    device = next(global_model.parameters()).device

    # ====================== Step 1: 扁平化全局模型（flat_global_model） ======================
    # 记录参数展开顺序与形状，保证展平与回填一致
    param_meta = []  # [(name, numel, shape, dtype, device)]
    flat_global_list = []
    for name, param in global_model.state_dict().items():
        if name == 'decoder.weight':
            # 按现有代码风格，部分方法会跳过该层
            continue
        flat_global_list.append(param.view(-1).detach().to(device))
        param_meta.append((name, param.numel(), param.shape, param.dtype, param.device))
    if len(flat_global_list) == 0:
        return True
    flat_global_model = torch.cat(flat_global_list, dim=0)

    # ====================== Step 2: 扁平化各客户端更新 ======================
    # 生成与全局模型相同维度与顺序的客户端更新向量
    local_updates = []  # List[Tensor[D]]
    for client_update in weight_accumulator_by_client:
        flat_list = []
        meta_idx = 0
        for name, _numel, _shape, _dtype, _param_dev in param_meta:
            if name not in client_update:
                # 若缺失某层更新，视为零更新（健壮性考虑）
                flat_list.append(torch.zeros(_numel, device=device, dtype=_dtype)) # 如果没有某一层，则用0填充
            else:
                upd = client_update[name]
                if not torch.is_tensor(upd):
                    upd = torch.tensor(upd)
                upd = upd.to(device=device, dtype=_dtype).view(-1)
                flat_list.append(upd)
            meta_idx += 1
        local_updates.append(torch.cat(flat_list, dim=0))

    if len(local_updates) == 0:
        return True

    inter_model_updates = torch.stack(local_updates, dim=0)  # [N, D]

    # ====================== Step 3: 计算 MPSA 和 TDA 指标 ======================
    # major_sign: 全体客户端更新符号的主导符号
    major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))

    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    tda_list = []  # 与全局模型的余弦相似度
    mpsa_list = [] # 主要符号一致性比例

    # 客户端数量  参数维度
    num_clients, dim = inter_model_updates.shape
    k_top = max(1, int(dim * SPARSITY))

    for i in range(num_clients):
        vec = inter_model_updates[i]
        _, init_indices = torch.topk(torch.abs(vec), k_top)
        # MPSA: 选中稀疏子集上，符号与主符号一致的比例
        mpsa = (torch.sum(torch.sign(vec[init_indices]) == major_sign[init_indices]).float() / init_indices.numel()).item()
        mpsa_list.append(mpsa)
        # TDA: 与全局模型的余弦相似度
        tda_list.append(cos(vec, flat_global_model).item())

    # ====================== Step 4: 计算 MZ 分数并筛选良性客户端 ======================
    def mzscores(values):
        values = np.asarray(values, dtype=np.float64)
        med = np.median(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return np.abs(values - med) / std

    mzscore_mpsa = mzscores(mpsa_list)
    mzscore_tda = mzscores(tda_list)

    benign_idx_by_mpsa = set([int(i) for i in np.argwhere(mzscore_mpsa < LAMBDA_S).flatten()])
    benign_idx_by_tda  = set([int(i) for i in np.argwhere(mzscore_tda < LAMBDA_C).flatten()])
    benign_idx = list(benign_idx_by_mpsa.intersection(benign_idx_by_tda))

    print(f"[AlignIns] 本轮全部参与客户端（全局编号）: {list(map(int, sampled_participants))}")

    benign_global_ids = [int(sampled_participants[i]) for i in benign_idx]
    print(f"[AlignIns] 本轮良性客户端（全局编号）: {benign_global_ids}")

    if len(benign_idx) == 0:
        # 无良性客户端，视为不更新
        return True

    # ====================== Step 5: 后过滤模型裁剪（范数裁剪） ======================
    # 注：与原实现一致，先以所有更新计算中位数裁剪阈值，再缩放每个客户端的更新向量
    updates_norm = torch.norm(inter_model_updates, dim=1, p=2).view(-1, 1)  # [N, 1]
    norm_clip = torch.median(updates_norm).item()
    updates_norm_clipped = torch.clamp(updates_norm, max=norm_clip)
    # 避免除零
    safe_denominator = updates_norm + 1e-12
    scaled_updates = (inter_model_updates / safe_denominator) * updates_norm_clipped  # [N, D]

    # ====================== Step 6: 对良性客户端进行平均聚合 ======================
    aggregated_update_vec = torch.mean(scaled_updates[benign_idx, :], dim=0)  # [D]

    # ====================== Step 7: 将扁平向量回填到各层并加到全局模型 ======================
    offset = 0
    for name, numel, shape, dtype, param_dev in param_meta:
        slice_vec = aggregated_update_vec[offset:offset + numel]
        slice_tensor = slice_vec.view(shape).to(dtype=dtype, device=param_dev)
        global_model.state_dict()[name].add_(slice_tensor)
        offset += numel

    return True


