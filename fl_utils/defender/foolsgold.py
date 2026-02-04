import torch
import numpy as np
from sklearn.metrics import pairwise as smp

def foolsgold(this_delta, summed_deltas, epsilon=1e-5):
    """
    FoolsGold defense algorithm.
    """
    n_clients = this_delta.shape[0]

    # 计算客户端历史更新的余弦相似度矩阵
    cs = smp.cosine_similarity(summed_deltas) - np.eye(n_clients) # ===余弦相似度矩阵-单位矩阵

    # Pardoning：根据最大相似度进行重新加权
    maxcs = np.max(cs, axis=1) + epsilon
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    # 计算权重向量
    wv = 1 - np.max(cs, axis=1)
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # 重新缩放权重
    wv = wv / np.max(wv)
    wv[wv == 1] = 0.99

    # 应用 Logit 函数
    wv = (np.log(wv / (1 - wv) + epsilon) + 0.5)
    wv[np.isinf(wv) | (wv > 1)] = 1
    wv[wv < 0] = 0

    return wv

def foolsgold_aggregate(global_model, weight_accumulator_by_client, sampled_participants, helper, history_updates=None, n_features=None, wv=None):
    """
    Implement FoolsGold defense algorithm for aggregation.
    """
    # 首先，将每个客户端的更新转换为向量形式，并维护历史更新
    client_updates = []
    if history_updates is None:
        history_updates = {}
    
    for idx, client_id in enumerate(sampled_participants):
        client_update = weight_accumulator_by_client[idx]
        update = []
        for name, data in client_update.items():
            if 'num_batches_tracked' in name:
                continue
            update.append(data.view(-1).cpu().numpy())
        update = np.concatenate(update)
        client_updates.append(update)

        # 更新历史更新(通过累加得到)  → 为所有客户端维护一个历史更新
        if client_id in history_updates:
            history_updates[client_id] += update
        else:
            history_updates[client_id] = update.copy()

    client_updates = np.array(client_updates)  # Shape: (n_clients, n_features)

    # 初始化特征维度
    if n_features is None:
        n_features = client_updates.shape[1]

    # 获取对应的历史更新  → 找到本轮参与聚合的客户端对应的历史更新
    summed_deltas = []
    for client_id in sampled_participants:
        summed_deltas.append(history_updates[client_id])
    summed_deltas = np.array(summed_deltas)  # Shape: [n_clients, n_features]

    # 获取全局模型的扁平化参数
    global_params = []
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        global_params.append(data.view(-1).cpu().numpy())
    global_params = np.concatenate(global_params)

    # 调用 foolsgold 方法，计算权重向量
    wv = foolsgold(client_updates, summed_deltas)
    
    # 根据权重向量，更新全局模型
    lr = 1
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        # 收集所有客户端在该参数上的更新
        updates = np.array([weight_accumulator_by_client[idx][name].cpu().numpy() for idx in range(len(sampled_participants))])
        # 计算加权平均的更新
        weighted_update = np.average(updates, axis=0, weights=wv)
        weighted_update = torch.tensor(weighted_update, dtype=data.dtype)
        data.add_(weighted_update.cuda())

    return True, history_updates, n_features, wv 