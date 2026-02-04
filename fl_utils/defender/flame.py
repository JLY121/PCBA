import torch
import numpy as np
import copy
import hdbscan

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    """将参数字典转换为向量形式，跳过num_batches_tracked"""
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    """将参数字典转换为向量形式，只包含weight和bias"""
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def no_defence_balance(params, global_parameters):
    """无防御聚合方法"""
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters

def flame_aggregate(global_model, weight_accumulator_by_client, sampled_participants, helper):
    """
    FLAME聚合方法实现
    
    参数:
    - global_model: 全局模型
    - weight_accumulator_by_client: 每个客户端的权重更新
    - sampled_participants: 采样的客户端列表  
    - helper: 辅助对象，包含配置信息
    """
    print(f"---Using FLAME defender---")
    # 计算客户端模型参数向量（用于聚类）
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    
    # 为了计算余弦相似度，我们需要重构客户端的模型参数
    # 由于只有更新参数，我们使用全局模型 + 更新来近似本地模型
    local_model_vectors = []
    global_state = global_model.state_dict()
    
    for client_update in weight_accumulator_by_client:
        # 重构客户端模型参数 (global_model + update)
        client_params = {}
        for key in global_state.keys():
            if key in client_update:
                client_params[key] = global_state[key] + client_update[key]
            else:
                client_params[key] = global_state[key]
        
        # 转换为向量
        local_model_vectors.append(parameters_dict_to_vector_flt(client_params))
    
    # 计算余弦相似度距离矩阵
    for i in range(len(local_model_vectors)):
        cos_i = []
        for j in range(len(local_model_vectors)):
            cos_ij = 1 - cos(local_model_vectors[i], local_model_vectors[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    
    # 获取客户端数量信息
    num_clients = len(sampled_participants)
    
    # 使用HDBSCAN进行聚类
    min_cluster_size = max(num_clients // 2 + 1, 2)  # 确保至少为2
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1, 
        allow_single_cluster=True
    ).fit(cos_list)
    
    print(f"FLAME clustering labels: {clusterer.labels_}")
    
    # 选择良性客户端
    benign_clients = []
    
    if clusterer.labels_.max() < 0:
        # 没有找到聚类，使用所有客户端
        benign_clients = list(range(len(weight_accumulator_by_client)))
    else:
        # 找到最大的聚类
        max_num_in_cluster = 0
        max_cluster_index = 0
        for cluster_idx in range(clusterer.labels_.max() + 1):
            cluster_size = len(clusterer.labels_[clusterer.labels_ == cluster_idx])
            if cluster_size > max_num_in_cluster:
                max_cluster_index = cluster_idx
                max_num_in_cluster = cluster_size
        
        # 选择最大聚类中的客户端
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_clients.append(i)
    
    print(f"FLAME selected benign clients: {benign_clients}")
    
    # 计算更新参数的范数
    norm_list = []
    for i in range(len(weight_accumulator_by_client)):
        norm = torch.norm(parameters_dict_to_vector(weight_accumulator_by_client[i]), p=2).item()
        norm_list.append(norm)
    norm_list = np.array(norm_list)
    
    # 范数裁剪
    clip_value = np.median(norm_list)
    selected_updates = []
    
    for i in benign_clients:
        client_update = copy.deepcopy(weight_accumulator_by_client[i])
        gamma = clip_value / norm_list[i]
        if gamma < 1:
            for key in client_update:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                client_update[key] *= gamma
        selected_updates.append(client_update)
    
    # 获取全局模型状态并进行聚合
    global_state = global_model.state_dict()
    global_state = no_defence_balance(selected_updates, global_state)
    
    # 添加差分隐私噪声
    noise_std = getattr(helper.config, 'noise_std', 0.001)  # 默认噪声标准差
    for key, var in global_state.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        temp = torch.zeros_like(var)
        temp = temp.normal_(mean=0, std=noise_std * clip_value)
        var += temp
    
    # 加载更新后的状态到全局模型
    global_model.load_state_dict(global_state)
    
    return True
