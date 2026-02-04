import torch
import numpy as np
from collections import OrderedDict

def compute_pairwise_distance(updates):
    '''
    计算客户端之间的权重更新的欧式距离
    '''
    def pairwise(u1, u2):
        ks = u1.keys()
        dist = 0
        for k in ks:
            if 'tracked' in k:
                continue
            d = u1[k] - u2[k]
            dist = dist + torch.sum(d * d)
        return round(float(torch.sqrt(dist)), 2)

    scores = [0.0 for u in range(len(updates))]
    for i in range(len(updates)):
        for j in range(i + 1, len(updates)):
            dist = pairwise(updates[i], updates[j])
            scores[i] = scores[i] + dist
            scores[j] = scores[j] + dist
    return scores

def bulyan_aggregate_global_model(global_model, weight_accumulator_by_client, sampled_participants, helper):
    """
    实现 Bulyan 聚合方法，并使用与 average_shrink_models 一致的方式更新全局模型。

    参数:
    - global_model: 全局模型
    - weight_accumulator_by_client: 每个客户端的更新权重（字典列表）
    - sampled_participants: 当前采样的客户端列表
    """
    num_clients = len(sampled_participants)
    num_adv = helper.config.num_adversaries  # 恶意客户端数量
    f = num_adv  # 假设拜占庭客户端数量为恶意客户端数量

    theta = num_clients - 2 * f  # 需要选择的更新数量

    selected_updates = []
    remaining_updates = weight_accumulator_by_client.copy()
    remaining_indices = list(range(len(weight_accumulator_by_client)))
    
    # Bulyan第一阶段：迭代使用Krum算法选择theta个更新
    for _ in range(theta):
        num_remaining = len(remaining_updates)
        nb_in_score = num_remaining - f - 2
        if nb_in_score < 1:
            nb_in_score = 1  # 确保至少有一个邻居

        # 将剩余更新转换为向量形式
        vectorized_updates = []
        for client_model in remaining_updates:
            updates = []
            for layer_update in client_model.values():
                updates.append(torch.tensor(layer_update).flatten())
            vectorized_updates.append(torch.cat(updates).cpu().detach().numpy())

        # 计算剩余更新之间的欧氏距离
        distances = np.zeros((num_remaining, num_remaining))
        for i in range(num_remaining):
            for j in range(i + 1, num_remaining):
                distances[i, j] = np.linalg.norm(vectorized_updates[i] - vectorized_updates[j]) ** 2
                distances[j, i] = distances[i, j]

        # 计算每个更新的得分
        scores = []
        for i in range(num_remaining):
            dists = np.sort(distances[i])
            scores.append(np.sum(dists[:nb_in_score + 1]))  # 包含自身距离（为0）

        # 选择得分最低的更新
        i_star = np.argmin(scores)
        selected_updates.append(remaining_updates[i_star])

        # 从剩余更新中移除已选中的更新
        del remaining_updates[i_star]
        del remaining_indices[i_star]

    original_params = global_model.state_dict()
    bulyan_update = OrderedDict()
    layers = selected_updates[0].keys()
    for layer in layers:
        bulyan_layer = None
        for update in selected_updates:
            bulyan_layer = update[layer][None, ...] if bulyan_layer is None else torch.cat(
                (bulyan_layer, update[layer][None, ...]), 0)

        med, _ = torch.median(bulyan_layer, 0)
        _, idxs = torch.sort(torch.abs(bulyan_layer - med), 0)
        bulyan_layer = torch.gather(bulyan_layer, 0, idxs[:-2*f, ...])
        
        if not 'tracked' in layer:
            bulyan_update[layer] = torch.mean(bulyan_layer, 0)
        else:
            bulyan_update[layer] = torch.mean(bulyan_layer*1.0, 0).long()
        original_params[layer] = original_params[layer] + bulyan_update[layer]

    global_model.load_state_dict(original_params)
    
    return True 