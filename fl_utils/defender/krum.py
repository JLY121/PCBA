import torch
import numpy as np

def krum(global_model, weight_accumulator_by_client, sampled_participants, helper):
    """
    实现 Krum 聚合方法，并使用与 average_shrink_models 一致的方式更新全局模型。

    参数:
    - global_model: 全局模型的参数
    - weight_accumulator_by_client: 每个客户端的更新权重
    - sampled_participants: 当前采样的客户端列表
    """
    # 将每个客户端的更新转化为向量
    vectorized_updates = []
    for client_model in weight_accumulator_by_client:
        # 如果 weight_accumulator_by_client 是字典类型，提取每个层的更新并转化为向量
        updates = []
        for layer_update in client_model.values():
            updates.append(torch.tensor(layer_update).flatten())  # 将每个层的更新展平成向量
        vectorized_updates.append(torch.cat(updates).cpu().detach().numpy())  # 将各层的向量合并成一个整体

    num_clients = len(sampled_participants)
    num_adv = helper.config.num_adversaries
    nb_in_score = num_clients - num_adv - 2

    # 计算每对客户端之间的欧几里得距离，并存储距离矩阵
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            distances[i, j] = np.linalg.norm(vectorized_updates[i] - vectorized_updates[j]) ** 2
            distances[j, i] = distances[i, j]  # 对称距离矩阵

    # 计算每个客户端的得分
    scores = []
    for i in range(num_clients):
        dists = np.sort(distances[i])  # 按距离排序
        scores.append(np.sum(dists[:nb_in_score + 1]))  # 计算前 nb_in_score 个邻居的距离总和

    # 选择得分最低的客户端
    i_star = np.argmin(scores)

    # 使用与 average_shrink_models 类似的方式更新模型
    lr = 1
    for name, data in global_model.state_dict().items():
        if name in weight_accumulator_by_client[i_star]:  # 仅处理有效的更新
            update_per_layer = weight_accumulator_by_client[i_star][name]
            update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)
            data.add_(update_per_layer.cuda())

    return True 