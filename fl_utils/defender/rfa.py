import torch
import numpy as np
import math
import copy
from collections import OrderedDict

def geometric_median(points, max_iter=4, eps=1e-5):
    """
    使用Weiszfeld算法计算一组点的几何中位数。
    """
    median = np.mean(points, axis=0)
    for _ in range(max_iter):
        prev_median = median.copy()
        distances = np.linalg.norm(points - median, axis=1)
        # 避免除以零
        near_zero = distances < eps
        if np.any(near_zero):
            # 如果任何距离为零，将中位数设置为该点
            median = points[near_zero][0]
            break
        inv_distances = 1.0 / distances
        weights = inv_distances / np.sum(inv_distances)
        median = np.sum(weights[:, np.newaxis] * points, axis=0)
        # 检查收敛性
        if np.linalg.norm(median - prev_median) < eps:
            break
    return median

def robust_federated_aggregation(global_model, weight_accumulator_by_client, helper):
    """
    使用几何中位数执行鲁棒联邦聚合（RFA）。
    """
    client_updates = []

    # 收集并展平客户端更新
    for client_update_dict in weight_accumulator_by_client:
        client_update = []
        for name, param in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update = client_update_dict[name]
            update = update.cpu().numpy().flatten()
            client_update.append(update)
        client_update = np.concatenate(client_update)
        client_updates.append(client_update)

    client_updates = np.array(client_updates)  # 形状: (num_clients, total_params)

    # 计算几何中位数
    median_update = geometric_median(client_updates)

    # 更新全局模型
    lr = 1  # 学习率
    idx = 0
    for name, param in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        param_shape = param.size()
        param_size = param.numel()
        median_param_update = median_update[idx:idx + param_size]
        median_param_update = median_param_update.reshape(param_shape)
        median_param_update = torch.tensor(median_param_update, dtype=param.dtype)
        param.data.add_(median_param_update.to(param.device) * lr)
        idx += param_size

    return True

def l2dist(p1, p2):
    """计算p1和p2之间的L2距离"""
    squared_sum = 0
    for name, data in p1.items():
        squared_sum += torch.sum(torch.pow(p1[name] - p2[name], 2))
    return math.sqrt(squared_sum)

def geometric_median_objective(median, points, alphas):
    """计算几何中位数的目标值"""
    temp_sum = 0
    for alpha, p in zip(alphas, points):
        temp_sum += alpha * l2dist(median, p)
    return temp_sum

def weighted_average_oracle(points, weights):
    """计算加权平均"""
    tot_weights = torch.sum(weights)

    weighted_updates = dict()

    for name, data in points[0].items():
        weighted_updates[name] = torch.zeros_like(data)
    for w, p in zip(weights, points):  # 对每一个agent
        for name, data in weighted_updates.items():
            temp = (w / tot_weights).float()
            temp = temp * (p[name].float())
            if temp.dtype != data.dtype:
                temp = temp.type_as(data)
            data.add_(temp)

    return weighted_updates

def geometric_median_update(target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm=None):
    points = updates
    alphas = [0.1] * len(updates)

    alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
    alphas = torch.from_numpy(alphas).float()

    median = weighted_average_oracle(points, alphas)  # 计算加权平均值
    num_oracle_calls = 1

    obj_val = geometric_median_objective(median, points, alphas)
    logs = []
    log_entry = [0, obj_val, 0, 0]
    logs.append(log_entry)

    wv = None
    for i in range(maxiter):
        prev_median, prev_obj_val = median, obj_val
        weights = torch.tensor([alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alphas, points)], dtype=alphas.dtype)
        weights = weights / weights.sum()
        median = weighted_average_oracle(points, weights)
        num_oracle_calls += 1
        obj_val = geometric_median_objective(median, points, alphas)
        log_entry = [i + 1, obj_val, (prev_obj_val - obj_val) / obj_val, l2dist(median, prev_median)]
        logs.append(log_entry)

        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
        wv = copy.deepcopy(weights)
    alphas = [l2dist(median, p) for p in points]

    update_norm = 0
    for name, data in median.items():
        update_norm += torch.sum(torch.pow(data, 2))
    update_norm = math.sqrt(update_norm)

    if max_update_norm is None or update_norm < max_update_norm:  # 如果更新的范数不过大，则应用更新
        for name, data in target_model.state_dict().items():
            update_per_layer = median[name]
            data.add_(update_per_layer)
        is_updated = True
    else:
        is_updated = False

    return True 