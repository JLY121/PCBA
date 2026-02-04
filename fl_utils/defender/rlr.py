import torch
from collections import OrderedDict

def compute_robust_lr(updates):
    """
    Compute the robust learning rates based on client updates.
    """
    layers = updates[0].keys()
    robust_lrs = OrderedDict()
    for layer in layers:
        robust_lrs[layer] = torch.zeros_like(updates[0][layer])

    for layer in layers:
        for update in updates:
            robust_lrs[layer] += torch.sign(update[layer])
        robust_lrs[layer] = torch.abs(robust_lrs[layer])
        robust_lrs[layer][robust_lrs[layer] >= 2] = 1.0
        robust_lrs[layer][robust_lrs[layer] != 1.0] = -1.0
    return robust_lrs

def robust_lr_add_weights(original_params, robust_lrs, update, prop):
    """
    Update global model weights using robust learning rate and client updates.
    """
    for layer in original_params.keys():
        if layer == 'decoder.weight':
            continue
        tmp_updates=update[layer]*prop*robust_lrs[layer]
        tmp_updates =torch.tensor(tmp_updates,dtype=original_params[layer].dtype)
        original_params[layer] += tmp_updates

def robust_lr_aggregate(global_model, weight_accumulator_by_client, sampled_participants, helper):
    """
    Perform robust learning rate aggregation using sign voting method.
    """
    original_params = global_model.state_dict()
    total_sample = sum([len(helper.train_data[id].dataset)for id in sampled_participants])

    # 收集客户端更新
    updates = weight_accumulator_by_client

    # 计算鲁棒学习率
    robust_lrs = compute_robust_lr(updates)

    # 进行符号投票聚合
    flip_analysis = {}
    for layer in robust_lrs.keys():
        n_flip = torch.sum(torch.gt(robust_lrs[layer], 0.0).int())
        n_unflip = torch.sum(torch.lt(robust_lrs[layer], 0.0).int())
        flip_analysis[layer] = [n_flip, n_unflip]

    # 根据符号投票结果更新全局模型
    for i, id in enumerate(sampled_participants):
        client_update = weight_accumulator_by_client[i]
        prop = len(helper.train_data[id].dataset) / total_sample
        for layer in original_params.keys():
            if layer == 'decoder.weight':
                continue
            tmp_updates=client_update[layer]*prop*robust_lrs[layer]
            tmp_updates =torch.tensor(tmp_updates,dtype=original_params[layer].dtype)
            original_params[layer] += tmp_updates

    global_model.load_state_dict(original_params)
    return True 