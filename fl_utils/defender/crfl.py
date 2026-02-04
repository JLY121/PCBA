import torch
import math
import copy

def get_global_model_norm(global_model):
    squared_sum = 0
    for name, layer in global_model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data, 2))
    return math.sqrt(squared_sum)

def clip_norm(global_model, clip=100):
    total_norm = get_global_model_norm(global_model)
    max_norm = clip
    clip_coef = max_norm / (total_norm + 1e-6)
    current_norm = total_norm
    if total_norm > max_norm:
        for name, layer in global_model.named_parameters():
            layer.data.mul_(clip_coef)
    return

def add_noise(global_model, sigma=0.002, cp=False):
    '''
    向模型权重添加差分隐私噪声
    '''
    if not cp:
        for name, param in global_model.state_dict().items():
            if 'tracked' in name or 'running' in name:
                continue
            dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
            param.add_(dp_noise)
    else:
        smoothed_model = copy.deepcopy(global_model)
        for name, param in smoothed_model.state_dict().items():
            if 'tracked' in name or 'running' in name:
                continue
            dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
            param.add_(dp_noise)
    return

def crfl_agg(global_model, weight_accumulator, helper):
    """
    CRFL aggregation method
    """
    lr = 1
    clip_norm(global_model)
    add_noise(global_model=global_model)
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        update_per_layer = weight_accumulator[name] * \
                           (1/helper.config.num_sampled_participants) * lr
        update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
        data.add_(update_per_layer.cuda())

    return True 