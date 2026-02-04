import torch

def dp_updates(weight_accumulator):
    """
    Add differential privacy noise to updates
    """
    for name, weights in weight_accumulator.items():
        # Ensure the weights are on the same device as the noise
        device = weights.device
        noise = torch.normal(0, 0.002, size=weights.size(), device=device)
        weights = weights + noise
    return

def dp_avg(global_model, weight_accumulator, helper):
    """
    Differential privacy aggregation method
    """
    sigma = 0.002
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        update_per_layer = weight_accumulator[name] * \
                           (1/helper.config.num_sampled_participants) 
        update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
        data.add_(update_per_layer.cuda())
    
    for name, param in global_model.state_dict().items():
        if 'tracked' in name or 'running' in name:
            continue
        dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
        param.add_(dp_noise)
    
    return True 