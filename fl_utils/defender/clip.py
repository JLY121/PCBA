import torch

def clip_updates(agent_updates_dict, helper):
    """
    Clip updates to prevent large gradients
    """
    for key in agent_updates_dict:
        if 'num_batches_tracked' not in key:
            update = agent_updates_dict[key]
            l2_update = torch.norm(update, p=2) 
            update.div_(max(1, l2_update/helper.config.clip_factor))
    return

def clip_aggregate(global_model, weight_accumulator, helper):
    """
    Clip updates and then perform FedAvg
    """
    # First clip the updates
    clip_updates(weight_accumulator, helper)
    
    # Then perform FedAvg
    lr = 1
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        update_per_layer = weight_accumulator[name] * \
                           (1/helper.config.num_sampled_participants) * lr
        update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
        data.add_(update_per_layer.cuda())

    return True 