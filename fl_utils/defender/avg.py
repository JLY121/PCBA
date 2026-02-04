import torch

def average_shrink_models(global_model, weight_accumulator, helper):
    """
    Perform FedAvg algorithm and perform some clustering on top of it.
    """
    lr = 1

    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        update_per_layer = weight_accumulator[name] * \
                           (1/helper.config.num_sampled_participants) * lr
        update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
        data.add_(update_per_layer.cuda())

    return True 