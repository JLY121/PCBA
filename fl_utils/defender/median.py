import torch

def median_shrink_models(global_model, weight_accumulator_by_client, helper):
    """
    Perform model aggregation using median
    """
    # 遍历全局模型的每一层参数
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        
        # 获取该层所有客户端的更新，并计算中位数
        layer_updates = torch.stack([participant_update[name] for participant_update in weight_accumulator_by_client], dim=0)

        # 计算每一层参数更新的中位数
        median_update_per_layer = torch.median(layer_updates, dim=0)[0]

        # 将中位数更新添加到全局模型的对应参数中
        median_update_per_layer = torch.tensor(median_update_per_layer, dtype=data.dtype)
        data.add_(median_update_per_layer.cuda())  # 将更新后的值应用到全局模型参数中

    return True 