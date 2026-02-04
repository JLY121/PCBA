import torch

def get_total_samples(data_loader):
    total_samples = 0
    for data in data_loader:
        total_samples += data[0].size(0)
    return total_samples

def fednova_aggregate_global_model(global_model, weight_accumulator_by_client, sampled_participants, helper):
    """
    FedNova algorithm.
    global_model: global model
    weight_accumulator_by_client: 每个客户端的权重更新（全剧模型减去局部模型）
    sampled_participants: 采样的客户端ID
    """
    # 获取全局模型的初始状态
    original_params = global_model.state_dict()

    # 收集每个客户端的数据样本数
    list_num_samples = [get_total_samples(helper.train_data[id]) for id in sampled_participants]

    # 计算总样本量
    total_sample = sum(list_num_samples)

    # 动量参数rho，FedNova中通常使用0.9（根据需要可调整）
    rho = 0.9
    tau_list = []

    # 计算每个客户端的tau值，tau代表每个客户端的训练步数
    for client_id in sampled_participants:
        num_samples = get_total_samples(helper.train_data[client_id])
        tau = num_samples * 2  # 假设每个样本进行两次训练，这可以根据实际需要调整
        tau_list.append(tau)

    # 初始化全局模型参数的累积更新量
    d_total_round = {key: torch.zeros_like(param) for key, param in original_params.items()}

    # 初始化FedNova中的总系数，用于全局模型参数更新
    total_coeff = 0.0

    # 遍历每个客户端，进行权重更新的归一化和聚合
    for i, client_id in enumerate(sampled_participants):
        # 计算当前客户端的tau值和归一化因子a_i
        tau = tau_list[i]
        a_i = (tau - rho * (1 - pow(rho, tau)) / (1 - rho)) / (1 - rho)

        # 获取客户端的模型更新 (global_model - local_model)
        client_weight_update = weight_accumulator_by_client[i]

        # 归一化客户端的权重更新，并根据数据量加权
        for key in client_weight_update:
            d_total_round[key] += torch.tensor((client_weight_update[key] / a_i) * (list_num_samples[i] / total_sample),dtype=d_total_round[key].dtype)

        # 计算全局模型更新的加权系数
        total_coeff += a_i * (list_num_samples[i] / total_sample)

    # 使用累积的更新量更新全局模型
    updated_model = global_model.state_dict()
    for key in updated_model:
        # 根据加权的更新值调整全局模型的参数
        if updated_model[key].dtype == torch.int64:
            updated_model[key] += (total_coeff * d_total_round[key]).type(torch.int64)
        else:
            updated_model[key] += total_coeff * d_total_round[key]

    # 将更新后的模型参数加载回全局模型
    global_model.load_state_dict(updated_model)

    return True 