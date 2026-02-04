import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import logging

logger = logging.getLogger("logger")

def pre_process_watermark(global_model, helper, round):  
    # 确定水印检测轮次范围
    watermarking_rounds = [r for r in range(helper.config.watermarking_start_round,
                                          helper.config.watermarking_end_round,
                                          helper.config.watermarking_round_interval)]
    
    # 如果当前轮次不在水印检测轮次内，直接返回
    if round not in watermarking_rounds:
        return False, None, None
    
    # print(f"开始在轮次 {round} 进行水印注入预处理")
    
    # 使用水印数据进行初始测试
    wm_data = copy.deepcopy(helper.ood_data)
    loss_w, acc_w, label_acc_w, label_ind, wm_label_acc_list, wm_label_dict = _global_watermarking_test_sub(
    test_data=wm_data, model=global_model, helper=helper)
    # print(f"注入前水印测试 - 准确率:{acc_w:.2f}%, 损失:{loss_w:.4f}, 目标标签({label_ind})准确率:{label_acc_w:.2f}%")
    # print(f"注入前各标签准确率: {wm_label_acc_list}")
    # print(f"注入前预测分布: {wm_label_dict}")
    
    # 初始化目标参数变量（用于计算距离损失）
    target_params_variables = dict()
    for name, param in global_model.state_dict().items():
        target_params_variables[name] = param.clone()
    
    # 保存注入水印前的BN层参数
    before_wm_injection_bn_stats_dict = dict()
    for key, value in global_model.state_dict().items():
        if "running_mean" in key or "running_var" in key:
            before_wm_injection_bn_stats_dict[key] = value.clone().detach()
    
    # print("开始进行良性水印注入")
    
    # 执行水印注入
    wm_data = copy.deepcopy(helper.ood_data)
    _global_watermark_injection(watermark_data=wm_data,
                               target_params_variables=target_params_variables,
                               model=global_model,
                               helper=helper,
                               round=round)
    
    # 计算水印注入后的参数更新范数
    watermarking_update_norm = _model_dist_norm(global_model, target_params_variables)
    print(f"水印注入后的参数更新范数: {watermarking_update_norm:.6f}")
    
    # 使用水印数据测试注入后的效果
    wm_data = copy.deepcopy(helper.ood_data)
    loss_w, acc_w, label_acc_w, label_ind, wm_label_acc_list, wm_label_dict = _global_watermarking_test_sub(
    test_data=wm_data, model=global_model, helper=helper)
    print(f"注入后水印测试 - 准确率:{acc_w:.2f}%, 损失:{loss_w:.4f}, 目标标签({label_ind})准确率:{label_acc_w:.2f}%")
    print(f"注入后各标签准确率: {wm_label_acc_list}")
    # print(f"注入后预测分布: {wm_label_dict}")
    
    # 保存注入水印后的BN层参数
    after_wm_injection_bn_stats_dict = dict()
    for key, value in global_model.state_dict().items():
        if "running_mean" in key or "running_var" in key:
            after_wm_injection_bn_stats_dict[key] = value.clone().detach()
    
    # 如果配置要求，恢复原始BN层参数到全局模型
    if helper.config.replace_original_bn:
        print("恢复全局模型的原始BN层参数")
        for key, value in global_model.state_dict().items():
            if "running_mean" in key or "running_var" in key:
                global_model.state_dict()[key].copy_(before_wm_injection_bn_stats_dict[key])
    
    # print(f"轮次 {round} 的水印注入预处理完成")
    
    return True, before_wm_injection_bn_stats_dict, after_wm_injection_bn_stats_dict


def _global_watermarking_test_sub(test_data, model, helper):
    """
    水印数据测试子函数
    参照 IndicatorServer.py 中的 _global_watermarking_test_sub 函数实现
    
    Args:
        test_data: 测试数据迭代器
        model: 要测试的模型
        helper: 助手对象
        
    Returns:
        total_l: 平均损失
        watermark_acc: 总体水印准确率
        wm_label_acc: 最佳标签准确率
        wm_index_label: 最佳标签索引
        wm_label_acc_list: 各标签准确率列表
        wm_label_dict: 预测分布字典
    """
    model.eval()
    total_loss = 0
    dataset_size = 0
    correct = 0
    data_iterator = test_data
    
    # 初始化各标签的统计信息
    num_classes = helper.num_classes
    wm_label_sum_list = [0 for i in range(num_classes)]
    wm_label_correct_list = [0 for i in range(num_classes)]
    wm_label_acc_list = [0 for i in range(num_classes)]
    wm_label_dict = dict()
    for i in range(num_classes):
        wm_label_dict[i] = 0
    
    with torch.no_grad():
        for batch_id, batch in enumerate(data_iterator):
            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)
            
            output = model(data)
            total_loss += F.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            
            # 统计预测分布
            for pred_item in pred:
                wm_label_dict[pred_item.item()] += 1
            
            # 为每个标签分别计算准确率统计
            for target_label in range(num_classes):
                wm_label_targets = torch.ones_like(targets) * target_label
                wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))
                
                wm_label_sum_list[target_label] += wm_label_index.cpu().sum().item()
                wm_label_correct_list[target_label] += pred.eq(targets.data.view_as(pred))[wm_label_index.bool()].cpu().sum().item()
            
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            dataset_size += len(targets)
    
    # 计算总体水印准确率
    watermark_acc = 100.0 * (float(correct) / float(dataset_size))
    
    # 计算预测分布比例
    for i in range(num_classes):
        wm_label_dict[i] = round(wm_label_dict[i] / dataset_size, 2)
    
    # 计算各标签的准确率
    for target_label in range(num_classes):
        if wm_label_sum_list[target_label] > 0:
            wm_label_acc_list[target_label] = round(100.0 * (float(wm_label_correct_list[target_label]) / float(wm_label_sum_list[target_label])), 2)
        else:
            wm_label_acc_list[target_label] = 0.0
    
    # 找到最佳标签准确率和对应的索引
    wm_label_acc = max(wm_label_acc_list) if wm_label_acc_list else 0.0
    wm_index_label = wm_label_acc_list.index(wm_label_acc) if wm_label_acc_list else 0
    total_l = total_loss / dataset_size
    
    model.train()
    return (total_l, watermark_acc, wm_label_acc, wm_index_label, wm_label_acc_list, wm_label_dict)


def _global_watermark_injection(watermark_data, target_params_variables, model, helper, round=None):
    """
    全局水印注入函数
    参照 IndicatorServer.py 中的 _global_watermark_injection 函数实现
    
    Args:
        watermark_data: 水印数据迭代器
        target_params_variables: 目标参数变量（用于计算距离损失）
        model: 要注入水印的模型
        helper: 助手对象
        round: 当前轮次
    """
    model.train()
    
    # 设置优化器
    optimizer = torch.optim.SGD(model.parameters(), 
                               lr=helper.config.global_lr,
                               momentum=helper.config.global_momentum,
                               weight_decay=helper.config.global_weight_decay)
    
    # 设置学习率调度器
    milestones = getattr(helper.config, 'global_milestones', [30, 60, 90])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=milestones,
                                                    gamma=helper.config.global_lr_gamma)
    
    # print(f"水印注入参数: mu={helper.config.watermarking_mu}, 训练轮次={helper.config.global_retrain_no_times}")
    
    retrain_no_times = helper.config.global_retrain_no_times
    total_loss = 0
    
    for internal_round in range(retrain_no_times):
        # if internal_round % 50 == 0:
        #     print(f"水印注入内部轮次: {internal_round}")
        
        data_iterator = copy.deepcopy(watermark_data)
        
        for batch_id, watermark_batch in enumerate(data_iterator):
            optimizer.zero_grad()
            wm_data, wm_targets = watermark_batch
            wm_data = wm_data.cuda().detach().requires_grad_(False)
            wm_targets = wm_targets.cuda().detach().requires_grad_(False)
            
            output = model(wm_data)
            pred = output.data.max(1)[1]
            
            # 计算分类损失和距离损失
            class_loss = F.cross_entropy(output, wm_targets)
            distance_loss = _model_dist_norm_var(model, target_params_variables)
            loss = class_loss + (helper.config.watermarking_mu / 2) * distance_loss
            
            loss.backward()
            optimizer.step()
            
            # 执行投影步骤
            _projection(model, target_params_variables, helper.config.global_projection_norm)
            total_loss += loss.data
            
            # 在最后一轮的第一个批次进行测试输出
            if internal_round == retrain_no_times - 1 and batch_id == 0:
                # 测试良性准确率（这里简化处理，实际应该有测试数据）
                print(f"轮次{internal_round} | 良性测试完成")
                
                # 测试水印效果
                wm_data_test = copy.deepcopy(helper.ood_data)
                loss_w, acc_w, label_acc_w, label_ind, _, _ = _global_watermarking_test_sub(
                    test_data=wm_data_test, model=model, helper=helper)
                print(f"水印准确率:{acc_w:.2f}%, 水印损失:{loss_w:.4f}, 目标标签({label_ind})准确率:{label_acc_w:.2f}%")
        
        scheduler.step()
    
    return True


def _model_dist_norm(model, target_params):
    """
    计算模型参数与目标参数的欧几里得距离
    参照 IndicatorServer.py 中的 _model_dist_norm 函数实现
    
    Args:
        model: 当前模型
        target_params: 目标参数字典
        
    Returns:
        distance: 欧几里得距离
    """
    squared_sum = 0
    for name, layer in model.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
    return math.sqrt(squared_sum)


def _model_dist_norm_var(model, target_params_variables, norm=2):
    """
    计算模型参数与目标参数变量的范数距离
    参照 IndicatorServer.py 中的 _model_dist_norm_var 函数实现
    
    Args:
        model: 当前模型
        target_params_variables: 目标参数变量字典
        norm: 范数类型，默认为2范数
        
    Returns:
        distance: 范数距离
    """
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]
    
    return torch.norm(sum_var, norm)


def _projection(model, target_params_variables, projection_norm):
    """
    将模型参数投影到指定范数球内
    参照 IndicatorServer.py 中的 _projection 函数实现
    
    Args:
        model: 要投影的模型
        target_params_variables: 目标参数变量
        projection_norm: 投影范数约束
    """
    model_norm = _model_dist_norm_var(model, target_params_variables)
    if model_norm > projection_norm:
        norm_scale = projection_norm / model_norm
        for name, param in model.named_parameters():
            clipped_difference = norm_scale * (param.data - target_params_variables[name])
            param.data.copy_(target_params_variables[name] + clipped_difference)


def indicator_defense(global_model, weight_accumulator, weight_accumulator_by_client, sampled_participants, helper, round):
    """
    Indicator防御方法的服务器端聚合实现
    参照 avg.py 文件的形式实现
    
    Args:
        global_model: 全局模型
        weight_accumulator: 权重累加器
        weight_accumulator_by_client: 按客户端分组的权重累加器
        sampled_participants: 采样的参与者列表
        helper: 助手对象
        round: 当前轮次
        
    Returns:
        bool: 聚合是否成功
    """
    # 获取水印注入后的BN统计信息
    after_wm_injection_bn_stats_dict = helper.after_wm_injection_bn_stats_dict
    
    # 确定水印检测轮次范围
    watermarking_rounds = [r for r in range(helper.config.watermarking_start_round,
                                          helper.config.watermarking_end_round,
                                          helper.config.watermarking_round_interval)]
    
    # 如果是水印检测轮次且有BN统计信息，进行恶意客户端检测
    if round in watermarking_rounds and after_wm_injection_bn_stats_dict is not None:
        print(f"轮次 {round}: 开始进行恶意客户端检测")
        
        # 使用深拷贝创建OOD数据迭代器的副本，避免迭代器耗尽问题
        fresh_ood_data = copy.deepcopy(helper.ood_data)
        
        # 检测恶意客户端
        benign_clients, client_asr_values = _detect_malicious_clients(
            global_model,
            weight_accumulator_by_client,
            fresh_ood_data,
            helper.config.VWM_detection_threshold,
            after_wm_injection_bn_stats_dict,
            helper.config.replace_original_bn,
            helper
        )
        
        # 输出检测结果
        num_clients = len(client_asr_values)
        all_clients = list(range(num_clients))
        malicious_clients = [c for c in all_clients if c not in benign_clients]
        
        print(f"本轮客户端水印攻击成功率 (ASR): {[f'{asr:.2f}%' for asr in client_asr_values]}")
        print(f"本轮检测到的恶意客户端: {malicious_clients}")
        print(f"本轮良性客户端: {benign_clients}")
        
        # 检查是否有良性客户端参与聚合
        if len(benign_clients) == 0:
            print("警告: 没有检测到良性客户端，跳过本轮聚合")
            return True
        
        # 根据检测结果聚合良性客户端的更新
        for name, data in global_model.state_dict().items():
            if "num_batches_tracked" in name:
                continue
            
            # 重新计算良性客户端的更新累加
            update = torch.zeros_like(data)
            for client_idx in benign_clients:
                if client_idx < len(weight_accumulator_by_client):
                    client_update = weight_accumulator_by_client[client_idx][name]
                    if client_update.dtype != update.dtype:
                        client_update = client_update.to(update.dtype)
                    update.add_(client_update)
            
            # 应用聚合后的更新
            update *= (helper.config.eta / len(benign_clients))
            data.add_(update.cuda())
    
    else:
        # 非水印检测轮次，使用常规FedAvg聚合
        # print(f"轮次 {round}: 使用常规FedAvg聚合")
        for name, data in global_model.state_dict().items():
            if "num_batches_tracked" in name:
                continue
            
            # 使用正确的FedAvg聚合公式：累积更新 / 参与者数量 * 学习率
            update = weight_accumulator[name] * (1.0 / helper.config.num_sampled_participants) * helper.config.eta
            if update.dtype != data.dtype:
                update = update.to(data.dtype)
            data.add_(update.cuda())
    
    return True


def _detect_malicious_clients(model, weight_accumulator_by_client, watermark_data, threshold, bn_stats_dict, replace_bn, helper):
    """
    检测恶意客户端
    参照原版实现，通过水印检测来识别恶意客户端
    
    Args:
        model: 全局模型
        weight_accumulator_by_client: 按客户端分组的权重累加器
        watermark_data: 水印数据
        threshold: 检测阈值
        bn_stats_dict: BN层统计信息
        replace_bn: 是否替换BN层参数
        helper: 助手对象
        
    Returns:
        benign_clients: 良性客户端列表
        client_asr_values: 各客户端的ASR值列表
    """
    benign_clients = []
    client_asr_values = []
    check_model = copy.deepcopy(model)
    
    # 将迭代器转换为列表
    watermark_data_list = list(watermark_data)
    
    # 验证BN统计信息
    if bn_stats_dict is None:
        print("警告: BN统计信息字典为None")
        return [], [0.0] * len(weight_accumulator_by_client)
    
    # 获取类别数量
    num_classes = helper.num_classes
    
    # 遍历每个客户端的更新
    for client_id, client_updates in enumerate(weight_accumulator_by_client):
        # 重置检查模型到全局模型状态
        check_model.load_state_dict(model.state_dict())
        check_model.eval()
        
        # 按照原版逻辑应用客户端更新
        for name, data in check_model.state_dict().items():
            if "num_batches_tracked" in name:
                continue
            if "running_mean" in name or "running_var" in name:
                # BN层统计信息处理
                if replace_bn and name in bn_stats_dict:
                    # 如果启用BN替换，使用水印注入后的BN统计信息
                    check_model.state_dict()[name].copy_(bn_stats_dict[name])
                # 否则保持当前BN统计信息不变
            else:
                # 可训练参数：应用客户端更新
                if name in client_updates:
                    check_model.state_dict()[name].add_(client_updates[name])

        # 使用水印数据测试模型 - 按标签计算准确率
        check_model.eval()
        
        # 初始化每个标签的统计信息
        label_correct_list = [0] * num_classes
        label_sum_list = [0] * num_classes
        
        with torch.no_grad():
            for batch in watermark_data_list:
                data, targets = batch
                data = data.cuda()
                targets = targets.cuda()
                
                output = check_model(data)
                pred = output.data.max(1)[1]
                
                # 为每个标签分别计算准确率
                for target_label in range(num_classes):
                    # 创建目标标签掩码
                    target_mask = targets.eq(target_label)
                    
                    # 统计该标签的样本数量
                    label_sum_list[target_label] += target_mask.sum().item()
                    
                    # 统计该标签的正确预测数量
                    if target_mask.sum() > 0:
                        label_correct_list[target_label] += pred.eq(targets)[target_mask].sum().item()
        
        # 计算每个标签的准确率
        label_acc_list = []
        for target_label in range(num_classes):
            if label_sum_list[target_label] > 0:
                acc = 100.0 * label_correct_list[target_label] / label_sum_list[target_label]
                label_acc_list.append(acc)
            else:
                label_acc_list.append(0.0)
        
        # 取最大准确率作为检测指标
        # max_label_acc = max(label_acc_list) if label_acc_list else 0.0
        # client_asr_values.append(max_label_acc)
        # ---尝试使用总的准确率作为检测指标---
        total_acc = sum(label_acc_list) / num_classes
        client_asr_values.append(total_acc)
        
        # 如果最大标签准确率低于阈值，认为是良性客户端
        # if max_label_acc < threshold:
        if total_acc < threshold:
            benign_clients.append(client_id)
    
    return benign_clients, client_asr_values
