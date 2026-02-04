import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import logging
import random
from torch.utils.data import DataLoader

logger = logging.getLogger("logger")

def pre_process_watermark(global_model, helper, round):
    """
    预处理阶段：注入水印后门
    对应 IndicatorServer.py 中的 pre_process 函数
    """
    # 检查是否为水印检测轮次
    watermarking_rounds = [r for r in range(helper.config.watermarking_start_round,
                                          helper.config.watermarking_end_round,
                                          helper.config.watermarking_round_interval)]
    
    if round in watermarking_rounds:
        # 保存原始BN层参数
        before_wm_injection_bn_stats_dict = {}
        for key, value in global_model.state_dict().items():
            if "running_mean" in key or "running_var" in key:
                before_wm_injection_bn_stats_dict[key] = value.clone().detach()
        # print("原始BN层参数维度：", before_wm_injection_bn_stats_dict)   # ---这里正常保存了BN层参数---

        # 保存当前模型参数，用于计算距离
        target_params_variables = {}
        for name, param in global_model.state_dict().items():
            target_params_variables[name] = param.clone()
        
        # 注入水印
        _inject_watermark(global_model, helper.ood_data, target_params_variables, 
                         helper.config.watermarking_mu, helper.config)
        
        # 保存注入水印后的BN层参数
        after_wm_injection_bn_stats_dict = {}
        for key, value in global_model.state_dict().items():
            if "running_mean" in key or "running_var" in key:
                after_wm_injection_bn_stats_dict[key] = value.clone().detach()
        
        # 如果配置要求，恢复原始BN层参数
        if helper.config.replace_original_bn:
            print("----恢复原始BN层参数----")
            for key, value in global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    global_model.state_dict()[key].copy_(before_wm_injection_bn_stats_dict[key])
        
        return True, before_wm_injection_bn_stats_dict, after_wm_injection_bn_stats_dict
    
    return False, None, None

def indicator_defense(global_model, weight_accumulator, weight_accumulator_by_client, sampled_participants, helper, round):
    """
    Indicator防御方法的服务器端实现
    """
    after_wm_injection_bn_stats_dict = helper.after_wm_injection_bn_stats_dict
    watermarking_rounds = [r for r in range(helper.config.watermarking_start_round,
                                        helper.config.watermarking_end_round,
                                        helper.config.watermarking_round_interval)]
    
    if round in watermarking_rounds and after_wm_injection_bn_stats_dict is not None:
        # 使用deepcopy创建OOD数据迭代器的副本，避免迭代器耗尽问题
        fresh_ood_data = copy.deepcopy(helper.ood_data)
        # 检测恶意客户端
        benign_clients, client_asr_values = _detect_malicious_clients(
            global_model, 
            weight_accumulator_by_client,
            fresh_ood_data,  # 使用深拷贝的OOD数据迭代器
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
        
        # 根据检测结果聚合良性客户端的更新
        for name, data in global_model.state_dict().items():
            if "num_batches_tracked" in name:
                continue
            
            # 重新计算良性客户端的更新累加
            update = torch.zeros_like(data)
            for client_idx in benign_clients:
                client_update = weight_accumulator_by_client[client_idx][name]
                if client_update.dtype != update.dtype:
                    client_update = client_update.to(update.dtype)
                update.add_(client_update)
            
            # 应用聚合后的更新
            update *= (helper.config.eta / len(benign_clients))
            data.add_(update.cuda())
            
    else:
        # 非水印检测轮次，使用weight_accumulator进行常规聚合
        for name, data in global_model.state_dict().items():
            if "num_batches_tracked" in name:
                continue
            
            # 使用已经累加好的更新
            update = weight_accumulator[name] * helper.config.eta
            if update.dtype != data.dtype:
                update = update.to(data.dtype)
            data.add_(update.cuda())
    
    return True

def _inject_watermark(model, watermark_data, target_params, wm_mu, config):
    """注入水印到模型中"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.global_lr,
                               momentum=config.global_momentum,
                               weight_decay=config.global_weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[30, 60, 90],
                                                    gamma=config.global_lr_gamma)
    
    for iter in range(config.global_retrain_no_times):

        data_iterator = copy.deepcopy(watermark_data)

        for batch in data_iterator:
            optimizer.zero_grad()
            data, targets = batch
            data = data.cuda()
            targets = targets.cuda()
            
            output = model(data)
            loss = F.cross_entropy(output, targets)
            # 计算与目标参数的距离损失
            # distance_loss = _model_dist_norm_var(model, target_params)
            # loss = loss + (wm_mu/2) * distance_loss
            
            loss.backward()
            optimizer.step()
            
            # 投影步骤
            # _projection(model, target_params, config.global_projection_norm)
        scheduler.step()

        # 每隔10个轮次检查一次当前模型识别水印数据集的准确率
        if iter % 10 == 0:
            accuracy = _check_accuracy(model, watermark_data)
            print(f"第{iter}个轮次当前模型识别水印数据集的准确率: {accuracy}")

def _check_accuracy(model, watermark_data):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    # 使用深拷贝避免修改原始数据
    data_iterator = copy.deepcopy(watermark_data)
    
    with torch.no_grad():  # 禁用梯度计算以提高效率
        for batch in data_iterator:
            data, targets = batch
            data = data.cuda()
            targets = targets.cuda()
            
            # 前向传播
            outputs = model(data)
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            # 统计正确预测的数量
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    # 计算准确率
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy


def _detect_malicious_clients(model, weight_accumulator_by_client, watermark_data, threshold, bn_stats_dict, replace_bn, helper):
    """检测恶意客户端"""
    benign_clients = []
    client_asr_values = []
    check_model = copy.deepcopy(model)
    
    # 将迭代器转换为列表
    watermark_data_list = list(watermark_data)
    
    # 验证BN统计信息
    if bn_stats_dict is None:
        print("Warning: BN statistics dictionary is None")
        return [], [0.0] * len(weight_accumulator_by_client)
    
    # 获取类别数量（从helper或config中获取）
    num_classes = helper.num_classes  # 根据您的数据集设置，CIFAR10=10, CIFAR100=100
    
    # 遍历每个客户端的更新
    for client_id, client_updates in enumerate(weight_accumulator_by_client):
        # 重置检查模型到全局模型状态
        check_model.load_state_dict(model.state_dict())
        check_model.eval()
        
        # 应用客户端更新
        for name, param in check_model.named_parameters():
            if name in weight_accumulator_by_client[client_id]:
                param.data += weight_accumulator_by_client[client_id][name]
        
        # 替换BN统计信息
        if replace_bn:
            for name, module in check_model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name in bn_stats_dict:
                    module.running_mean.copy_(bn_stats_dict[name]['running_mean'])
                    module.running_var.copy_(bn_stats_dict[name]['running_var'])
        
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
        
        # 取最大准确率作为检测指标（论文伪代码第20行：αm = max({α1, α2, ..., αN})）
        max_label_acc = max(label_acc_list) if label_acc_list else 0.0
        client_asr_values.append(max_label_acc)
        
        # 如果最大标签准确率低于阈值，认为是良性客户端（论文伪代码第21行）
        if max_label_acc < threshold:
            benign_clients.append(client_id)
    
    return benign_clients, client_asr_values

def _aggregate_benign_updates(model, weight_accumulator, benign_clients, eta):
    """聚合良性客户端的更新"""
    if not benign_clients:
        return
    
    for name, data in model.state_dict().items():
        if "num_batches_tracked" in name:
            continue
            
        update = torch.zeros_like(data)
        for client_id in benign_clients:
            client_update = weight_accumulator[name][client_id]
            if client_update.dtype != update.dtype:
                client_update = client_update.to(update.dtype)
            update.add_(client_update)
        
        update *= (eta / len(benign_clients))
        data.add_(update.cuda())

def _model_dist_norm_var(model, target_params, norm=2):
    """计算模型与目标参数之间的距离"""
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    
    sum_var = torch.cuda.FloatTensor(size).fill_(0)
    size = 0
    
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params[name]).view(-1)
        size += layer.view(-1).shape[0]
    
    return torch.norm(sum_var, norm)

def _projection(model, target_params, projection_norm):
    """将模型参数投影到指定范数球内"""
    model_norm = _model_dist_norm_var(model, target_params)
    if model_norm > projection_norm:
        norm_scale = projection_norm / model_norm
        for name, param in model.named_parameters():
            clipped_difference = norm_scale * (param.data - target_params[name])
            param.data.copy_(target_params[name] + clipped_difference)

