import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import copy
import math
import heapq
import random
import time
from models.resnet import ResNet18
from models.vgg import VGG
from models.cnn import CNN


class DatasetSplit(Dataset):
    """数据集切分工具类"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def add_trigger_bc(helper, image):
    """
    为图像添加触发器
    
    Args:
        helper: helper对象，包含config配置
        image: 输入图像
    
    Returns:
        添加触发器后的图像
    """
    pixel_max = max(1, torch.max(image))
    
    # 根据helper.config中的触发器类型添加触发器
    trigger_type = getattr(helper.config, 'trigger_type', 'square')
    trigger_x = getattr(helper.config, 'trigger_x', 0)
    trigger_y = getattr(helper.config, 'trigger_y', 0)
    
    if trigger_type == 'square':
        if hasattr(helper.config, 'dataset') and helper.config.dataset == 'cifar':
            pixel_max = 1
        image[:, trigger_y:trigger_y + 5, trigger_x:trigger_x + 5] = pixel_max
    elif trigger_type == 'pattern':
        image[:, trigger_y + 0, trigger_x + 0] = pixel_max
        image[:, trigger_y + 1, trigger_x + 1] = pixel_max
        image[:, trigger_y - 1, trigger_x + 1] = pixel_max
        image[:, trigger_y + 1, trigger_x - 1] = pixel_max
    
    return image


def test_model(model, dataset, helper, test_backdoor=True):
    """
    测试模型性能
    
    Args:
        model: 要测试的模型
        dataset: 测试数据集
        helper: helper对象
        test_backdoor: 是否测试后门性能
    
    Returns:
        acc_test: 主任务准确率
        back_acc: 后门成功率（如果test_backdoor=True）
    """
    model.eval()
    correct = 0
    backdoor_correct = 0
    total = 0
    backdoor_total = 0
    
    with torch.no_grad():
        for data, target in dataset:
            if isinstance(data, list):
                data = torch.stack(data)
            if isinstance(target, list):
                target = torch.tensor(target)
                
            data, target = data.cuda(), target.cuda()
            
            # 测试正常样本
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
            
            if test_backdoor:
                # 测试后门样本
                backdoor_data = data.clone()
                for i in range(backdoor_data.size(0)):
                    backdoor_data[i] = add_trigger_bc(helper, backdoor_data[i])
                
                backdoor_output = model(backdoor_data)
                backdoor_pred = backdoor_output.argmax(dim=1, keepdim=True)
                target_class = getattr(helper.config, 'target_class', 0)
                backdoor_target = torch.full_like(target, target_class)
                backdoor_correct += backdoor_pred.eq(backdoor_target.view_as(backdoor_pred)).sum().item()
                backdoor_total += backdoor_data.size(0)
    
    acc_test = 100. * correct / total
    back_acc = 100. * backdoor_correct / backdoor_total if test_backdoor and backdoor_total > 0 else None
    
    return acc_test, back_acc


def benign_train(model, dataset, helper):
    """
    训练良性模型的辅助函数
    
    Args:
        model: 要训练的模型
        dataset: 训练数据集
        helper: helper对象
    
    功能：使用干净数据训练模型一个epoch，用于获得良性基准模型
    """
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.5)

    model.train()
    for images, labels in train_loader:
        if isinstance(images, list):
            images = torch.stack(images)
        if isinstance(labels, list):
            labels = torch.tensor(labels)
            
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()


def malicious_train(model, dataset, helper):
    """
    训练恶意模型的辅助函数
    
    Args:
        model: 要训练的模型
        dataset: 训练数据集
        helper: helper对象
    
    功能：使用投毒数据训练模型一个epoch，在原始数据基础上添加触发器样本
    """
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    learning_rate = 0.1
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.5)

    model.train()
    for images, labels in train_loader:
        if isinstance(images, list):
            images = torch.stack(images)
        if isinstance(labels, list):
            labels = torch.tensor(labels)
            
        # 复制原始数据制作投毒样本
        bad_data, bad_label = copy.deepcopy(images), copy.deepcopy(labels)
        target_class = getattr(helper.config, 'target_class', 0)
        
        for xx in range(len(bad_data)):
            bad_label[xx] = target_class  # 修改为攻击目标标签
            bad_data[xx] = add_trigger_bc(helper, bad_data[xx])  # 添加触发器
        
        # 将投毒数据与原始数据合并
        images = torch.cat((images, bad_data), dim=0)
        labels = torch.cat((labels, bad_label))
        images, labels = images.cuda(), labels.cuda()
        
        # 正常的训练流程
        model.zero_grad()
        log_probs = model(images)
        loss = error(log_probs, labels)
        loss.backward()
        optimizer.step()


def FLS(model_benign, model_malicious, BSR, mal_val_dataset, helper):
    """
    前向层替换 (Forward Layer Substitution) 分析
    
    Args:
        model_benign: 良性模型
        model_malicious: 恶意模型
        BSR: 基准后门成功率
        mal_val_dataset: 恶意验证数据集
        helper: helper对象
    
    Returns:
        key_arr: 所有层的名称列表
        value_arr: 每层对后门性能的贡献值列表
    
    功能：
    逐层用良性模型的参数替换恶意模型的参数，测试每层对后门任务的重要性。
    value值越小（负值越大），说明该层对后门任务越重要。
    """
    bad_weight = model_malicious.state_dict()
    key_arr = []
    value_arr = []
    net3 = copy.deepcopy(model_benign)

    # 遍历每一层参数
    for key, var in model_benign.named_parameters():
        # 创建混合模型：恶意模型+当前层用良性参数
        param = copy.deepcopy(bad_weight)
        param[key] = var  # 用良性参数替换当前层
        net3.load_state_dict(param)
        
        # 测试替换后的后门性能
        acc, back_acc2 = test_model(net3, mal_val_dataset, helper, test_backdoor=True)
        key_arr.append(key)
        # 计算性能差异：替换后性能 - 原始性能
        # 负值越大说明该层越重要
        value_arr.append(back_acc2 - BSR)

    return key_arr, value_arr


def BLS(key_arr, value_arr, model_benign, model_malicious, BSR, mal_val_dataset, helper, threshold=0.8):
    """
    后向层替换 (Backward Layer Substitution) 选择攻击层
    
    Args:
        key_arr: 层名称列表
        value_arr: 层重要性值列表
        model_benign: 良性模型
        model_malicious: 恶意模型
        BSR: 基准后门成功率
        mal_val_dataset: 恶意验证数据集
        helper: helper对象
        threshold: 后门性能阈值
    
    Returns:
        attack_list: 选定的攻击层列表
    
    功能：
    从最重要的层开始，逐步添加攻击层，直到达到目标后门性能。
    这样可以用最少的层数实现有效的后门攻击。
    """
    good_weight = model_benign.state_dict()
    bad_weight = model_malicious.state_dict()
    n = 1
    temp_BSR = 0
    attack_list = []
    np_key_arr = np.array(key_arr)
    net3 = copy.deepcopy(model_benign)
    
    # 逐步增加攻击层数，直到达到目标性能
    while (temp_BSR < BSR * threshold and n <= len(key_arr)):
        # 选择前n个最重要的层（value_arr中最小的n个值）
        minValueIdx = heapq.nsmallest(n, range(len(value_arr)), value_arr.__getitem__)
        attack_list = list(np_key_arr[minValueIdx])
        
        # 创建攻击模型：良性模型+选定层用恶意参数
        param = copy.deepcopy(good_weight)
        for layer in attack_list:
            param[layer] = bad_weight[layer]  # 用恶意参数替换选定层
        net3.load_state_dict(param)
        
        # 测试当前攻击层组合的性能
        acc, temp_BSR = test_model(net3, mal_val_dataset, helper, test_backdoor=True)
        n += 1
        
    return attack_list


def split_dataset(dataset):
    """
    将数据集分割为训练集和验证集
    
    Args:
        dataset: 要分割的数据集
    
    Returns:
        mal_train_dataset: 训练数据集（75%）
        mal_val_dataset: 验证数据集（25%）
    
    功能：随机打乱数据后，将25%作为验证集，75%作为训练集
    """
    num_dataset = len(dataset)
    # 随机打乱数据顺序
    data_distribute = np.random.permutation(num_dataset)
    malicious_dataset = []
    mal_val_dataset = []
    mal_train_dataset = []
    
    for i in range(num_dataset):
        malicious_dataset.append(dataset[data_distribute[i]])
        if i < num_dataset // 4:  # 前25%作为验证集
            mal_val_dataset.append(dataset[data_distribute[i]])
        else:  # 后75%作为训练集
            mal_train_dataset.append(dataset[data_distribute[i]])
    return mal_train_dataset, mal_val_dataset


def get_attacker_dataset(helper, participant_id):
    """
    获取攻击者数据集
    
    Args:
        helper: helper对象
        participant_id: 参与者ID
    
    Returns:
        mal_train_dataset: 恶意训练数据集
        mal_val_dataset: 恶意验证数据集
    
    功能：
    为攻击者创建专用的数据集，使用指定参与者的训练数据。
    """
    # 获取参与者的训练数据
    client_dataset = []
    for data, target in helper.train_data[participant_id]:
        if isinstance(data, torch.Tensor):
            data = data.cpu()
        if isinstance(target, torch.Tensor):
            target = target.cpu().item()
        client_dataset.append((data, target))
        
    # 分割数据集为训练集和验证集
    mal_train_dataset, mal_val_dataset = split_dataset(client_dataset)
    return mal_train_dataset, mal_val_dataset


def layer_analysis(model_param, helper, participant_id, threshold=0.8):
    """
    层级分析的核心函数
    
    Args:
        model_param: 初始模型参数
        helper: helper对象
        participant_id: 参与者ID
        threshold: 性能阈值
    
    Returns:
        attack_list: 选定的攻击层列表
    
    功能：
    1. 基于给定模型参数训练良性和恶意模型
    2. 使用FLS分析各层重要性
    3. 使用BLS选择最优攻击层组合
    """
    # 根据模型类型创建相应的网络架构
    model = copy.deepcopy(helper.global_model)
    model.load_state_dict(model_param)

    # 获取攻击者数据集
    mal_train_dataset, mal_val_dataset = get_attacker_dataset(helper, participant_id)

    # ==================== 训练良性模型 ====================
    model_benign = copy.deepcopy(model)
    acc, backdoor = test_model(copy.deepcopy(model_benign), mal_train_dataset, helper)
    
    # 设置最小准确率要求
    min_acc = 80
    
    # 训练良性模型直到达到最小准确率要求
    num_time = 0
    while (acc < min_acc and num_time < 10):
        benign_train(model_benign, mal_train_dataset, helper)
        num_time += 1
        if num_time % 2 == 0:
            acc, _ = test_model(copy.deepcopy(model_benign), mal_train_dataset, helper, False)
            if num_time > 8:  # 防止无限循环
                if acc > 60:
                    break
                else:
                    return []

    # ==================== 训练恶意模型 ====================
    model_malicious = copy.deepcopy(model)
    model_malicious.load_state_dict(model.state_dict())
    malicious_train(model_malicious, mal_train_dataset, helper)

    # 测试恶意模型的后门性能
    acc, back_acc = test_model(model_malicious, mal_val_dataset, helper)
    
    if back_acc is None or back_acc < 20:  # 如果后门效果太差，返回空列表
        return []

    # ==================== 执行FLS和BLS分析 ====================
    # 前向层替换分析：找出每层的重要性
    key_arr, value_arr = FLS(model_benign, model_malicious, back_acc, mal_val_dataset, helper)
    
    # 后向层替换选择：选出最优攻击层组合
    attack_list = BLS(key_arr, value_arr, model_benign, model_malicious, back_acc, mal_val_dataset, helper,
                      threshold=threshold)
    
    return attack_list


def get_attack_layers_no_acc(model_param, helper, participant_id):
    """
    主入口函数：获取攻击层列表（不依赖准确率）
    
    Args:
        model_param: 模型参数
        helper: helper对象
        participant_id: 参与者ID
    
    Returns:
        attack_list: 选定的攻击层列表
    
    功能：
    这是LPA攻击中调用的主要函数，用于自动识别最重要的攻击层。
    该函数封装了数据集准备和层级分析的完整流程。
    """
    # 执行层级分析
    return layer_analysis(model_param, helper, participant_id)
