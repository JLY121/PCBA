import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets, transforms
from collections import defaultdict, OrderedDict
import random
import numpy as np
import copy
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fl_utils.attacker.BC_Layer_utils import (
    add_trigger_bc, 
    get_attack_layers_no_acc,
    get_attacker_dataset,
    test_model,
    benign_train,
    malicious_train
)


class BC_Layer_Attacker:
    """
    BC_Layer攻击器 - 实现层级投毒攻击 (Layer-wise Poisoning Attack, LPA)
    
    LPA的核心思想:
    1. 分别训练一个恶意模型和一个良性模型
    2. 计算两者参数的差异
    3. 识别出对后门任务最重要的层
    4. 在良性模型的基础上，将这些重要层的参数向恶意模型的方向进行增强
    """
    
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.malicious_model = None
        self.benign_model = None
        self.attack_list = []
        self.setup()
    
    def setup(self):
        """初始化攻击器参数"""
        self.handcraft_rnds = 0
        # 设置触发器相关参数
        self.trigger_type = getattr(self.helper.config, 'trigger_type', 'square')
        self.trigger_x = getattr(self.helper.config, 'trigger_x', 0)
        self.trigger_y = getattr(self.helper.config, 'trigger_y', 0)
        self.target_class = getattr(self.helper.config, 'target_class', 0)
        self.poison_frac = getattr(self.helper.config, 'bkd_ratio', 0.1)
        
    def search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        """
        触发器搜索（在BC_Layer中不需要优化触发器，使用固定触发器）
        
        Returns:
            trigger: 固定触发器
            mask: 固定掩码
        """
        # BC_Layer使用固定触发器，不需要优化
        image_size = self.helper.config.image_size  
        in_channels = self.helper.config.in_channels  
        trigger = torch.ones((1, in_channels, image_size, image_size), requires_grad=False, device='cuda') * 2
        mask = torch.zeros_like(trigger)
        mask[:, :, self.trigger_y:self.trigger_y+5, self.trigger_x:self.trigger_x+5] = 1
        mask = mask.cuda()
        return trigger, mask
    
    def train_malicious(self, participant_id, model, epoch, lr):
        """
        BC_Layer的恶意训练过程 - 实现LPA攻击
        
        Args:
            participant_id: 参与者ID
            model: 全局模型
            epoch: 当前轮次
            lr: 学习率
        """
        # 保存初始的全局模型参数
        good_param = copy.deepcopy(model.state_dict())
        
        # ==================== 1. 训练恶意模型 ====================
        badnet = copy.deepcopy(model)
        badnet.train()
        optimizer = torch.optim.SGD(
            badnet.parameters(), 
            lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        
        # 训练恶意模型
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                # 对输入数据进行后门攻击，注入触发器
                inputs, labels = self.poison_input(inputs, labels)
                
                output = badnet(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 保存训练好的恶意模型参数
        bad_net_param = badnet.state_dict()
        self.malicious_model = copy.deepcopy(badnet)
        
        # ==================== 2. 训练良性模型 ====================
        net = copy.deepcopy(model)
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), 
            lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay
        )
        
        # 训练良性模型（使用干净数据）
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                # 使用干净数据进行正常训练
                
                output = net(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.benign_model = copy.deepcopy(net)
        
        # ==================== 3. 制作层级投毒攻击模型 ====================
        attack_param = {}
        
        # 使用FLS/BLS等技术识别对后门任务最重要的层
        try:
            attack_list = get_attack_layers_no_acc(
                copy.deepcopy(net.state_dict()), 
                self.helper, 
                participant_id
            )
            self.attack_list = attack_list
        except Exception as e:
            print(f"Layer analysis failed: {e}")
            # 如果层级分析失败，使用预设的攻击层或返回原模型
            attack_list = getattr(self.helper.config, 'attack_layers', [])
            self.attack_list = attack_list
        
        print(f'BC_Layer attack_list: {attack_list}')
        
        # 遍历所有模型参数
        for key, var in net.state_dict().items():
            if key in attack_list:
                # 如果是攻击层
                # 计算良性模型和恶意模型在该层的参数差异
                difference = (bad_net_param[key] - good_param[key])
                x = 1  # 放大因子
                # 将良性模型的参数向恶意模型的方向进行增强
                attack_param[key] = good_param[key] + x * difference
            else:
                # 如果不是攻击层，保持良性模型的参数不变
                attack_param[key] = var
        
        # 将攻击参数加载到模型中
        model.load_state_dict(attack_param)
    
    def poison_input(self, inputs, labels, eval=False):
        """
        中毒数据生成过程
        
        Args:
            inputs: 输入数据
            labels: 标签
            eval: 是否为评估模式
            
        Returns:
            投毒后的输入数据和标签
        """
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.poison_frac * inputs.shape[0])
        
        # 为前bkd_num个样本添加触发器并修改标签
        for i in range(bkd_num):
            inputs[i] = add_trigger_bc(self.helper, inputs[i])
            labels[i] = self.target_class
            
        return inputs, labels
    
    def trigger_data(self, images, labels):
        """
        数据投毒函数：为训练数据添加触发器并修改标签
        
        Args:
            images: 原始图像数据
            labels: 原始标签数据
        
        Returns:
            images: 投毒后的图像数据
            labels: 投毒后的标签数据
        """
        # 根据投毒比例确定要投毒的样本数量
        poison_num = int(len(images) * self.poison_frac)
        
        # 复制数据制作投毒样本
        bad_data, bad_label = copy.deepcopy(images), copy.deepcopy(labels)
        
        # 为前poison_num个样本添加触发器并修改标签
        for xx in range(poison_num):
            bad_label[xx] = self.target_class  # 修改标签为攻击目标
            bad_data[xx] = add_trigger_bc(self.helper, bad_data[xx])  # 添加触发器
        
        # 将投毒数据与原始数据合并
        images = torch.cat((images, bad_data[:poison_num]), dim=0)
        labels = torch.cat((labels, bad_label[:poison_num]))
        
        return images, labels
    
    def get_attack_info(self):
        """
        获取攻击信息
        
        Returns:
            包含攻击信息的字典
        """
        return {
            'attack_type': 'BC_Layer_LPA',
            'attack_list': self.attack_list,
            'malicious_model': self.malicious_model.state_dict() if self.malicious_model else None,
            'benign_model': self.benign_model.state_dict() if self.benign_model else None,
            'trigger_type': self.trigger_type,
            'target_class': self.target_class,
            'poison_fraction': self.poison_frac
        }
    
    def test_backdoor(self, model, test_data):
        """
        测试后门攻击效果
        
        Args:
            model: 要测试的模型
            test_data: 测试数据
            
        Returns:
            后门成功率
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data = data.cuda()
                # 添加触发器
                data = add_trigger_bc(self.helper, data)
                
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                target_tensor = torch.full_like(pred.squeeze(), self.target_class).cuda()
                correct += pred.eq(target_tensor.view_as(pred)).sum().item()
                total += data.size(0)
        
        backdoor_acc = 100. * correct / total if total > 0 else 0
        return backdoor_acc
