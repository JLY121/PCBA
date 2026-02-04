import sys

from fl_utils import fler
sys.path.append("../")
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18, layer2module
import copy
import os
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Chameleon_Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()
    
    '''初始化触发器和掩码'''
    def setup(self):
        self.handcraft_rnds = 0
        #=====根据数据集的图像尺寸，初始化触发器和掩码=====
        image_size = self.helper.config.image_size  
        in_channels = self.helper.config.in_channels  
        
        # 根据数据集设置归一化后的像素值范围（原像素值255经过ToTensor和Normalize后的值）
        if self.helper.config.dataset == 'cifar10':
            # CIFAR-10: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            # 计算: (1.0 - mean) / std
            self.pixel_max = torch.tensor([2.515, 2.598, 2.754], device='cuda')  # [R, G, B]
        elif self.helper.config.dataset == 'cifar100':
            # CIFAR-100: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            self.pixel_max = torch.tensor([1.843, 2.001, 2.025], device='cuda')  # [R, G, B]
        elif self.helper.config.dataset == 'TinyImageNet':
            # TinyImageNet: mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821)
            self.pixel_max = torch.tensor([1.877, 2.050, 2.136], device='cuda')  # [R, G, B]
        elif self.helper.config.dataset == 'GTSRB':
            # GTSRB: mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629)
            self.pixel_max = torch.tensor([2.494, 2.705, 2.598], device='cuda')  # [R, G, B]
        else:  # EMNIST或其他数据集
            self.pixel_max = torch.tensor([1.0], device='cuda')  # 保持原有逻辑
            
        # 初始化基础触发器和掩码
        self.trigger = torch.zeros((1, in_channels, image_size, image_size), requires_grad=False, device='cuda')
        self.mask = torch.zeros_like(self.trigger)
        
        # 根据pattern_type生成不同的触发器模式
        self._generate_trigger_pattern()
        
        self.trigger = self.trigger.cuda()
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone() # 保存初始触发器
    
    '''第一部分：优化 触发器 和 掩码'''
    def search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        """动态生成触发器（支持扩散）"""
        pattern_type = getattr(self.helper.config, 'pattern_type', 1)
        
        if pattern_type == 2:
            # 对于复杂模式，可以动态调整扩散
            pattern_diffusion = getattr(self.helper.config, 'pattern_diffusion', 0)
            if pattern_diffusion > 0:
                # 重新生成带扩散的触发器
                self._generate_dynamic_trigger(pattern_diffusion)
        if pattern_type == 3:
            # 实现矩形块状触发器模式 (仿照Neurotoxin的矩形区域)
            self._generate_block_trigger()

        return self.trigger, self.mask

    '''第二部分：后门注入的训练过程'''
    def train_malicious(self, participant_id, model, epoch, lr):
        print("========JLY:加载对比学习模型==========")
        model_copy = self.create_model_copy(model)
        supCon_model = self.helper.supCon_model
        self.copy_common_params(supCon_model, model_copy) # 复制model的Encoder部分参数到supCon_model中
        supCon_model.train()
        lr1 = getattr(self.helper.config, 'SupCon_lr1', 0.005)
        lr2 = getattr(self.helper.config, 'SupCon_lr2', 0.001)

        # =====为Encoder设置单独的优化器=====
        SupCon_criterion = SupConLoss(temperature = 0.07)
        attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        optimizer_Encoder = torch.optim.SGD(
            supCon_model.parameters(),  # 绑定到supCon_model的参数
            lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay
        )
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr,  # 用filter函数确定优化器只更新未冻结的参数
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        clean_model = copy.deepcopy(model)
        # 1.实现Encoder训练
        for internal_epoch in range(self.helper.config.SupCon_round1):
            print("========JLY:进行Encoder训练==========")
            total_SupCon_loss = 0.0  # 初始化损失累加器
            num_batches = 0   # 初始化批次计数器
            for inputs, labels in self.helper.train_data[participant_id]:  # 已经按狄利克雷分布采样之后的每个客户端的数据
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.poison_input(inputs, labels) # 给该批次数据的前25%的数据添加触发器，并修改
                # Encoder输出
                features = supCon_model(inputs)
                SupCon_loss = SupCon_criterion(features, labels, 
                                                            poison_per_batch=int(self.helper.config.bkd_ratio * inputs.shape[0]),                                         
                                                            scale_weight=self.helper.config.scale_weight, 
                                                            down_scale_weight=self.helper.config.down_scale_weight,
                                                            fac_label=self.helper.config.target_class
                                                            ) # 计算对比损失
                optimizer_Encoder.zero_grad()
                SupCon_loss.backward()
                optimizer_Encoder.step()
            # ===输出每轮训练的SupCon_loss
                total_SupCon_loss += SupCon_loss.item()  # 累加损失
                num_batches += 1
            avg_loss = total_SupCon_loss / num_batches if num_batches > 0 else 0
            print("→ → Encoder训练轮次：", internal_epoch, "Avg SupCon_loss: ", avg_loss)
                
        # 2.将Encoder的参数复制到model中，并冻结这部分参数
        supCon_model_copy = self.create_model_copy(supCon_model)
        self.copy_freeze_common_params(model, supCon_model_copy)

        # 3.训练分类器部分
        for internal_epoch in range(self.helper.config.SupCon_round2):
            print("========JLY:进行分类器训练==========")
            total_loss = 0.0
            num_batches = 0 
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.poison_input(inputs, labels)  #====对输入数据进行后门攻击，注入的是优化后的触发器====
                output = model(inputs)
                loss = attacker_criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===用于输出分类器的训练损失===
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print("→ → 分类器训练轮次：",internal_epoch,"loss: ", loss.item())
        # 4.并解冻Encoder参数(supCon_model_copy只借用了它的参数字典的name，并不使用其参数)
        self.unfreeze_common_params(model, supCon_model_copy)
    
    # def train_malicious_close(self, participant_id, model, epoch, lr):
    #     """
    #     恶意客户端的后门攻击训练方法
    #     """
    #     # 优化器、损失函数
    #     optimizer = torch.optim.SGD(model.parameters(), 
    #                                 lr=lr,
    #                                 momentum=self.helper.config.momentum,
    #                                 weight_decay=self.helper.config.decay)
    #     criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)

    #     for internal_epoch in range(self.helper.config.attacker_retrain_times):
    #         total_loss = 0.0
    #         for inputs, labels in self.helper.train_data[participant_id]:
    #             inputs, labels = inputs.cuda(), labels.cuda()
    #             inputs, labels = self.poison_input(inputs, labels)  # 对输入数据进行后门攻击，注入优化后的触发器
    #             output = model(inputs)
    #             loss = criterion(output, labels)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()


    '''中毒数据生成过程'''
    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        
        # 对于pattern_type=2且有扩散的情况，为每个样本生成不同的触发器
        pattern_type = getattr(self.helper.config, 'pattern_type', 1)
        pattern_diffusion = getattr(self.helper.config, 'pattern_diffusion', 0)
        
        if pattern_type == 2 and pattern_diffusion > 0:
            # 为每个中毒样本生成独特的触发器
            for i in range(bkd_num):
                # 生成当前样本的触发器
                sample_trigger, sample_mask = self._generate_sample_trigger()
                # 应用触发器到单个样本
                inputs[i:i+1] = sample_trigger * sample_mask + inputs[i:i+1] * (1 - sample_mask)
        else:
            # 使用统一的触发器
            inputs[:bkd_num] = self.trigger * self.mask + inputs[:bkd_num] * (1 - self.mask)
        
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels
    
    def _generate_trigger_pattern(self):
        """根据配置生成触发器模式"""
        pattern_type = getattr(self.helper.config, 'pattern_type', 1)
        channels, height, width = self.trigger.shape[1], self.trigger.shape[2], self.trigger.shape[3]
        
        if pattern_type == 1:
            # 简单四点模式
            positions = [
                (height-3, width-3),
                (height-2, width-4), 
                (height-4, width-2),
                (height-2, width-2)
            ]
            
            for c in range(channels):
                for h, w in positions:
                    if 0 <= h < height and 0 <= w < width:
                        # 按通道设置不同的像素值
                        if len(self.pixel_max.shape) > 0 and self.pixel_max.shape[0] > 1:
                            self.trigger[0, c, h, w] = self.pixel_max[c]
                        else:
                            self.trigger[0, c, h, w] = self.pixel_max[0]
                        self.mask[0, c, h, w] = 1
                        
        elif pattern_type == 2:
            # 复杂十字模式 - 固定部分
            base_positions = [
                (height-6, width-6),
                (height-5, width-5),  # 中心点，无扩散
                (height-4, width-6),
                (height-6, width-4),
                (height-4, width-4)
            ]
            
            for c in range(channels):
                for h, w in base_positions:
                    if 0 <= h < height and 0 <= w < width:
                        # 按通道设置不同的像素值
                        if len(self.pixel_max.shape) > 0 and self.pixel_max.shape[0] > 1:
                            self.trigger[0, c, h, w] = self.pixel_max[c]
                        else:
                            self.trigger[0, c, h, w] = self.pixel_max[0]
                        self.mask[0, c, h, w] = 1

    def _generate_dynamic_trigger(self, pattern_diffusion):
        """生成带随机扩散的动态触发器"""
        # 重置触发器和掩码
        self.trigger.zero_()
        self.mask.zero_()
        
        channels, height, width = self.trigger.shape[1], self.trigger.shape[2], self.trigger.shape[3]
        change_range = 4
        
        # 生成随机扩散的位置
        positions_with_diffusion = []
        
        # 中心点（无扩散）
        positions_with_diffusion.append((height-5, width-5))
        
        # 其他点（有扩散）
        base_offsets = [
            (-6, -6), (-4, -6), (-6, -4), (-4, -4)
        ]
        
        for h_offset, w_offset in base_offsets:
            if h_offset == -6 and w_offset == -6:
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset - diffusion
                w = width + w_offset - diffusion
            elif h_offset == -4 and w_offset == -6:
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset + diffusion
                w = width + w_offset - diffusion
            elif h_offset == -6 and w_offset == -4:
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset - diffusion
                w = width + w_offset + diffusion
            else:  # (-4, -4)
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset + diffusion
                w = width + w_offset + diffusion
            
            positions_with_diffusion.append((h, w))
        
        # 设置触发器和掩码
        for c in range(channels):
            for h, w in positions_with_diffusion:
                if 0 <= h < height and 0 <= w < width:
                    # 按通道设置不同的像素值
                    if len(self.pixel_max.shape) > 0 and self.pixel_max.shape[0] > 1:
                        self.trigger[0, c, h, w] = self.pixel_max[c]
                    else:
                        self.trigger[0, c, h, w] = self.pixel_max[0]
                    self.mask[0, c, h, w] = 1
    
    def _generate_sample_trigger(self):
        """为单个样本生成带扩散的触发器"""
        channels, height, width = self.trigger.shape[1], self.trigger.shape[2], self.trigger.shape[3]
        sample_trigger = torch.zeros_like(self.trigger)
        sample_mask = torch.zeros_like(self.mask)
        
        change_range = 4
        pattern_diffusion = getattr(self.helper.config, 'pattern_diffusion', 0)
        
        # 生成位置
        positions = []
        
        # 中心点（无扩散）
        positions.append((height-5, width-5))
        
        # 其他点（有扩散）
        base_offsets = [(-6, -6), (-4, -6), (-6, -4), (-4, -4)]
        
        for h_offset, w_offset in base_offsets:
            if h_offset == -6 and w_offset == -6:
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset - diffusion
                w = width + w_offset - diffusion
            elif h_offset == -4 and w_offset == -6:
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset + diffusion
                w = width + w_offset - diffusion
            elif h_offset == -6 and w_offset == -4:
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset - diffusion
                w = width + w_offset + diffusion
            else:  # (-4, -4)
                diffusion = int(random.random() * pattern_diffusion * change_range)
                h = height + h_offset + diffusion
                w = width + w_offset + diffusion
            
            positions.append((h, w))
        
        # 设置触发器和掩码
        for c in range(channels):
            for h, w in positions:
                if 0 <= h < height and 0 <= w < width:
                    # 按通道设置不同的像素值
                    if len(self.pixel_max.shape) > 0 and self.pixel_max.shape[0] > 1:
                        sample_trigger[0, c, h, w] = self.pixel_max[c]
                    else:
                        sample_trigger[0, c, h, w] = self.pixel_max[0]
                    sample_mask[0, c, h, w] = 1
        
        return sample_trigger, sample_mask

    def _generate_block_trigger(self):
        """生成矩形块状触发器模式 (pattern_type=3)"""
        # 重置触发器和掩码
        self.trigger.zero_()
        self.mask.zero_()
        
        channels, height, width = self.trigger.shape[1], self.trigger.shape[2], self.trigger.shape[3]
        
        # 获取触发器大小配置（如果存在）
        trigger_h = getattr(self.helper.config, 'trigger_size_h', 6)
        trigger_w = getattr(self.helper.config, 'trigger_size_w', 6)
        
        # 确保触发器大小不超过图像尺寸
        trigger_h = min(trigger_h, height - 2)
        trigger_w = min(trigger_w, width - 2)
        
        # 设置触发器位置（左上角区域，留2像素边距）
        start_h = 2
        start_w = 2
        end_h = start_h + trigger_h
        end_w = start_w + trigger_w
        
        # 为所有通道设置矩形块触发器
        for c in range(channels):
            # 按通道设置不同的像素值
            if len(self.pixel_max.shape) > 0 and self.pixel_max.shape[0] > 1:
                self.trigger[0, c, start_h:end_h, start_w:end_w] = self.pixel_max[c]
            else:
                self.trigger[0, c, start_h:end_h, start_w:end_w] = self.pixel_max[0]
            self.mask[0, c, start_h:end_h, start_w:end_w] = 1
        
        print(f"Pattern Type 3: 生成 {trigger_h}x{trigger_w} 矩形块触发器，位置: ({start_h}:{end_h}, {start_w}:{end_w})")

    def create_model_copy(self, model):
        model_copy = dict()
        for name, param in model.named_parameters():
            model_copy[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
        return model_copy

    '''=====实现对比学习模型参数的复制====='''
    def copy_common_params(self, SupCon_model, target_params_variables):
        for name, layer in SupCon_model.named_parameters():  
            if name in target_params_variables:
                layer.data = copy.deepcopy(target_params_variables[name])

    '''====实现Encoder参数复制并冻结====='''
    def copy_freeze_common_params(self, model, target_params_variables, Freeze = None):
        for name, layer in model.named_parameters():
            if name in target_params_variables:
                layer.data = copy.deepcopy(target_params_variables[name])
                layer.requires_grad = False

    '''=====分类器训练结束后解冻参数====='''
    def unfreeze_common_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            if name in target_params_variables:
                layer.requires_grad = True


# ===有监督对比损失函数====
class SupConLoss(nn.Module):
    """有监督对比学习损失函数：
    同时支持SimCLR的无监督对比损失"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """初始化函数
        
        参数:
            temperature: 温度系数，默认为0.07，用于调节logits的分布平滑度
            contrast_mode: 对比模式，'all'表示所有特征作为锚点，'one'表示仅使用第一个视图
            base_temperature: 基础温度系数，默认为0.07，用于最终损失的缩放
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature       # 温度系数，用于控制对比损失的分布
        self.contrast_mode = contrast_mode   # 对比模式，决定锚点特征的选择方式
        self.base_temperature = base_temperature   # 基础温度系数，用于损失计算的缩放

    def forward(self, features, labels=None, mask=None, poison_per_batch=None, scale_weight=1, down_scale_weight=1, fac_label=0):
        """前向传播函数，计算模型的对比损失。如果labels和mask都为None，则退化为SimCLR的无监督损失
        
        参数:
            features: 输入的隐藏特征向量，形状为[bsz, n_views, ...]，bsz为批次大小，n_views为视图数
            labels: 真实标签，形状为[bsz]，用于有监督对比学习
            mask: 对比掩码矩阵，形状为[bsz, bsz]，若样本i和j同类则mask[i,j]=1
            poison_per_batch: 每个批次中的毒化样本数量，默认为None
            poison_images_len: 毒化图像的长度，默认为None
            scale_weight: 特定类别的缩放权重，默认为1
            down_scale_weight: 降权系数，默认为1，用于毒化样本或特定标签的权重调整

            helper: 辅助对象，包含特定标签信息（如fac_label），默认为None
        
        返回:
            loss: 计算得到的损失标量值
        """
        # 确定设备类型（GPU或CPU）
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # 获取批次大小
        batch_size = features.shape[0]  

        # 处理标签和掩码的互斥逻辑   →→ 这里的掩码指的是标记同类样本对的掩码
        if labels is not None and mask is not None:
            raise ValueError('不能同时定义`labels`和`mask`')  # 标签和掩码不能同时提供
        elif labels is None and mask is None:  # 无监督模式（SimCLR）
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 创建对角线为1的单位矩阵作为掩码
        elif labels is not None:  # 有监督对比学习
            labels = labels.contiguous().view(-1, 1)  # 将标签重塑为列向量[bsz, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('标签数量与特征数量不匹配')  # 检查标签数量是否与批次大小一致
            # 生成同类掩码矩阵，比较labels的每一行与每一列，若相等则为1 =====
            mask = torch.eq(labels, labels.T).float().to(device)
            #====================以下部分是chameleon新添的===============
            # 创建用于缩放权重的掩码副本
            mask_scale = mask.clone().detach()
            # 创建特征对比的掩码，初始化为全1矩阵   ？？？？
            mask_cross_feature = torch.ones_like(mask_scale).to(device)
            
            # 处理毒化样本和特定标签的权重调整
            label_flatten = labels.view(-1)  # 将标签展平为1维向量
            for ind, label in enumerate(label_flatten):
                # 对特定标签（由helper.fac_label指定）的样本进行权重缩放
                if label == fac_label:  # =====目标标签====
                    mask_scale[ind, :] = mask[ind, :] * scale_weight  # =====该权重值为6=====
                # 对毒化样本（前poison_per_batch*poison_images_len个样本）进行降权处理
                if ind < poison_per_batch:  # 1 * 7，每个批次毒化的样本数
                    # 对标签为1的样本应用降权系数，其他标签保持不变（毒化样本与标签为1的样本之间的相似度被降权）
                    label_1_row_eq = torch.eq(label_flatten, 1).float().to(device) * down_scale_weight # 找标签为1的样本，并乘以降权系数
                    label_1_row_nq = torch.ne(label_flatten, 1).float().to(device)                     # 找标签不为1的样本，正常给权重  
                    label_1_row = label_1_row_eq + label_1_row_nq
                    mask_cross_feature[ind, :] = mask_cross_feature[ind, :] * label_1_row
                # 对标签为1的样本与毒化样本的交互特征进行降权（标签为1的样本与毒化样本之间的相似度也被降权）
                if label == 1:
                    mask_cross_feature[ind, 0:poison_per_batch] = \
                        mask_cross_feature[ind, 0:poison_per_batch] * down_scale_weight
                # =============================================
        else:
            mask = mask.float().to(device)  # 如果只提供了mask，则转换为浮点型并移到指定设备

        # 设置对比特征
        contrast_feature = features
        # 根据contrast_mode选择锚点特征
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # 仅使用第一个视图作为锚点
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = features  # 使用所有特征作为锚点
        else:
            raise ValueError('未知的对比模式: {}'.format(self.contrast_mode))

        # 计算logits：锚点特征与对比特征的点积，除以温度系数，并应用mask_cross_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature
        # 为数值稳定性，减去每行的最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 创建logits掩码，排除自身对比（对角线置0）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # 应用掩码到mask和mask_scale，排除自身对比
        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask

        # 计算log probability
        exp_logits = torch.exp(logits) * logits_mask  # 计算指数logits并应用掩码
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 计算log概率

        # 计算正样本的平均log probability
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)  # 对正样本的log概率加权求和
        mask_check = mask.sum(1)  # 计算每个锚点的正样本数量
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue  # 如果没有正样本，跳过
            else:
                mask_check[ind] = 1 / mask_item  # 计算正样本数量的倒数
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply  # 应用正样本数量归一化

        # 计算最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # 缩放并取负值
        loss = loss.view(batch_size).mean()  # 对批次内的损失取平均值

        return loss  # 返回损失标量值
