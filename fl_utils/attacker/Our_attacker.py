from __future__ import print_function
import sys
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

class Our_Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()

    def setup(self):
        self.handcraft_rnds = 0
        #=====根据数据集的图像尺寸，初始化触发器和掩码=====
        image_size = self.helper.config.image_size  #指定图像尺寸
        in_channels = self.helper.config.in_channels  #指定输入通道数
        self.trigger = torch.ones((1,in_channels,image_size,image_size), requires_grad=False, device = 'cuda')*0.5
        self.mask = torch.zeros_like(self.trigger) #触发器的掩码
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone() # 保存初始触发器
    
    def get_adv_model(self, model, dl, trigger, mask):
        """
        获取对抗全局模型 (adversarial global model)，通过本地数据集和触发器对全局模型进行训练。
        计算对抗模型与原始模型之间的梯度相似性，以衡量模型对触发器的鲁棒性。

        参数:
        - model: 当前全局模型
        - dl: 数据加载器，包含本地训练数据
        - trigger: 当前触发器模式
        - mask: 触发器掩码，用于指定触发器的位置

        返回:
        - adv_model: 对抗全局模型
        - sim_sum/sim_count: 对抗模型和原始模型梯度之间的余弦相似性
        """
        adv_model = copy.deepcopy(model)  # 深拷贝当前模型，创建对抗模型
        adv_model.train()  # 切换模型到训练模式
        ce_loss = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        adv_opt = torch.optim.SGD(  # 定义对抗模型的优化器，使用 SGD
            adv_model.parameters(),
            lr=0.01,  # 学习率
            momentum=0.9,  # 动量
            weight_decay=5e-4  # 权重衰减，防止过拟合
        )

        # 对抗模型的训练过程
        for _ in range(self.helper.config.dm_adv_epochs):  # 对抗训练的迭代次数
            for inputs, labels in dl:  # 遍历本地数据加载器
                inputs, labels = inputs.cuda(), labels.cuda()  # 将数据加载到 GPU
                # 应用触发器模式到输入数据
                inputs = trigger * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)  # 通过对抗模型进行前向传播
                loss = ce_loss(outputs, labels)  # 计算损失（目标是训练对抗模型识别触发器样本）
                adv_opt.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                adv_opt.step()  # 更新对抗模型的参数

        # 计算对抗模型和原始模型的梯度余弦相似性
        # 原始实现仅依据参数名中是否包含 'conv' 来筛选卷积层（适用于 ResNet），
        # 在 VGG 等网络中卷积层一般命名为 features.*.weight，不含 'conv'，会导致 sim_count 为 0。
        sim_sum = 0.  # 累积相似性总和
        sim_count = 0.  # 累积的计算次数
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)  # 定义余弦相似性计算

        model_params_dict = dict(model.named_parameters())
        adv_params_dict = dict(adv_model.named_parameters())

        # 第一轮：优先只用“卷积层”权重
        # - ResNet: 名称中包含 'conv'
        # - VGG 等: 参数梯度为 4 维张量（[out_c, in_c, kH, kW]）
        for name, p_adv in adv_model.named_parameters():  # 遍历对抗模型的所有参数
            if name not in model_params_dict:
                continue
            p_org = model_params_dict[name]
            if p_adv.grad is None or p_org.grad is None:
                continue

            is_conv_like = ('conv' in name) or (p_adv.grad.dim() == 4)  # VGG的对应卷积层是4维张量，可以作为判别条件
            if not is_conv_like:
                continue

            sim_count += 1
            sim_sum += cos_loss(p_adv.grad.reshape(-1), p_org.grad.reshape(-1))

        # 若未找到任何“卷积层”，则退而求其次：对所有有梯度的参数做平均相似性，避免除零
        if sim_count == 0:
            for name, p_adv in adv_model.named_parameters():
                if name not in model_params_dict:
                    continue
                p_org = model_params_dict[name]
                if p_adv.grad is None or p_org.grad is None:
                    continue
                sim_count += 1
                sim_sum += cos_loss(p_adv.grad.reshape(-1), p_org.grad.reshape(-1))

        # 理论上此时 sim_count > 0；若极端情况下仍为 0，则返回 0 相似性以避免报错
        if sim_count == 0:
            avg_sim = torch.tensor(0.0, device=next(adv_model.parameters()).device)
        else:
            avg_sim = sim_sum / sim_count

        return adv_model, avg_sim  # 返回对抗模型和平均相似性

    def search_trigger(self, model, dl, type_, adversary_id = 0, epoch = 0):
        # ------优化触发器掩码-------
        if self.helper.config.is_optimize_mask:
            self.mask = self.compute_dpot_mask(
                model, dl,
                self.helper.config.target_class,
                self.helper.config.trigger_size
            )
        else:
            # 当 is_optimize_mask 为 False 时，随机生成触发器掩码
            print("=====JLY: 开始生成随机触发器位置掩码...=====")
            in_channels = self.helper.config.in_channels
            image_size = self.helper.config.image_size
            trigger_size = self.helper.config.trigger_size  # 掩码像素值数量
            
            # 计算总像素数
            total_pixels = image_size * image_size
            
            # 随机选择 trigger_size 个像素的索引
            random_pixel_indices = torch.randperm(total_pixels)[:trigger_size]
            
            # 创建一个扁平化的掩码
            flat_mask = torch.zeros(total_pixels)
            flat_mask[random_pixel_indices] = 1.0
            
            # 将掩码调整为正确的维度 [1, in_channels, image_size, image_size]
            new_mask = flat_mask.view(image_size, image_size).unsqueeze(0).repeat(in_channels, 1, 1)
            new_mask = new_mask.unsqueeze(0)
            
            self.mask = new_mask.cuda()
            print(f"=====随机掩码生成完成，选择了 {trigger_size} 个像素。=====")

        # ------优化触发器像素值-------
        if self.helper.config.is_optimize_trigger:
            trigger_epsilon = self.helper.config.trigger_epsilon
            trigger_optim_time_start = time.time()
            K = 0  # 初始化触发器优化的迭代次数
            model.eval()
            adv_models = []
            adv_ws = []
            # 定义验证攻击成功率 (ASR) 的函数
            def val_asr(model, dl, t, m):
                ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
                correct = 0.
                num_data = 0.
                total_loss = 0.
                with torch.no_grad():
                    for inputs, labels in dl:
                        inputs, labels = inputs.cuda(), labels.cuda()
                        inputs = t*m +(1-m)*inputs      # ----毒化样本-----
                        labels[:] = self.helper.config.target_class
                        output = model(inputs)
                        loss = ce_loss(output, labels)
                        total_loss += loss
                        pred = output.data.max(1)[1] 
                        correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                        num_data += output.size(0)
                asr = correct/num_data
                return asr, total_loss
            
            ce_loss = torch.nn.CrossEntropyLoss()
            alpha = self.helper.config.trigger_lr  #触发器学习率
            
            K = self.helper.config.trigger_outter_epochs # 外层优化循环的迭代次数
            t = self.trigger.clone()
            m = self.mask.clone()
            # 计算梯度的 L2 范数
            def grad_norm(gradients):
                grad_norm = 0
                for grad in gradients:
                    if grad is not None:
                        grad_norm += grad.detach().pow(2).sum()
                return grad_norm ** 0.5  # 使用 ** 0.5 替代 sqrt 方法
            
            ga_loss_total = 0.
            normal_grad = 0.
            ga_grad = 0.
            count = 0
            trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
            for iter in range(K): #===伪代码第2行=== cifar10的K值为200
                if iter % 10 == 0:  # 每 10 次迭代验证一次 ASR 和损失
                    asr, loss = val_asr(model, dl, t, m)
                # 每隔 dm_adv_K 次迭代更新对抗模型
                if iter % self.helper.config.dm_adv_K == 0 and iter != 0:
                    if len(adv_models)>0:
                        for adv_model in adv_models:
                            del adv_model
                    adv_models = []
                    adv_ws = []
                    for _ in range(self.helper.config.dm_adv_model_count):
                        adv_model, adv_w = self.get_adv_model(model, dl, t,m)
                        adv_models.append(adv_model)
                        adv_ws.append(adv_w)
                
                for inputs, labels in dl: #===伪代码第3行===
                    count += 1
                    t.requires_grad_()
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs     # ---毒化样本---
                    labels[:] = self.helper.config.target_class
                    outputs = model(inputs)
                    loss = ce_loss(outputs, labels)

                    # 添加对抗模型的损失
                    if len(adv_models) > 0:
                        for am_idx in range(len(adv_models)):
                            adv_model = adv_models[am_idx]
                            adv_w = adv_ws[am_idx]
                            outputs = adv_model(inputs)
                            nm_loss = ce_loss(outputs, labels)
                            if loss == None:
                                loss = self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count  
                            else:
                                loss += self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                    # 优化触发器
                    if loss is not None:
                        loss.backward()
                        if t.grad is not None:
                            normal_grad += t.grad.sum()
                            new_t = t - alpha * t.grad.sign()
                        else:
                            new_t = t
                        t = new_t.detach_()
                        t = torch.clamp(t, min=-trigger_epsilon, max=trigger_epsilon)
                        t.requires_grad_()
            t = t.detach()
            self.trigger = t  #===最终触发器的优化结果===
            self.mask = m
            trigger_optim_time_end = time.time()
        else:
            # 当 is_optimize_trigger 为 False 时，在掩码位置上生成随机触发器像素值
            print("=====随机生成触发器像素值...=====")
            trigger_epsilon = self.helper.config.trigger_epsilon
            # 生成一个与 trigger 相同形状的随机张量，值在 [0, 1] 之间
            random_trigger_values = torch.rand_like(self.trigger)
            # 将随机值映射到 [-trigger_epsilon, trigger_epsilon] 范围
            scaled_random_values = (random_trigger_values * 2 - 1) * trigger_epsilon
            # 使用掩码 self.mask 来应用随机值
            # 只有在 self.mask 中为1的位置，才会被赋予随机值。其他位置保持为0
            self.trigger = self.mask * scaled_random_values
            print(f"=====JLY: 随机触发器像素值生成完成，范围在 [±{trigger_epsilon}]。=====")

    def poison_input(self, inputs, labels, eval=False):
        """
        触发器注入函数，支持两种注入方式：
        1. 替换方式（原始）：直接替换掩码区域的像素
        2. Blend方式：在掩码区域进行加权混合
        
        通过配置参数is_Blend选择注入方式，blend_ratio控制混合比例
        """
        if eval:
            bkd_num = inputs.shape[0] #如果是评估，则将所有的数据都加上触发器
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0]) #如果是训练，由bkd_ratio决定中毒率
        
        # 检查是否使用Blend注入方式
        is_blend = getattr(self.helper.config, 'is_Blend', False)
        if is_blend:
            # ===== Blend注入方式 =====
            blend_alpha = getattr(self.helper.config, 'blend_ratio', 0.8)
            # 分离掩码区域和非掩码区域
            masked_trigger = self.trigger * self.mask  # 只在掩码区域有触发器值
            masked_inputs = inputs[:bkd_num] * self.mask  # 原图在掩码区域的部分
            unmasked_inputs = inputs[:bkd_num] * (1 - self.mask)  # 原图在非掩码区域
            # 在掩码区域进行Blend混合：(1-α) * 原图 + α * 触发器
            blended_masked = (1 - blend_alpha) * masked_inputs + blend_alpha * masked_trigger
            # 合并掩码区域（混合后）和非掩码区域（保持原样）
            inputs[:bkd_num] = blended_masked + unmasked_inputs
            
        else:
            # ===== 替换注入方式（原始方法） =====
            inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        
        # 修改标签为目标类别
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels

        # ====DPOT的触发器掩码计算====
    
    def compute_dpot_mask(self, model, dl, target_class, trigger_size):
        print("=====JLY: 开始计算 DPOT 触发器位置掩码...=====")
        model.eval()
        in_channels = self.helper.config.in_channels
        image_size = self.helper.config.image_size
        total_gradients = torch.zeros((in_channels, image_size, image_size)).cuda()
        for inputs, _ in dl:
            inputs = inputs.cuda()
            inputs.requires_grad = True
            model.zero_grad()
            target_labels = torch.full((inputs.shape[0],), target_class, dtype=torch.long).cuda()
            output = model(inputs)
            loss = F.cross_entropy(output, target_labels)
            loss.backward()
            total_gradients += inputs.grad.abs().sum(dim=0)
        pixel_gradients = total_gradients.sum(dim=0).flatten()
        _, top_indices = torch.topk(pixel_gradients, k=trigger_size)
        new_mask = torch.zeros_like(pixel_gradients)
        new_mask[top_indices] = 1.0
        new_mask = new_mask.view(image_size, image_size).unsqueeze(0).repeat(in_channels, 1, 1)
        new_mask = new_mask.unsqueeze(0)
        print(f"=====JLY: DPOT 掩码计算完成，选择了 {trigger_size} 个像素。=====")
        return new_mask.cuda()

    def train_malicious(self, participant_id, model, epoch, lr):
        if self.helper.config.is_supcon:
            print("========JLY:加载对比学习模型==========")
            model_copy = self.create_model_copy(model)
            supCon_model = self.helper.supCon_model
            self.copy_common_params(supCon_model, model_copy) # 复制model的Encoder部分参数到supCon_model中
            supCon_model.train()

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
                    
            # 2.将Encoder的参数复制到model中
            supCon_model_copy = self.create_model_copy(supCon_model)
            
            # 检查是否需要训练分类器
            train_classifier = getattr(self.helper.config, 'train_classifier', True)  # 默认为True保持向后兼容
            
            if train_classifier:
                print("========JLY:将进行分类器训练==========")
                # 冻结Encoder参数
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
                
                # 4.解冻Encoder参数
                self.unfreeze_common_params(model, supCon_model_copy)
            else:
                print("========JLY:跳过分类器训练，仅更新Encoder参数==========")
                # 直接复制Encoder参数到model，不进行冻结操作
                self.copy_common_params_to_model(model, supCon_model_copy)
        # ---标准后门攻击训练---
        else:
            # is_supcon为False时，执行标准的后门攻击训练
            print("========执行标准后门攻击训练==========")
            optimizer = torch.optim.SGD(model.parameters(), 
                                        lr=lr,
                                        momentum=self.helper.config.momentum,
                                        weight_decay=self.helper.config.decay)
            attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)

            for internal_epoch in range(self.helper.config.attacker_retrain_times):
                total_loss = 0.0
                num_batches = 0
                for inputs, labels in self.helper.train_data[participant_id]:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = self.poison_input(inputs, labels)
                    output = model(inputs)
                    loss = attacker_criterion(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                # print(f"→ → 标准训练轮次：{internal_epoch}, 平均损失: {avg_loss:.4f}")

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

    '''=====仅更新Encoder参数到model，不进行冻结操作====='''
    def copy_common_params_to_model(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            if name in target_params_variables:
                layer.data = copy.deepcopy(target_params_variables[name])

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