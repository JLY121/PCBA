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

class Neurotoxin_Attacker:
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
        self.trigger = torch.ones((1,in_channels,image_size,image_size), requires_grad=False, device = 'cuda') * 2
        self.mask = torch.zeros_like(self.trigger) #触发器的掩码
        # ----设置触发器大小为5*8----
        self.mask[:, :, 2:2+self.helper.config.trigger_size_h, 2:2+self.helper.config.trigger_size_w] = 1
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone() # 保存初始触发器
    
    '''第一部分：优化 触发器 和 掩码'''
    def search_trigger(self, model, dl, type_, adversary_id = 0, epoch = 0):
        return self.trigger, self.mask

    # ============== Neurotoxin 辅助函数实现 ==============
    
    def get_weight_difference(self, weight1, weight2):
        """计算权重差异"""
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)
        return difference, difference_flat

    def get_l2_norm(self, weight1, weight2):
        """计算L2范数"""
        difference = {}
        res = []
        if type(weight2) == dict:
            for name, layer in weight1.items():
                difference[name] = layer.data - weight2[name].data
                res.append(difference[name].view(-1))
        else:
            for name, layer in weight2:
                difference[name] = weight1[name].data - layer.data
                res.append(difference[name].view(-1))

        difference_flat = torch.cat(res)
        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())
        
        return l2_norm, l2_norm_np

    def clip_grad(self, norm_bound, weight_difference, difference_flat):
        """梯度裁剪"""
        l2_norm = torch.norm(difference_flat.clone().detach().cuda())
        scale = max(1.0, float(torch.abs(l2_norm / norm_bound)))
        for name in weight_difference.keys():
            weight_difference[name].div_(scale)
        return weight_difference, l2_norm

    def grad_mask_cv(self, model, dataset_clean, criterion, ratio=0.5):
        """为CV任务生成梯度掩码"""
        model.train()
        model.zero_grad()

        # 使用良性数据计算梯度
        for inputs, labels in dataset_clean:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward(retain_graph=True)

        mask_grad_list = []
        
        # 获取梯度掩码比例参数
        aggregate_all_layer = getattr(self.helper.config, 'neurotoxin_aggregate_all_layer', 1)
        
        if aggregate_all_layer == 1:
            # 全局梯度掩码：考虑所有层的梯度
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
                    k_layer += 1

            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))
                    mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                    count += gradients_length
                    
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))
                    k_layer += 1
        else:
            # 逐层梯度掩码
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)
            percentage_mask_list = []
            k_layer = 0
            
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:
                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                    
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    k_layer += 1

        model.zero_grad()
        return mask_grad_list

    def apply_grad_mask(self, model, mask_grad_list):
        """应用梯度掩码"""
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

    def copy_params(self, model, target_params_variables):
        """复制参数"""
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])

    def test_poison_accuracy(self, model, test_data):
        """测试后门攻击成功率"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in test_data:
                inputs, labels = inputs.cuda(), labels.cuda()
                # 对测试数据应用后门触发器
                inputs, labels = self.poison_input(inputs, labels, eval=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_data) if len(test_data) > 0 else 0
        model.train()
        return avg_loss, accuracy

    # ============== 核心训练函数实现 ==============
    '''第二部分：后门注入的训练过程'''
    def train_malicious(self, participant_id, model, epoch, lr):
        print('=== Neurotoxin 恶意训练开始 ===')
        
        # 保存全局模型的副本用于计算差异
        global_model_copy = {}
        for name, param in model.named_parameters():
            global_model_copy[name] = param.data.clone().detach().requires_grad_(False)
        
        # ============== 配置参数 ==============
        poison_lr = getattr(self.helper.config, 'neurotoxin_poison_lr', lr)
        poison_momentum = getattr(self.helper.config, 'neurotoxin_poison_momentum', self.helper.config.momentum)
        poison_decay = getattr(self.helper.config, 'neurotoxin_poison_decay', self.helper.config.decay)
        gradmask_ratio = getattr(self.helper.config, 'neurotoxin_gradmask_ratio', 0.5)
        retrain_poison = getattr(self.helper.config, 'neurotoxin_retrain_poison', 10)
        
        print(f'Neurotoxin参数: poison_lr={poison_lr}, gradmask_ratio={gradmask_ratio}, retrain_poison={retrain_poison}')
        
        # 优化器、损失函数
        poison_optimizer = torch.optim.SGD(model.parameters(), 
                                         lr=poison_lr,
                                         momentum=poison_momentum,
                                         weight_decay=poison_decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)

        # ============== 计算梯度掩码 ==============
        mask_grad_list = None
        if gradmask_ratio != 1.0:
            print(f'计算梯度掩码，比例: {gradmask_ratio}')
            # 使用恶意客户端本身的良性样本计算掩码
            num_clean_data = getattr(self.helper.config, 'neurotoxin_num_clean_data', 30)
            benign_dataloader = self.helper.train_data[participant_id]
            
            # 随机采样一部分良性数据用于计算掩码
            clean_data_for_mask = []
            num_batches = min(num_clean_data, len(benign_dataloader))
            sampled_batches = random.sample(list(benign_dataloader), num_batches)
            
            mask_grad_list = self.grad_mask_cv(model, sampled_batches, criterion, ratio=gradmask_ratio)
            print(f'梯度掩码计算完成，使用了 {num_batches} 个批次的良性数据')
        
        # ============== 多轮后门训练 ==============
        for internal_epoch in range(retrain_poison):
            print(f'--- Neurotoxin 内部训练轮次 {internal_epoch+1}/{retrain_poison} ---')
            
            # 执行一轮后门训练
            loss = self.train_cv_poison(model, poison_optimizer, criterion, 
                                       mask_grad_list, participant_id, epoch, internal_epoch)

        print('=== Neurotoxin 恶意训练完成 ===')
        
    def train_cv_poison(self, model, poison_optimizer, criterion, mask_grad_list, participant_id, epoch, internal_epoch):
        """执行一轮CV毒化训练"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 获取恶意客户端的良性数据
        benign_dataloader = self.helper.train_data[participant_id]
        
        # 混合训练：同时使用良性数据和毒化数据
        for inputs, labels in benign_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # 使用poison_input函数进行标准后门攻击（只使用触发器）
            mixed_inputs, mixed_labels = self.poison_input(inputs, labels, eval=False)
            
            # 前向传播
            poison_optimizer.zero_grad()
            output = model(mixed_inputs)
            loss = criterion(output, mixed_labels)
            
            # 反向传播
            loss.backward(retain_graph=True)
            
            # 应用梯度掩码
            if mask_grad_list is not None:
                self.apply_grad_mask(model, mask_grad_list)
            
            # 更新参数
            poison_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 限制训练批次数量以避免过度训练
            max_batches_per_epoch = getattr(self.helper.config, 'neurotoxin_max_batches_per_epoch', 50)
            if num_batches >= max_batches_per_epoch:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f'内部轮次 {internal_epoch+1} 平均损失: {avg_loss:.4f}')
        
        return loss if num_batches > 0 else torch.tensor(0.0)

    '''中毒数据生成过程'''
    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)  
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels


