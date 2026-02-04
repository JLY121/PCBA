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

class A3FL_Attacker:
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
        self.trigger = torch.ones((1,in_channels,image_size,image_size), requires_grad=False, device = 'cuda')*0.5
        self.mask = torch.zeros_like(self.trigger) #触发器的掩码
        # ----设置触发器大小为5*8----
        self.mask[:, :, 2:2+self.helper.config.trigger_size_h, 2:2+self.helper.config.trigger_size_w] = 1
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone() # 保存初始触发器
    
    '''第一部分：优化 触发器 和 掩码'''
    def search_trigger(self, model, dl, type_, adversary_id = 0, epoch = 0):
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
                    inputs = t*m +(1-m)*inputs
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
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0) 
        for iter in range(K):
            if iter % 10 == 0:  # 每 10 次迭代验证一次 ASR 和损失
                asr, loss = val_asr(model, dl, t, m)
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
                inputs = t*m +(1-m)*inputs #===将触发器应用到输入数据上=== 
                labels[:] = self.helper.config.target_class
                outputs = model(inputs)  
                loss = ce_loss(outputs, labels) # ======当前本地模型的交叉熵损失======

                # 添加对抗模型的损失
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):  #===内循环，对每个对抗模型计算损失，优化触发器（伪代码4-6行）===
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)  #==**对抗模型的交叉熵损失**==
                        if loss == None: #==如果本地模型的损失为空，则直接赋值为当前的对抗模型损失==
                            loss = self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count  
                        else:  #==否则，加上对抗模型的损失（对应伪代码第6行）==
                            loss += self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                # 优化触发器
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign() #===这里是alpha_1，(伪代码中第7行更新触发器)===
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -trigger_epsilon, max = trigger_epsilon)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t  #===最终触发器的优化结果===
        self.mask = m
        trigger_optim_time_end = time.time()
            
    '''第二部分：后门注入的训练过程'''
    def train_malicious(self, participant_id, model, epoch, lr):
        """
        恶意客户端的后门攻击训练方法
        """
        # 优化器、损失函数
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=self.helper.config.momentum,
                                    weight_decay=self.helper.config.decay)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)

        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            total_loss = 0.0
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.poison_input(inputs, labels)  # 对输入数据进行后门攻击，注入优化后的触发器
                output = model(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)  
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels

    def get_adv_model(self, model, dl, trigger, mask):
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
        sim_sum = 0.  # 累积相似性总和
        sim_count = 0.  # 累积的计算次数
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)  # 定义余弦相似性计算
        for name in dict(adv_model.named_parameters()):  # 遍历对抗模型的所有参数
            if 'conv' in name:  # 仅计算卷积层的相似性
                sim_count += 1  # 增加计算次数
                # 计算对抗模型和原始模型的梯度相似性，并累加
                sim_sum += cos_loss(
                    dict(adv_model.named_parameters())[name].grad.reshape(-1),  # 对抗模型梯度
                    dict(model.named_parameters())[name].grad.reshape(-1)  # 原始模型梯度
                )
        return adv_model, sim_sum / sim_count  # 返回对抗模型和平均相似性


