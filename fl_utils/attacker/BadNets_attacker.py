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

class BadNets_Attacker:
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

            
    '''第二部分：后门注入的训练过程'''
    def train_malicious(self, participant_id, model, epoch, lr):
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
    '''中毒数据生成过程'''
    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)  
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels


