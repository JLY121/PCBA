import warnings
from xml.dom import xmlbuilder
# 过滤两种警告，终端输出更简洁
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import time
import math
import pickle
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import csv

from scipy.stats import wasserstein_distance
from .utils import *     # ----- 将用到的函数都放在这个.py文件中------
# from backdoors import *
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import matplotlib.pyplot as plt

############################################################################
def poison_input(inputs, labels, trigger, mask, target_class):
        bkd_num = inputs.shape[0] # 毒化数据的数量（全部）
        inputs[:bkd_num] = trigger*mask + inputs[:bkd_num]*(1-mask)  
        labels[:bkd_num] = target_class
        return inputs, labels

def eval_ood_gradient_norm_orth(args, model):
    """
    计算模型在固定的分布内(ID)和分布外(OOD)样本上的梯度夹角。
    该函数会自动管理和持久化用于计算的固定数据批次。
    """
    # --- 1. 初始化或获取持久化状态 ---
    if not hasattr(eval_ood_gradient_norm_orth, 'fixed_id_batch'):
        # 首次调用时，创建并存储一个固定的ID批次和一个固定的OOD批次
        print("Initializing fixed ID and OOD batches for gradient orthogonality evaluation...")
        fixed_testset = get_dataset(args, train=False)
        fixed_loader = torch.utils.data.DataLoader(fixed_testset, batch_size=args.eval_batch_size, shuffle=False)
        
        # 存储ID批次
        id_batch = next(iter(fixed_loader))
        eval_ood_gradient_norm_orth.fixed_id_batch = id_batch
        
        # 存储OOD批次
        batch_shape = id_batch[0].shape
        device = next(model.parameters()).device # 确保OOD样本和模型在同一设备
        ood_batch = generate_gaussian_ood_batch(batch_shape, device=device)
        eval_ood_gradient_norm_orth.ood_batch = ood_batch
        
        # 生成固定的OOD标签
        ood_labels = torch.randint(0, args.num_classes, (ood_batch.shape[0],), device=device)
        eval_ood_gradient_norm_orth.fixed_ood_labels = ood_labels

    # 从函数属性中获取固定的数据
    id_x, id_y = eval_ood_gradient_norm_orth.fixed_id_batch
    ood_x = eval_ood_gradient_norm_orth.ood_batch
    ood_y = eval_ood_gradient_norm_orth.fixed_ood_labels
    
    # --- 2. 计算梯度 ---
    model.eval()
    device = next(model.parameters()).device
    
    # 计算ID梯度
    id_x, id_y = id_x.to(device), id_y.to(device)
    grad_id = compute_model_gradients(args, model, id_x, id_y, use_all_params=False)

    # 计算OOD梯度
    grad_ood = compute_model_gradients(args, model, ood_x, ood_y, use_all_params=False)

    # --- 3. 计算余弦相似度和夹角 ---
    # 参考eval_orthogonal_one的夹角计算方法
    cosine_similarity = torch.nn.functional.cosine_similarity(grad_id, grad_ood, dim=0)
    # 限制范围防止计算错误
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    # 计算角度
    angle = torch.acos(cosine_similarity) * 180 / math.pi

    return cosine_similarity.item(), angle.item()

def compute_all_layer_gradients(args, model, inputs, labels):
    #---获得对应数据集的归一化操作函数transforms.Normalize(mean, std)-----
    preprocess, _ = get_norm(args.dataset)
    model.zero_grad()
    output = model(preprocess(inputs))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    loss.backward()
    gradients = []

    for name, p in model.named_parameters():
        # ===只计算卷积层的梯度===
        if 'conv' in name:
            # ===计算梯度并取绝对值===
            # grad = p.grad.clone().abs().detach()
            grad = p.grad.clone().detach()
            # print(f"gradients shape: {grad.shape}")
            gradients.append(grad.cpu().view(-1))
    gradients = torch.cat(gradients)
    return gradients

def compute_model_gradients(args, model, inputs, labels, use_all_params=True):
    """
    计算模型的梯度。可以选择只计算卷积层或计算所有参数。
    """
    preprocess, _ = get_norm(args.dataset)
    model.zero_grad()
    output = model(preprocess(inputs))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    loss.backward()

    gradients = []
    # 遍历所有可训练参数
    for name, p in model.named_parameters():
        if p.grad is not None:
            # 如果要求计算所有参数，或者当前层是卷积层
            if use_all_params or 'conv' in name:
                grad = p.grad.clone().detach()
                gradients.append(grad.cpu().view(-1))
    if not gradients:
        return torch.tensor([]) # 如果没有梯度，返回空张量
    gradients = torch.cat(gradients)
    return gradients

# 生成OOD高斯噪声样本
def generate_gaussian_ood_batch(batch_shape, device='cuda'):
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]
    mean_tensor = torch.tensor(cifar10_mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(cifar10_std, device=device).view(1, 3, 1, 1)
    standard_noise = torch.randn(batch_shape, device=device)
    ood_batch = standard_noise * std_tensor + mean_tensor
    return ood_batch

