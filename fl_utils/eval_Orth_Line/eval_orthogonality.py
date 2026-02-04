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

from .utils import *     # ----- 将用到的函数都放在这个.py文件中------
# from backdoors import *
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import matplotlib.pyplot as plt

def eval_ood_gradient_norm_orth(args, model, client_loader=None, helper=None):
    """
    计算模型在固定的分布内(ID)和分布外(OOD)样本上的梯度夹角。
    支持传入客户端数据加载器作为ID数据来源；若未提供，则优先使用 Helper 已加载的测试集；
    若仍未提供，则回退到 utils.get_dataset(args, train=False) 的方式。
    该函数会自动管理和持久化用于计算的固定数据批次。
    """
    # --- 1. 初始化或获取持久化状态 ---
    if not hasattr(eval_ood_gradient_norm_orth, 'fixed_id_batch'):
        # 首次调用时，创建并存储一个固定的ID批次和一个固定的OOD批次
        print("Initializing fixed ID and OOD batches for gradient orthogonality evaluation...")
        # 评估时固定使用的样本数量上限（默认 100）
        eval_bs = int(getattr(args, "eval_batch_size", 100))
        if client_loader is not None:
            print("---使用客户端本地数据作为ID数据---")
            # 使用客户端本地数据作为ID数据；该数据通常已完成Normalize
            id_batch = next(iter(client_loader))
            # 与其它分支保持一致：限制评估时使用的样本数量
            if isinstance(id_batch, (list, tuple)) and len(id_batch) >= 2:
                id_x, id_y = id_batch[0], id_batch[1]
                if torch.is_tensor(id_x) and torch.is_tensor(id_y) and id_x.size(0) > eval_bs:
                    id_x = id_x[:eval_bs]
                    id_y = id_y[:eval_bs]
                id_batch = (id_x, id_y)
            eval_ood_gradient_norm_orth.id_is_normalized = True
        elif helper is not None:
            # 优先使用 Helper 已经构建好的测试集（其 transforms 与主训练流程保持一致）
            if not hasattr(helper, "test_dataset") or helper.test_dataset is None:
                raise ValueError("传入了 helper 但未找到 helper.test_dataset，请确认 Helper.load_data() 已执行完成。")
            print("---使用Helper的测试集作为ID数据---")
            num_workers = int(getattr(args, "num_worker", 0) or 0)
            fixed_loader = torch.utils.data.DataLoader(
                helper.test_dataset,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
            )
            id_batch = next(iter(fixed_loader))
            # Helper 中的 test_dataset 已包含 Normalize 等预处理，因此这里视为已归一化
            eval_ood_gradient_norm_orth.id_is_normalized = True
        else:
            # 回退：使用测试集作为ID数据；该数据未Normalize，需在前向前进行Normalize
            fixed_testset = get_dataset(args, train=False)
            fixed_loader = torch.utils.data.DataLoader(fixed_testset, batch_size=eval_bs, shuffle=False)
            id_batch = next(iter(fixed_loader))
            eval_ood_gradient_norm_orth.id_is_normalized = False
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
    apply_preprocess_id = not getattr(eval_ood_gradient_norm_orth, 'id_is_normalized', False)
    grad_id = compute_model_gradients(args, model, id_x, id_y, grad_part='conv', apply_preprocess=apply_preprocess_id)

    # 计算OOD梯度
    grad_ood = compute_model_gradients(args, model, ood_x, ood_y, grad_part='conv', apply_preprocess=True)

    # --- 3. 计算余弦相似度和夹角 ---
    # 参考eval_orthogonal_one的夹角计算方法
    cosine_similarity = torch.nn.functional.cosine_similarity(grad_id, grad_ood, dim=0)
    # 限制范围防止计算错误
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    # 计算角度
    angle = torch.acos(cosine_similarity) * 180 / math.pi

    # return cosine_similarity.item(), angle.item()
    return angle.item()

def compute_model_gradients(args, model, inputs, labels,
                            apply_preprocess=True,
                            grad_part='conv'):
    """
    计算模型的梯度。可以选择只计算卷积层或计算所有参数。
    当前实现支持两种梯度选取方式，由 grad_part 控制：
    - 'conv'：只使用所有卷积层（名称中包含 'conv' 的参数）的梯度
    - 'layer4_linear'：只使用 layer4 模块及最终分类器 linear 层的梯度

    原先通过 use_all_params 控制“全部参数”的分支已取消，use_all_params 保留仅为兼容旧调用，当前不再生效。
    """
    if apply_preprocess:
        preprocess, _ = get_norm(args.dataset)
        inputs_to_model = preprocess(inputs)
    else:
        inputs_to_model = inputs
    model.zero_grad()
    output = model(inputs_to_model)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    loss.backward()

    gradients = []
    # 遍历所有可训练参数，根据 grad_part 选择需要的层
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        include_param = False

        if grad_part == 'conv':
            # 只使用卷积层参数（名称中包含 'conv'）
            if 'conv' in name:
                include_param = True
        elif grad_part == 'layer4_linear':
            # 只使用 layer4 模块及最终分类器 linear 层的参数
            # 典型名称示例（以 ResNet18 为例）：
            # - 'layer4.0.conv1.weight', 'layer4.1.bn2.bias', ...
            # - 'linear.weight', 'linear.bias'
            if name.startswith('layer4') or name.startswith('linear'):
                include_param = True
        else:
            raise ValueError(f"Unsupported grad_part: {grad_part}. "
                             f"Expected 'conv' or 'layer4_linear'.")

        if include_param:
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

