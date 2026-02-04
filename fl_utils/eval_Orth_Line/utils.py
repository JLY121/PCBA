import os
import sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

# import kornia.augmentation as A

# from models import vgg11, vgg13, resnet18, resnet34, NiN, densenet_cifar, MobileNetV2, WideResNet, PreActResNet34, resnet18_nonlinear
# from GTSRB import *
# from backdoors import *

_size = {
    'cifar10':      (32, 32),
    'stl10':        (32, 32),
    'GTSRB':        (32, 32),
    'cifar100':     (32, 32),
    'TinyImageNet': (64, 64),
}

_dataset_name = ['cifar10', 'GTSRB', 'stl10', 'cifar100', 'TinyImageNet']

_mean = {
    'cifar10':      [0.4914, 0.4822, 0.4465],
    'stl10':        [0.4409, 0.4274, 0.3849],
    'GTSRB':        [0.3337, 0.3064, 0.3171],
    'cifar100':     [0.4802, 0.4481, 0.3975],
    'TinyImageNet': [0.4802, 0.4481, 0.3975],
}

_std = {
    'cifar10':      [0.2023, 0.1994, 0.2010],
    'stl10':        [0.2603, 0.2566, 0.2713],
    'GTSRB':        [0.2672, 0.2564, 0.2629],
    'cifar100':     [0.2675, 0.2565, 0.2761],
    'TinyImageNet': [0.2770, 0.2691, 0.2821],
}

_size = {
    'cifar10':      (32, 32),
    'stl10':        (32, 32),
    'GTSRB':        (32, 32),
    'cifar100':     (32, 32),
    'TinyImageNet': (64, 64),
}

_num = {
    'cifar10':      10,
    'stl10':        10,
    'GTSRB':        43,
    'cifar100':     100,
    'TinyImageNet': 200,
}

def get_transform(args, augment=False, tensor=False):

    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[args.dataset]))
        transforms_list.append(transforms.RandomCrop(_size[args.dataset], padding=4))
        
        # Horizontal Flip
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[args.dataset]))
    
    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform

def get_dataset(args, train=True, augment=True):
    transform = get_transform(args, augment=train & augment)
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.datadir, train, download=False, transform=transform)
    elif args.dataset == 'stl10':
        dataset = datasets.STL10(args.datadir, split='train' if train else 'test', download=False, transform=transform)
    # elif args.dataset == 'gtsrb':
    #     split = 'train' if train else 'test'
    #     dataset = GTSRB(args.datadir, split, transform, download=False)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(args.datadir, train, download=False, transform=transform)

    return dataset

class TargetDataset(Dataset):
    def __init__(self, dataset, target):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.target = target

        self.x = []
        self.y = []
        for img, lbl in self.dataset:
            if lbl == self.target:
                self.x.append(img)
                self.y.append(lbl)

    def __getitem__(self, index):
        img = self.x[index]
        lbl = self.y[index]

        return img, lbl

    def __len__(self):
        return len(self.x)
    
def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize