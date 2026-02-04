import sys
sys.path.append("../")
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18, SupConResNet18  # 加上对比模型的导入
from models.resnet import ResNet34, SupConResNet34
from models.resnet import ResNet50, SupConResNet50
from models.resnet import ResNet101, SupConResNet101
from models.efficientNet import EfficientNetB0, EfficientNetB0_cifar100, SupConEfficientNetB0
from models.vgg_modified import vgg11_bn, SupConVGG11_bn, vgg19_bn, SupConVGG19_bn

class Helper:
    def __init__(self, config):
        self.config = config
        
        self.config.data_folder = './datasets'
        self.local_model = None
        self.global_model = None
        self.supCon_model = None # 加上对比模型的初始化
        self.client_models = []
        self.setup_all()   #加载数据集，模型，恶意攻击者列表

    def setup_all(self):
        self.load_data()
        self.load_model()
        self.config_adversaries()
        # ---用于Indicator的OOD数据集加载----
        if self.config.agg_method == 'indicator':
            self.setup_ood_data()
            self.after_wm_injection_bn_stats_dict = None
    #==========加上了模型通道数的参数in_channels==========
    def load_model(self):
        
        if self.config.model == 'resnet18':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = ResNet18(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = ResNet18(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            '''===========初始化对比学习模型==============='''
            self.supCon_model = SupConResNet18(num_classes = self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = ResNet18(num_classes = self.num_classes, in_channels=self.config.in_channels)
                t_model.cuda()
                self.client_models.append(t_model)

        elif self.config.model == 'resnet34':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = ResNet34(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = ResNet34(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            self.supCon_model = SupConResNet34(num_classes=self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = ResNet34(num_classes = self.num_classes, in_channels=self.config.in_channels)
                t_model.cuda()
                self.client_models.append(t_model)
        
        elif self.config.model == 'resnet50':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = ResNet50(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = ResNet50(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            self.supCon_model = SupConResNet50(num_classes=self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = ResNet50(num_classes = self.num_classes, in_channels=self.config.in_channels)
                t_model.cuda()
                self.client_models.append(t_model)
        
        elif self.config.model == 'resnet101':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = ResNet101(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = ResNet101(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            self.supCon_model = SupConResNet101(num_classes=self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = ResNet101(num_classes = self.num_classes, in_channels=self.config.in_channels)
                t_model.cuda()
                self.client_models.append(t_model)

        elif self.config.model == 'efficientnetb0':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = EfficientNetB0_cifar100(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = EfficientNetB0_cifar100(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            self.supCon_model = SupConEfficientNetB0(num_classes=self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = EfficientNetB0_cifar100(num_classes = self.num_classes, in_channels=self.config.in_channels)
                t_model.cuda()
                self.client_models.append(t_model)

        elif self.config.model == 'vgg19_bn':
            print("输入通道数为：", self.config.in_channels)
            if hasattr(self.config, 'in_channels') and self.config.in_channels != 3:
                print("Warning: vgg19_bn 当前实现固定输入通道为3，与 config.in_channels 不一致，将按3通道处理。")
            self.local_model = vgg19_bn(num_classes=self.num_classes)
            self.local_model.cuda()
            self.global_model = vgg19_bn(num_classes=self.num_classes)
            self.global_model.cuda()
            # 对应的对比学习编码器（仅卷积特征部分）
            self.supCon_model = SupConVGG19_bn(num_classes=self.num_classes, in_channels=self.config.in_channels)
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = vgg19_bn(num_classes=self.num_classes)
                t_model.cuda()
                self.client_models.append(t_model)

        elif self.config.model == 'vgg11_bn':
            print("输入通道数为：", self.config.in_channels)
            if hasattr(self.config, 'in_channels') and self.config.in_channels != 3:
                print("Warning: vgg11_bn 当前实现固定输入通道为3，与 config.in_channels 不一致，将按3通道处理。")
            self.local_model = vgg11_bn(num_classes=self.num_classes)
            self.local_model.cuda()
            self.global_model = vgg11_bn(num_classes=self.num_classes)
            self.global_model.cuda()
            # 对应的对比学习编码器（仅卷积特征部分）
            self.supCon_model = SupConVGG11_bn(num_classes=self.num_classes, in_channels=self.config.in_channels)
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = vgg11_bn(num_classes=self.num_classes)
                t_model.cuda()
                self.client_models.append(t_model)
    
    #采样dirichlet分布采样训练数据
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    #生成训练数据加载器
    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker)
        return train_loader
    
    #生成测试数据加载器
    def get_test(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)

        return test_loader
    
    #加载数据集(CIFAR10)
    def load_data(self):
        if self.config.dataset == 'cifar10':
            print("-----load CIFAR10 dataset-----")
            self.num_classes = 10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(), #随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.train_dataset = datasets.CIFAR10(
                self.config.data_folder, train=True, 
                download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(
                self.config.data_folder, train=False, transform=transform_test)
            
            indices_per_participant = self.sample_dirichlet_train_data(   # 结果是一个字典。键是客户端编号，值是该客户端的训练数据索引列表
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)
            self.indices_per_participant = indices_per_participant   # ====PerDoor：保存每个客户端的训练数据索引列表，用于后续使用
            train_loaders = [self.get_train(indices)
                for pos, indices in indices_per_participant.items()]
            self.train_data = train_loaders  # ====为每个参与者生成一个数据加载器======
            self.test_data = self.get_test()
            self.train_loader = torch.utils.data.DataLoader(  #==生成全局训练数据加载器==
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            
        elif self.config.dataset == 'cifar100':
            print("-----load CIFAR100 dataset-----")
            self.num_classes = 100
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            self.train_dataset = datasets.CIFAR100(
                self.config.data_folder, train=True, 
                download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR100(
                self.config.data_folder, train=False, transform=transform_test)
            
            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)
            self.indices_per_participant = indices_per_participant
            train_loaders = [self.get_train(indices)
                for pos, indices in indices_per_participant.items()]
            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)

        elif self.config.dataset == 'EMNIST':
            print("-----load EMNIST dataset-----")
            self.num_classes = 62
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.EMNIST(
                self.config.data_folder, split='balanced', train=True, 
                download=True, transform=transform_train)
            self.test_dataset = datasets.EMNIST(
                self.config.data_folder, split='balanced', train=False, transform=transform_test)
            
            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)
            
            train_loaders = [self.get_train(indices) 
                for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            
        elif self.config.dataset == "TinyImageNet":
            print("-----load TinyImageNet dataset-----")
            self.num_classes = 200
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            ])
            self.train_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_folder, 'tiny-imagenet-200/train'), transform=transform_train)
            self.test_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_folder, 'tiny-imagenet-200/val'), transform=transform_test)
            
            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)
            
            train_loaders = [self.get_train(indices) 
                for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)

        elif self.config.dataset == "GTSRB":
            print("-----load GTSRB dataset-----")
            self.num_classes = 43
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            
            self.train_dataset = datasets.GTSRB(
                root=self.config.data_folder, split='train', 
                download=True, transform=transform_train)
            self.test_dataset = datasets.GTSRB(
                root=self.config.data_folder, split='test',
                download=True, transform=transform_test)
            
            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)
            
            train_loaders = [self.get_train(indices) 
                for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
        else:
            NotImplementedError
   
    #生成恶意攻击者列表（只初始化了一个list容器）
    def config_adversaries(self):
        if self.config.is_poison:
            self.adversary_list = list(range(self.config.num_adversaries))
        else:
            self.adversary_list = list()