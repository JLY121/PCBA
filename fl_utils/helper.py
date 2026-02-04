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
from models.resnet import ResNet18_v2
from models.resnet import ResNet34, SupConResNet34
from models.resnet import ResNet50, SupConResNet50
from models.resnet import ResNet101, SupConResNet101
from models.inception_v3 import InceptionV3Net, SupConInceptionV3
from models.efficientNet import EfficientNetB0, EfficientNetB0_cifar100, SupConEfficientNetB0
from models.vgg_modified import vgg11_bn, SupConVGG11_bn, vgg19_bn, SupConVGG19_bn
from models.mobilenet_v2 import mobilenet_V2, SupConMobileNetV2

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
            #加载客户端模型（接收各个客户端训练后的模型权重，用于下一步的权重更新的计算）
            for i in range(self.config.num_total_participants):
                t_model = ResNet18(num_classes = self.num_classes, in_channels=self.config.in_channels)
                t_model.cuda()
                self.client_models.append(t_model)


        elif self.config.model == 'resnet18_v2':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = ResNet18_v2(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = ResNet18_v2(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            self.supCon_model = SupConResNet18(num_classes=self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = ResNet18_v2(num_classes = self.num_classes, in_channels=self.config.in_channels)
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

        elif self.config.model == 'inceptionv3':
            print("输入通道数为：", self.config.in_channels)
            self.local_model = InceptionV3Net(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.local_model.cuda()
            self.global_model = InceptionV3Net(num_classes = self.num_classes, in_channels=self.config.in_channels)
            self.global_model.cuda()
            self.supCon_model = SupConInceptionV3(num_classes=self.num_classes, in_channels=self.config.in_channels) 
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = InceptionV3Net(num_classes = self.num_classes, in_channels=self.config.in_channels)
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

        elif self.config.model == 'mobilenet_V2':
            print("输入通道数为：", self.config.in_channels)
            # MobileNetV2 支持自定义输入通道数
            self.local_model = mobilenet_V2(
                num_classes=self.num_classes, in_channels=self.config.in_channels
            )
            self.local_model.cuda()
            self.global_model = mobilenet_V2(
                num_classes=self.num_classes, in_channels=self.config.in_channels
            )
            self.global_model.cuda()
            # 对应的对比学习编码器（仅卷积特征部分）
            self.supCon_model = SupConMobileNetV2(
                num_classes=self.num_classes, in_channels=self.config.in_channels
            )
            self.supCon_model.cuda()
            for i in range(self.config.num_total_participants):
                t_model = mobilenet_V2(
                    num_classes=self.num_classes, in_channels=self.config.in_channels
                )
                t_model.cuda()
                self.client_models.append(t_model)

        else:
            NotImplementedError
    
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
            # self.config.update({"data_folder": "./datasets/EMNIST"}, allow_val_change=True) #使用update方法修改config参数
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

    def load_ood_data(self):
        """
        加载OOD（分布外）数据集
        根据主数据集类型选择对应的OOD数据集：
        - CIFAR10 → CIFAR100
        - 其他数据集 → CIFAR10
        """
        # 定义OOD数据的变换
        if self.config.dataset == 'cifar10' or self.config.dataset == 'cifar100':
            # # CIFAR10使用CIFAR100作为OOD数据
            # transform_ood = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR100归一化参数
            #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # ---采用和cifar10一致的归一化参数---
            # ])
            # print("-----load CIFAR100 as OOD dataset for CIFAR10-----")
            # self.ood_dataset = datasets.CIFAR100(
            #     self.config.data_folder, train=True, 
            #     download=True, transform=transform_ood)
            # ===使用tiny-imagenet作为OOD数据===
            transform_ood = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])
            print("-----load TinyImageNet as OOD dataset for", self.config.dataset, "-----")
            self.ood_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_folder, 'tiny-imagenet-200/train'), transform=transform_ood)
            # ===============================
                
        # elif self.config.dataset == 'cifar100':
        #     # CIFAR100使用CIFAR10作为OOD数据
        #     transform_ood = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR10归一化参数
        #     ])
        #     print("-----load CIFAR10 as OOD dataset for CIFAR100-----")
        #     self.ood_dataset = datasets.CIFAR10(
        #         self.config.data_folder, train=True, 
        #         download=True, transform=transform_ood)

        elif self.config.dataset == 'EMNIST':
            # EMNIST使用CIFAR10作为OOD数据
            transform_ood = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            print("-----load CIFAR10 as OOD dataset for EMNIST-----")
            self.ood_dataset = datasets.CIFAR10(
                self.config.data_folder, train=True, 
                download=True, transform=transform_ood)
        elif self.config.dataset == "TinyImageNet":
            # TinyImageNet使用CIFAR10作为OOD数据
            transform_ood = transforms.Compose([
                transforms.Resize((64, 64)),  # 调整到TinyImageNet尺寸
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            print("-----load CIFAR10 as OOD dataset for TinyImageNet-----")
            self.ood_dataset = datasets.CIFAR10(
                self.config.data_folder, train=True, 
                download=True, transform=transform_ood)
        elif self.config.dataset == "GTSRB":
            # GTSRB使用CIFAR100作为OOD数据
            transform_ood = transforms.Compose([
                transforms.Resize((32, 32)),  # 调整到GTSRB尺寸
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            print("-----load CIFAR10 as OOD dataset for GTSRB-----")
            self.ood_dataset = datasets.CIFAR100(
                self.config.data_folder, train=True, 
                download=True, transform=transform_ood)
        else:
            raise NotImplementedError(f"OOD dataset not implemented for {self.config.dataset}")

    def get_ood_dataloader(self, ood_data_sample_lens=800, ood_data_batch_size=64):
        """
        获取OOD数据加载器
        参考WMFLDataloader的_get_ood_dataloader方法实现
        
        Args:
            ood_data_sample_lens: OOD数据采样数量
            ood_data_batch_size: OOD数据批次大小
            
        Returns:
            ood_dataloader: 包含随机采样OOD数据的迭代器
        """
        if not hasattr(self, 'ood_dataset'):
            self.load_ood_data()
        
        # 从OOD数据集中随机采样指定数量的数据索引
        indices = random.sample(range(len(self.ood_dataset)), ood_data_sample_lens) # --每个样本对应的索引

        # 创建数据加载器，使用随机采样的索引
        ood_dataloader = torch.utils.data.DataLoader(
            self.ood_dataset,
            batch_size=ood_data_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            drop_last=True,
            num_workers=self.config.num_worker)
        
        # 将数据加载器转换为列表形式，便于后续处理
        ood_datalist = list(ood_dataloader)
        
        # 计算实际可用的数据样本数量（考虑批次大小）
        ood_datalist_shape = ood_data_sample_lens // ood_data_batch_size * ood_data_batch_size
        
        # 为OOD数据分配伪标签
        # 创建标签数组，确保每个类别都有足够的样本
        assigned_labels = np.array([i for i in range(self.num_classes)] * \
            (ood_datalist_shape // self.num_classes) + [i for i in range(ood_datalist_shape % self.num_classes)])
        
        # 随机打乱标签顺序，增加数据的随机性
        np.random.shuffle(assigned_labels)
        
        # 将标签重新整形为批次形式，便于后续分配
        assigned_labels = assigned_labels.reshape(
            ood_data_sample_lens // ood_data_batch_size, ood_data_batch_size)
        
        # 遍历每个批次，处理数据和标签
        for batch_id, batch in enumerate(ood_datalist):
            data, targets = batch
            
            # # 处理不同数据集之间的通道数差异
            # if self.config.dataset.upper() == "EMNIST":
            #     # 如果主数据集是EMNIST（单通道），将OOD数据转换为单通道
            #     if hasattr(self.config, 'in_channels') and self.config.in_channels == 1:
            #         ood_datalist[batch_id] = (data[:, 0, :, :].unsqueeze(1), targets)
            # else:
            #     # 如果OOD数据是单通道而主数据集是三通道，将单通道数据重复为三通道
            #     if data.shape[1] == 1 and hasattr(self.config, 'in_channels') and self.config.in_channels == 3:
            #         ood_datalist[batch_id] = (data.repeat(1, 3, 1, 1), targets)

            # 为每个样本分配伪标签
            for ind in range(len(targets)):
                targets[ind] = assigned_labels[batch_id][ind]
        
        # 将处理后的数据列表转换为迭代器并返回
        ood_dataloader = iter(ood_datalist)
        return ood_dataloader

    def setup_ood_data(self, ood_data_sample_lens=800, ood_data_batch_size=64):
        """
        设置OOD数据，包括加载OOD数据集和创建数据加载器
        
        Args:
            ood_data_sample_lens: OOD数据采样数量
            ood_data_batch_size: OOD数据批次大小
        """
        self.load_ood_data()
        self.ood_data = self.get_ood_dataloader(ood_data_sample_lens, ood_data_batch_size)
        print(f"-----OOD数据加载器创建完成，采样数量: {ood_data_sample_lens}, 批次大小: {ood_data_batch_size}-----")