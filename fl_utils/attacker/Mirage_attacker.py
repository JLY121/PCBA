import sys
sys.path.append("../")
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

class Mirage_Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()
    
    def setup(self):
        """初始化触发器和掩码"""
        self.handcraft_rnds = 0
        # 根据数据集的图像尺寸，初始化触发器和掩码
        image_size = self.helper.config.image_size  
        in_channels = self.helper.config.in_channels  
        self.trigger = torch.ones((1, in_channels, image_size, image_size), requires_grad=False, device='cuda') * 0.5
        self.mask = torch.zeros_like(self.trigger)  # 触发器的掩码
        
        # 根据mask_type参数设置不同的掩码模式
        mask_type = getattr(self.helper.config, 'mask_type', 'local')  # 默认为local模式
        
        if mask_type == 'local':
            # 局部触发器模式：在指定位置设置触发器
            self.mask[:, :, 2:2+self.helper.config.trigger_size_h, 2:2+self.helper.config.trigger_size_w] = 1
            print(f"Mirage攻击使用局部触发器模式，触发器大小: {self.helper.config.trigger_size_h}x{self.helper.config.trigger_size_w}")
        elif mask_type == 'global':
            # 全局触发器模式：整个图像都作为触发器
            self.mask[:, :, :, :] = 1
            print(f"Mirage攻击使用全局触发器模式，触发器大小: {image_size}x{image_size}")
        else:
            # 未知模式，默认使用局部模式并给出警告
            print(f"警告：未知的mask_type '{mask_type}'，使用默认的局部触发器模式")
            self.mask[:, :, 2:2+self.helper.config.trigger_size_h, 2:2+self.helper.config.trigger_size_w] = 1
        
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone()  # 保存初始触发器
        
        # 根据模型类型确定分类器名称
        if "resnet" in self.helper.config.model.lower():
            self.classifier_name = "linear"
        elif "vgg" in self.helper.config.model.lower():
            self.classifier_name = "classifier"
        elif "mobilenet" in self.helper.config.model.lower():
            self.classifier_name = "classifier"
        else:
            self.classifier_name = "linear"  # 默认值
    
    def poisoned_batch_injection(self, batch, trigger, mask, is_eval=False, label_swap=None):
        """
        批量数据后门注入函数
        """
        inputs, labels = batch
        
        mask_type = getattr(self.helper.config, 'mask_type', 'local')
        blend_alpha = getattr(self.helper.config, 'blend_alpha', None)

        if mask_type == 'global' and blend_alpha is not None:
            # 全局触发器使用Blend方法
            def injection_logic(data):
                return trigger * blend_alpha + (1 - blend_alpha) * data
        else:
            # 局部触发器使用原始方法
            def injection_logic(data):
                return trigger * mask + (1 - mask) * data

        if is_eval:
            # 评估时对所有样本注入后门
            poisoned_inputs = injection_logic(inputs)
            if label_swap is not None:
                poisoned_labels = torch.full_like(labels, label_swap)
            else:
                poisoned_labels = labels
        else:
            # 训练时按比例注入后门
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
            poisoned_inputs = inputs.clone()
            poisoned_labels = labels.clone()
            poisoned_inputs[:bkd_num] = injection_logic(inputs[:bkd_num])
            if label_swap is not None:
                poisoned_labels[:bkd_num] = label_swap
        
        return poisoned_inputs, poisoned_labels
    
    def generate_discriminator_dataloader(self, model, train_loader, trigger, mask, target_class):
        """
        生成训练判别器的数据集，判别器用于区分正常样本和后门样本
        """
        class_num = self.helper.config.discriminator_class_num
        # 为每个类别初始化一个空的样本张量
        samples_per_class = {i: torch.tensor([], device='cuda') for i in range(class_num)}
        criterion = nn.CrossEntropyLoss(reduction='none').to('cuda')
        label_list = [0 for _ in range(class_num)]
        
        # 遍历训练集，按类别收集样本
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            for class_ind in range(class_num):
                indices = labels == class_ind
                label_list[class_ind] += sum(indices)
                samples_per_class[class_ind] = torch.cat((samples_per_class[class_ind], inputs[indices]), dim=0)
        
        # 选取每类的代表性样本
        for i in range(class_num):
            sample = samples_per_class[i]
            if len(sample) == 0:
                continue
            outputs = model(sample)
            tmp_label = torch.ones(len(outputs), dtype=torch.long, device='cuda') * i
            loss_sort_by_samples = criterion(outputs, tmp_label)
            
            # 选取损失最小的若干个样本
            samples_selected_len = self.helper.config.discriminator_train_samples_pre_class if len(outputs) > self.helper.config.discriminator_train_samples_pre_class else len(outputs)
            if i == target_class:
                samples_selected_len = len(outputs)  # 目标类别保留全部样本
            _, indices = torch.topk(loss_sort_by_samples, samples_selected_len, largest=False)
            representative_samples = sample[indices]
            samples_per_class[i] = representative_samples
        
        # 构建判别器训练集
        samples_discriminator_dataloader = torch.tensor([], device='cuda')
        labels_discriminator_dataloader = torch.tensor([], dtype=torch.long, device='cuda')
        
        for i in range(class_num):
            if i == target_class:
                continue  # 目标类别不做后门注入
            samples = samples_per_class[i]
            labels = torch.ones(len(samples), dtype=torch.long, device='cuda')  # 后门样本标签为1
            # 对非目标类别样本注入trigger和mask
            poisoned_sample, _ = self.poisoned_batch_injection((samples, labels), trigger=trigger, mask=mask, is_eval=True, label_swap=target_class)
            samples_discriminator_dataloader = torch.cat((samples_discriminator_dataloader, poisoned_sample), dim=0)
            labels_discriminator_dataloader = torch.cat((labels_discriminator_dataloader, labels), dim=0)
        
        # 目标类别的代表性样本直接作为正常样本，标签为0
        samples_discriminator_dataloader = torch.cat((samples_discriminator_dataloader, samples_per_class[target_class]), dim=0)
        labels_discriminator_dataloader = torch.cat((labels_discriminator_dataloader, torch.zeros(len(samples_per_class[target_class]), dtype=torch.long, device='cuda')), dim=0)
        
        # 打包成DataLoader
        discriminator_dataloader = DataLoader(
            TensorDataset(samples_discriminator_dataloader, labels_discriminator_dataloader),
            batch_size=self.helper.config.discriminator_batch_size, shuffle=True)
        
        return discriminator_dataloader
    
    def get_discriminator(self, model, discriminator_dataloader):
        """
        基于当前全局模型，构建一个二分类判别器
        """
        discriminator = copy.deepcopy(model)
        
        # 针对不同模型结构，替换为二分类头
        if "resnet" in self.helper.config.model.lower():
            discriminator.linear = torch.nn.Sequential(
                torch.nn.Linear(discriminator.linear.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "vgg" in self.helper.config.model.lower():
            discriminator.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator.classifier.in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        elif "mobilenet" in self.helper.config.model.lower():
            discriminator.classifier = torch.nn.Sequential(
                torch.nn.Linear(discriminator.classifier[1].in_features, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 2)
            )
        
        # 冻结特征提取层，只训练分类头
        for name, param in discriminator.named_parameters():
            if self.classifier_name not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 定义优化器和损失函数
        discriminator_optimizer = torch.optim.SGD(
            discriminator.parameters(), 
            lr=self.helper.config.discriminator_lr, 
            momentum=self.helper.config.discriminator_momentum, 
            weight_decay=self.helper.config.discriminator_weight_decay
        )
        discriminator_criterion = nn.CrossEntropyLoss().to('cuda')
        discriminator = discriminator.to('cuda')
        
        # 多轮训练判别器
        for iter in range(self.helper.config.discriminator_train_no_times):
            total_loss = 0.
            for batch in discriminator_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = discriminator(inputs)
                loss = discriminator_criterion(outputs, labels)
                discriminator_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                total_loss += loss.item()
                discriminator_optimizer.step()
        
        discriminator.eval()
        return discriminator
    
    def search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        """
        Mirage触发器优化方法，结合判别器增强隐蔽性
        """
        model.eval()
        target_class = self.helper.config.target_class
        ce_loss = nn.functional.cross_entropy
        cos_loss = nn.CosineSimilarity(dim=1, eps=1e-08)
        
        t = self.trigger.clone()
        m = self.mask.clone()
        
        # 触发器优化器
        trigger_optim = torch.optim.Adam([t], lr=self.helper.config.trigger_lr, weight_decay=5e-4)
        
        # 多轮优化触发器
        for iters in tqdm(range(self.helper.config.trigger_search_no_times)):
            # 生成判别器训练集
            dataloader_discriminator = self.generate_discriminator_dataloader(model, dl, t, m, target_class)
            
            # 训练判别器
            model_discriminator = self.get_discriminator(model, dataloader_discriminator)
            
            # 遍历训练集，优化触发器
            for inputs, targets in dl:
                t.requires_grad_()
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                
                # 找到目标类别和非目标类别的样本
                batch_clean_indices = targets == target_class
                if batch_clean_indices.sum() == 0:
                    continue
                
                batch_backdoor_indices = ~batch_clean_indices
                backdoor_inputs = inputs[batch_backdoor_indices]
                backdoor_targets = targets[batch_backdoor_indices]
                
                if len(backdoor_inputs) == 0:
                    continue
                
                # 注入后门
                backdoor_inputs, backdoor_targets = self.poisoned_batch_injection(
                    (backdoor_inputs, backdoor_targets), 
                    trigger=t, mask=m, is_eval=False, label_swap=target_class
                )
                backdoor_inputs = backdoor_inputs.to('cuda')
                
                # 判别器输出
                backdoor_pred_disc = model_discriminator(backdoor_inputs)
                
                # 判别器损失（希望判别器难以区分后门样本）
                loss_discriminator = ce_loss(
                    backdoor_pred_disc, 
                    torch.zeros(len(backdoor_pred_disc), device='cuda').long()
                )
                
                # 主模型输出
                backdoor_pred = model(backdoor_inputs)
                
                # 攻击成功率损失（希望后门样本被误判为目标类别）
                loss_asr = ce_loss(backdoor_pred, backdoor_targets)
                
                # 特征相似性损失（希望后门样本与原样本特征接近）
                original_pred = model(inputs[batch_backdoor_indices])
                loss_sim = -cos_loss(backdoor_pred, original_pred).mean()
                
                # 总损失
                # 权重值分别为：1/1/1
                loss = (self.helper.config.mirage_discriminator_weight * loss_discriminator +   
                       self.helper.config.mirage_asr_weight * loss_asr + 
                       self.helper.config.mirage_sim_weight * loss_sim)
                
                # 反向传播优化触发器
                if loss is not None and loss.item() != 0.:
                    trigger_optim.zero_grad()
                    loss.backward(retain_graph=True)
                    new_t = t - t.grad.sign() * self.helper.config.trigger_lr
                    t = new_t.detach()
                    t = torch.clamp(t, min=-self.helper.config.trigger_epsilon, max=self.helper.config.trigger_epsilon)
                    t.requires_grad_()
        
        t = t.detach()
        self.trigger = t
        self.mask = m
    
    def train_malicious(self, participant_id, model, epoch, lr):
        """
        恶意客户端的后门攻击训练方法
        """
        # 优化器、损失函数
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
        
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            total_loss = 0.0
            for inputs, labels in self.helper.train_data[participant_id]:
                # if inputs.size(0) == 1:  # 如果输入的batchsize为1，则跳过
                #     continue
                inputs, labels = inputs.cuda(), labels.cuda()
                # 对输入数据进行后门攻击，注入优化后的触发器
                inputs, labels = self.poison_input(inputs, labels)
                output = model(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def poison_input(self, inputs, labels, eval=False):
        """
        对输入数据注入后门触发器
        """
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        
        mask_type = getattr(self.helper.config, 'mask_type', 'local')
        blend_alpha = getattr(self.helper.config, 'blend_alpha', None)

        if mask_type == 'global' and blend_alpha is not None:
            # 全局触发器使用Blend方法
            inputs[:bkd_num] = self.trigger * blend_alpha + inputs[:bkd_num] * (1 - blend_alpha)
        else:
            # 局部触发器使用原始方法
            inputs[:bkd_num] = self.trigger * self.mask + inputs[:bkd_num] * (1 - self.mask)  

        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels
