import sys
sys.path.append("../")
import time
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18
import copy
import os
from math import ceil
import pickle
# 保存触发器图片（部分攻击方法可能会用到）
from PIL import Image
import csv

from .select_attacker import AttackerDispatcher
from .select_aggregator import Aggregator
from .vis_train import TrainProcessVisualizer

class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.helper = helper
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.attack_sum = 0 
        self.aggregator = Aggregator(self.helper)
        self.start_time = time.time()
        # 训练过程可视化（CSV + 曲线图），每次训练自动新建保存子文件夹
        self.train_visualizer = TrainProcessVisualizer(self.helper.config)
        self.attacker_dispatcher = AttackerDispatcher(self.helper)
        self.attacker = self.attacker_dispatcher.get_attacker()

        if self.helper.config.sample_method == 'random_updates':
            self.init_advs()
        if self.helper.config.load_benign_model:
            model_path = f'../saved/benign_new_{self.helper.config.model}/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.agg_method}.pt'
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location = 'cuda')['model'])
            loss,acc = self.test_once()
            print(f'Load benign model {model_path}, acc {acc:.3f}')
        return
    
    def init_advs(self):
        """random_updates 采样方式下：预先确定哪些更新步骤由恶意客户端触发。"""
        num_updates = self.helper.config.num_sampled_participants * self.helper.config.poison_epochs 
        num_poison_updates = ceil(self.helper.config.sample_poison_ratio * num_updates)
        updates = list(range(num_updates))
        advs = np.random.choice(updates, num_poison_updates, replace=False)
        print(f'Using random updates, sampled {",".join([str(x) for x in advs])}')
        adv_dict = {}
        for adv in advs:
            epoch = adv//self.helper.config.num_sampled_participants
            idx = adv % self.helper.config.num_sampled_participants
            if epoch in adv_dict:
                adv_dict[epoch].append(idx)
            else:
                adv_dict[epoch] = [idx]
        self.advs = adv_dict

    def test_once(self, poison = False):
        model = self.helper.global_model
        model.eval()
        with torch.no_grad():
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0) 
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        return loss, acc
    
    def test_local_once(self, model, poison = False):
        print("本地测试时poison参数为：", poison)
        model.eval()
        with torch.no_grad():
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        acc = 100.0 * (float(correct) / float(num_data))
        loss = total_loss / float(num_data)
        model.train()
        return loss, acc
    
    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc, lr):
        log_dict = {
            'epoch': epoch, 
            'test_acc': acc,
            'test_loss': loss, 
            'bkd_acc': bkd_acc,
            'bkd_loss': bkd_loss,
            'lr': lr
            }
        wandb.log(log_dict)
        print('|'.join([f'{k}:{float(log_dict[k]):.3f}' for k in log_dict]))
        self.save_model(epoch, log_dict)
        # 追加 CSV 并更新可视化图表（失败不影响训练）
        try:
            self.train_visualizer.log(epoch, loss, acc, bkd_loss, bkd_acc, lr)
        except Exception as e:
            print(f"[TrainProcessVisualizer] 记录/可视化失败（不影响训练）：{e}")

    def save_model(self, epoch, log_dict):
        if epoch % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            if self.helper.config.is_poison:
                pass
            else:
                save_path = f'../saved/benign_new_{self.helper.config.model}/{self.helper.config.dataset}_{epoch}_{self.helper.config.agg_method}.pt'
                torch.save(log_dict, save_path)
                print(f'Model saved at {save_path}')
    
    def save_res(self, accs, asrs):
        log_dict = {
            'accs': accs,
            'asrs': asrs
        }
        atk_method = self.helper.config.attacker_method
        if self.helper.config.sample_method == 'random':
            file_name = f'{self.helper.config.dataset}/{self.helper.config.agg_method}_{atk_method}_r_{self.helper.config.num_adversaries}_{self.helper.config.poison_epochs}_ts{self.helper.config.trigger_size}.pkl'
        else:
            raise NotImplementedError
        save_path = os.path.join('../saved/res', file_name)
        dir_path = os.path.dirname(save_path) 
        if not os.path.exists(dir_path): 
            os.makedirs(dir_path)
        with open(save_path, 'wb') as f_save:  
            pickle.dump(log_dict, f_save)  
        print(f'训练结果已保存到 {save_path}')     

    def train(self):
        print('Training')
        accs = []
        asrs = []
        self.local_asrs = {}  # 存储各本地客户端的后门攻击成功率
        
        for epoch in range(-2, self.helper.config.epochs):
            sampled_participants = self.sample_participants(epoch)
            weight_accumulator, weight_accumulator_by_client = self.train_once(epoch, sampled_participants)
            self.aggregator.agg(self.helper.global_model, weight_accumulator, weight_accumulator_by_client, self.helper.client_models, sampled_participants, epoch)
            
            loss, acc = self.test_once()
            bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison)
            lr = self.get_lr(epoch)
            self.log_once(epoch, loss, acc, bkd_loss, bkd_acc, lr)

    def write_to_file(self, file_path, content, mode='a'):
        try:
            with open(file_path, mode, encoding='utf-8') as file:
                file.write(content)
        except Exception as e:
            print(f"写入文件时出错: {e}")   

    def train_once(self, epoch, sampled_participants):
        weight_accumulator = self.create_weight_accumulator()
        weight_accumulator_by_client = []
        client_count = 0
        attacker_idxs = []
        global_model_copy = self.create_global_model_copy()
        local_asr = []
        first_adversary = self.contain_adversary(epoch, sampled_participants)
        if first_adversary >= 0 and ('sin' in self.helper.config.attacker_method):
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            self.attacker.search_trigger(model, self.helper.train_data[first_adversary], 'outter', first_adversary, epoch)
            print("触发器搜索优化完成")
        if first_adversary >= 0:
            # 统计训练过程中触发攻击的轮次
            self.attack_sum += 1
            print(f'Epoch {epoch}, poisoning by {first_adversary}, attack sum {self.attack_sum}.')
        else:
            print(f'Epoch {epoch}, no adversary.')

        for participant_id in sampled_participants:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            model.train()
            is_adversary = self.if_adversary(epoch, participant_id, sampled_participants)
            if not is_adversary:
                self.train_benign(participant_id, model, epoch)
            else: 
                attacker_idxs.append(client_count)
                if self.helper.config.two_stage_training == True:
                    print("-----良性训练+后门训练方式-----")
                    self.train_benign(participant_id, model, epoch)
                self.attacker.train_malicious(participant_id, model, epoch, lr=self.get_lr(epoch))
                bd_loss, asr = self.test_local_once(model, poison=True)
                print("训练后的后门模型的loss：",bd_loss,"asr：", asr)

            weight_accumulator, single_wa = self.update_weight_accumulator(model, weight_accumulator)
            weight_accumulator_by_client.append(single_wa)
            self.helper.client_models[participant_id].load_state_dict(model.state_dict())
            client_count += 1
        return weight_accumulator, weight_accumulator_by_client

    def norm_of_update(self, single_wa_by_c, attacker_idxs):
        cossim = torch.nn.CosineSimilarity(dim=0)
        def sim_was(wa1, wa2):
            sim = 0.0
            for name in wa1:
                v1 = wa1[name]
                v2 = wa2[name]
                if v1.dtype == torch.float:
                    sim += cossim(v1.view(-1),v2.view(-1)).item()
            return sim
        count = 0
        sim_sum = 0.
        for i in range(len(single_wa_by_c)):
            for j in range(len(single_wa_by_c)):
                if i in attacker_idxs and i != j:
                    sim = sim_was(single_wa_by_c[i], single_wa_by_c[j])
                    sim_sum += sim
                    count += 1
        return sim_sum/count

    def contain_adversary(self, epoch, sampled_participants):
        if self.helper.config.is_poison and epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random':
                for p in sampled_participants:
                    # 约定：客户端 id < num_adversaries 视为恶意客户端
                    if p < self.helper.config.num_adversaries:
                        return p
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    return self.advs[epoch][0]
        return -1

    def contain_adversary_preTrain(self, epoch, sampled_participants):
        """
        从头开始训练时（例如未加载良性模型）用于检查本轮是否包含恶意客户端。
        """
        if self.helper.config.sample_method == 'random':
            for p in sampled_participants:
                if p < self.helper.config.num_adversaries:
                    return p
        elif self.helper.config.sample_method == 'random_updates':
            if epoch in self.advs:
                return self.advs[epoch][0]
        return -1

    def num_attackers(self, epoch, sampled_participants):
        n = 0
        if self.helper.config.is_poison and \
            epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random':
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        n += 1
        return n

    def if_adversary(self, epoch, participant_id, sampled_participants):
        # 只有攻击模式（is_poison）且在攻击轮次范围内才进行后门训练
        if self.helper.config.is_poison and epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'random' and participant_id < self.helper.config.num_adversaries:
                return True 
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    for idx in self.advs[epoch]:
                        if sampled_participants[idx] == participant_id:
                            return True
        return False

    def create_local_model_copy(self, model):
        """返回模型参数的冻结拷贝（用于后续对比/还原）。"""
        model_copy = dict()
        for name, param in model.named_parameters():
            model_copy[name] = model.state_dict()[name].clone().detach().requires_grad_(False)
        return model_copy

    def create_global_model_copy(self):
        """返回全局模型参数的冻结拷贝。"""
        global_model_copy = dict()
        for name, param in self.helper.global_model.named_parameters():
            global_model_copy[name] = self.helper.global_model.state_dict()[name].clone().detach().requires_grad_(False)
        return global_model_copy
    
    

    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            # tied weights/特殊字段不参与累加
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator
    
    def update_weight_accumulator(self, model, weight_accumulator):
        single_weight_accumulator = dict()
        for name, data in model.state_dict().items():
            if name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - self.helper.global_model.state_dict()[name])
            single_weight_accumulator[name] = data - self.helper.global_model.state_dict()[name]
        return weight_accumulator, single_weight_accumulator

    def train_benign(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=self.helper.config.momentum,
                                    weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.retrain_times):
            total_loss = 0.0
            for inputs, labels in self.helper.train_data[participant_id]:
                # 避免 batchsize=1 触发 BatchNorm 异常
                if inputs.size(0) == 1:
                    continue
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = self.criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def scale_up(self, model, curren_num_adv):
        clip_rate = 2/curren_num_adv
        for key, value in model.state_dict().items():
            # tied weights/特殊字段不参与缩放
            if  key == 'decoder.weight' or '__'in key:
                continue
            target_value = self.helper.global_model.state_dict()[key]
            new_value = target_value + (value - target_value) * clip_rate

            model.state_dict()[key].copy_(new_value)
        return model

    def get_lr(self, epoch):
        if self.helper.config.lr_method == 'exp':
            tmp_epoch = epoch
            if self.helper.config.is_poison and self.helper.config.load_benign_model:
                tmp_epoch += self.helper.config.poison_start_epoch
            lr = self.helper.config.lr * (self.helper.config.gamma**tmp_epoch)
        elif self.helper.config.lr_method == 'linear':
            # 攻击阶段使用固定学习率
            if self.helper.config.is_poison:
                if self.helper.config.dataset == "GTSRB":
                    lr = 0.005
                elif self.helper.config.dataset == "cifar100":
                    lr = 0.002
                else:
                    lr = 0.002
            else:
                lr_init = self.helper.config.lr
                target_lr = self.helper.config.target_lr
                if epoch <= self.helper.config.epochs/2.:
                    lr = epoch*(target_lr - lr_init)/(self.helper.config.epochs/2.-1) + lr_init - (target_lr - lr_init)/(self.helper.config.epochs/2. - 1)
                else:
                    lr = (epoch-self.helper.config.epochs/2)*(-target_lr)/(self.helper.config.epochs/2) + target_lr
                if lr <= 0.002:
                    lr = 0.002
        elif self.helper.config.lr_method == 'warmup_cosine':
            import math
            total_epochs = self.helper.config.epochs
            warmup_epochs = int(total_epochs / 2)  
            initial_lr = self.helper.config.lr
            peak_lr = self.helper.config.target_lr
            min_lr = 0.0  
            if epoch < warmup_epochs:
                progress = epoch / warmup_epochs
                lr = initial_lr + progress * (peak_lr - initial_lr)
            
            else:
                decay_epochs = total_epochs - warmup_epochs
                current_decay_epoch = epoch - warmup_epochs
                progress = current_decay_epoch / decay_epochs
                
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = min_lr + (peak_lr - min_lr) * cosine_decay
            if lr <= 0.002:
                lr = 0.002
        else:
            raise NotImplementedError
        return lr

    def _sample_participants_continuous_attack(self, epoch):
        """
        为连续攻击场景采样参与者，确保每轮至少有一个攻击者。
        """
        num_adversaries = self.helper.config.num_adversaries
        num_sampled = self.helper.config.num_sampled_participants
        num_total = self.helper.config.num_total_participants

        adversary_ids = list(range(num_adversaries))
        chosen_adversary = random.choice(adversary_ids)
        
        other_participants_pool = [p for p in range(num_total) if p != chosen_adversary]
        num_to_sample_others = num_sampled - 1
        
        sampled_others = random.sample(other_participants_pool, num_to_sample_others)
        
        sampled_participants = [chosen_adversary] + sampled_others
        random.shuffle(sampled_participants)
        
        print(f"Epoch {epoch} [连续攻击]: 选中客户端 {sampled_participants}, 其中攻击者为 {chosen_adversary}")
        
        return sampled_participants

    def sample_participants(self, epoch):
        """
        采样当前通信轮次的参与客户端
        - 根据不同的采样方法从所有可用客户端中选择一定数量的客户端参与本轮训练
        返回:
        - sampled_participants: 选中的客户端ID列表
        """
        # 连续攻击：确保每轮至少有一个攻击者
        if hasattr(self.helper.config, 'continuous_attack') and \
           self.helper.config.continuous_attack and \
           self.helper.config.is_poison and \
           epoch < self.helper.config.continuous_attack_epochs and epoch >= 0:
            print(f"Epoch {epoch}: 启动连续攻击采样策略。")
            return self._sample_participants_continuous_attack(epoch)
        
        if self.helper.config.sample_method in ['random', 'random_updates']:
            sampled_participants = random.sample(
                range(self.helper.config.num_total_participants),
                self.helper.config.num_sampled_participants
            )
        elif self.helper.config.sample_method == 'fix-rate':
            start_index = (epoch * self.helper.config.num_sampled_participants) % self.helper.config.num_total_participants
            sampled_participants = list(range(start_index, start_index + self.helper.config.num_sampled_participants))
        else:
            raise NotImplementedError  
        
        assert len(sampled_participants) == self.helper.config.num_sampled_participants
        
        return sampled_participants
    
    def copy_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])