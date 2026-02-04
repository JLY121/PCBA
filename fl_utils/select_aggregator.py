#=======学长给的防御方法的实现，就是基于A3FL的聚合代码添加的========
import sys
sys.path.append("../")
import torch
import numpy as np

# 导入所有聚合方法
from fl_utils.defender import *

def get_total_samples(data_loader):
    total_samples = 0
    for data in data_loader:
        total_samples += data[0].size(0)
    return total_samples

class Aggregator:
    def __init__(self, helper):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []
        self.sum_updates =[]

        # 初始化客户端数量和特征维度等参数
        self.n_clients = self.helper.config.num_sampled_participants
        self.n_features = None  # 稍后根据模型参数确定
        self.wv = np.ones(self.n_clients)  # 初始化权重向量
        # 添加历史更新的存储，使用字典来保存每个客户端的累积更新
        self.history_updates = {}  # key: client_id, value: cumulative update vector
        
        # ============== FEDDMC 相关历史记录维护 ==============
        # 用于记录每轮的恶意客户端检测结果，实现多轮投票机制
        self.feddmc_malicious_records = []  # 存储每轮的检测结果（标签数组）
        self.feddmc_window_size = getattr(self.helper.config, 'feddmc_window_size', 10)  # 滑动窗口大小
        self.feddmc_vote_threshold = getattr(self.helper.config, 'feddmc_vote_threshold', 0.5)  # 投票阈值

    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants, epoch):
        """
        聚合方法的分发器，根据配置选择相应的聚合方法
        """
        if self.helper.config.agg_method == 'avg':
            return average_shrink_models(global_model, weight_accumulator, self.helper)
        elif self.helper.config.agg_method == 'fedprox':
            return fedprox_aggregate(global_model, weight_accumulator, self.helper)
        elif self.helper.config.agg_method == 'fednova':
            return fednova_aggregate_global_model(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        elif self.helper.config.agg_method == 'clip':
            return clip_aggregate(global_model, weight_accumulator, self.helper)
        elif self.helper.config.agg_method == 'krum':
            return krum(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        elif self.helper.config.agg_method == 'multi_krum':
            return multi_krum(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        elif self.helper.config.agg_method == 'dp':
            return dp_avg(global_model, weight_accumulator, self.helper)
        elif self.helper.config.agg_method == 'median':
            return median_shrink_models(global_model, weight_accumulator_by_client, self.helper)
        elif self.helper.config.agg_method == 'rfa':
            return geometric_median_update(global_model, weight_accumulator_by_client)
        elif self.helper.config.agg_method == 'rlr':
            return robust_lr_aggregate(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        elif self.helper.config.agg_method == 'crfl':
            return crfl_agg(global_model, weight_accumulator, self.helper)
        elif self.helper.config.agg_method == 'bulyan':
            return bulyan_aggregate_global_model(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        elif self.helper.config.agg_method == 'deep':
            return deepsight_aggregate_global_model_v2(client_models, sampled_participants, global_model, weight_accumulator_by_client, self.helper)
        elif self.helper.config.agg_method == 'foolsgold':
            return foolsgold_aggregate(global_model, weight_accumulator_by_client, sampled_participants, self.helper, self.history_updates, self.n_features, self.wv)
        elif self.helper.config.agg_method == 'flame':
            return flame_aggregate(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        elif self.helper.config.agg_method == 'feddmc':
            return feddmc_aggregate(global_model, weight_accumulator_by_client, sampled_participants, self.helper, 
                                  self.feddmc_malicious_records, self.feddmc_window_size, self.feddmc_vote_threshold)
        elif self.helper.config.agg_method == 'alignins':
            return alignins_aggregate(global_model, weight_accumulator_by_client, sampled_participants, self.helper)
        else:
            raise NotImplementedError(f"Aggregation method '{self.helper.config.agg_method}' not implemented") 