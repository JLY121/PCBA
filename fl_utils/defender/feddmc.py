import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from scipy.cluster.hierarchy import dendrogram
import os

# ============== PCA相关函数 ==============
def PCA_skl(X, n_components=2, random_state=0):
    """PCA降维函数"""
    # 使用公开API进行输入校验，避免依赖私有方法的版本差异
    X = check_array(X, accept_sparse=['csr'], ensure_min_samples=2,
                    dtype=[np.float32, np.float64])
    # 处理随机种子
    random_state = check_random_state(random_state)
    # 初始化PCA对象
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_state)
    # 用PCA拟合并转换数据
    X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
    u = pca.fit_transform(X)
    return X_embedded, u

# ============== 树结构相关类和函数 ==============
class Node(object):
    def __init__(self, index, lchild=None, rchild=None, distances=None, counts=None):
        self.index = index
        self.lchild = lchild
        self.rchild = rchild
        self.distances = distances
        self.counts = counts
        self.leaves = []

    def postorder_travel(self, node):
        """后序遍历获取叶子节点"""
        if node == None:
            return []
        
        self.postorder_travel(node.lchild)
        self.postorder_travel(node.rchild)
        if node.counts == 1:
            self.leaves.append(node.index)
        return self.leaves

def get_linkage_matrix(agglomer):
    """获取层次聚类的链接矩阵"""
    counts = np.zeros(agglomer.children_.shape[0])
    n_samples = len(agglomer.labels_)

    for i, merge in enumerate(agglomer.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    d = agglomer.distances_
    child = agglomer.children_
    linkage_matrix = np.column_stack([agglomer.children_, d, counts]).astype(float)

    return linkage_matrix

def Building_tree(linkage_matrix, n_samples):
    """构建二叉树"""
    cluster_id = n_samples
    queue = {}
    root = None

    for child in linkage_matrix:
        if child[0] < n_samples:
            lchild = Node(child[0], counts=1)
        else:
            lchild = queue[child[0]]
            del queue[child[0]]

        if child[1] < n_samples:
            rchild = Node(child[1], counts=1)
        else:
            rchild = queue[child[1]]
            del queue[child[1]]

        root = Node(cluster_id, lchild, rchild, child[2], child[3])
        queue[cluster_id] = root
        cluster_id = cluster_id + 1
    return root

def Removing_outliers(root, min_cluster_size=3):
    """去除异常值，识别良性和恶意客户端"""
    outlier_all = []
    n_clients = root.counts

    while root.rchild.counts <= min_cluster_size or root.lchild.counts <= min_cluster_size:
        if root.rchild.counts >= min_cluster_size:
            outlier = root.lchild.postorder_travel(root.lchild)
            root = root.rchild
        elif root.lchild.counts >= min_cluster_size:
            outlier = root.rchild.postorder_travel(root.rchild)
            root = root.lchild
        else:
            outlier = root.postorder_travel(root)
            outlier_all.extend(outlier)
            root = None
            break
        outlier_all.extend(outlier)
        if len(outlier_all) > (n_clients // 2):
            break

    # root中，左右孩子分别是良性和恶意用户
    benign, malicious = [], []
    if root:
        if root.lchild.counts > (n_clients // 2) or root.rchild.counts > (n_clients // 2):
            if root.rchild.counts < root.lchild.counts:
                malicious = root.rchild.postorder_travel(root.rchild)
                benign = root.lchild.postorder_travel(root.lchild)
            elif root.rchild.counts > root.lchild.counts:
                benign = root.rchild.postorder_travel(root.rchild)
                malicious = root.lchild.postorder_travel(root.lchild)
        else:
            benign = root.postorder_travel(root)
    return benign, malicious, outlier_all

def mkdirs(dirpath):
    """创建目录"""
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def parameters_dict_to_vector(net_dict):
    """将参数字典转换为向量形式，只包含weight和bias"""
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def feddmc_aggregate(global_model, weight_accumulator_by_client, sampled_participants, helper, 
                    malicious_records, window_size, vote_threshold):
    """
    FEDDMC聚合方法实现 (基于PCA + 层次聚类的恶意客户端检测 + 多轮投票机制)
    
    参数:
    - global_model: 全局模型
    - weight_accumulator_by_client: 每个客户端的权重更新
    - sampled_participants: 采样的客户端列表
    - helper: 辅助对象，包含配置信息
    - malicious_records: 历史恶意检测记录列表（引用传递，会被修改）
    - window_size: 滑动窗口大小
    - vote_threshold: 投票阈值
    """
    print(f"---Using FEDDMC defender with multi-round voting---")
    
    # ============== 防御强度调整参数 ==============
    # PCA降维维度 - 控制特征压缩程度，越小压缩越多，可能丢失更多信息但计算更快
    pca_d = getattr(helper.config, 'feddmc_pca_dim', 10)
    
    # 最小聚类大小 - 控制异常值检测敏感度，越大越严格，可能误删良性客户端
    min_cluster_size = getattr(helper.config, 'feddmc_min_cluster_size', 3)
    
    # 是否保存聚类树图像（用于调试分析）
    save_tree_plot = getattr(helper.config, 'feddmc_save_tree', False)
    
    # 提取客户端参数向量
    user_grads = []
    for client_update in weight_accumulator_by_client:
        # 将客户端更新转换为向量
        local_parameters = parameters_dict_to_vector(client_update)
        
        # 如果是cifar10数据集，进行特殊处理（根据原始代码）
        if helper.config.dataset == 'cifar10' and len(local_parameters) % 2 == 0:
            local_parameters = (local_parameters.reshape(-1, 2))[:, 0]
        
        # 拼接所有客户端的参数向量
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)
    
    # 转换为numpy数组进行PCA处理
    param = user_grads.cpu().numpy()
    
    # PCA降维
    param_pca, _ = PCA_skl(param, pca_d)
    print(f"FEDDMC: PCA降维 {param.shape} -> {param_pca.shape}")
    
    # 层次聚类 - 将客户端聚类成2类（良性vs恶意）
    agglomer = AgglomerativeClustering(
        n_clusters=2, 
        linkage='ward', 
        compute_distances=True
    ).fit(param_pca)
    
    # 构建链接矩阵用于后续树分析
    linkage_matrix = get_linkage_matrix(agglomer)
    
    # 可选：保存聚类树图像用于分析
    if save_tree_plot and hasattr(helper, 'current_epoch'):
        current_epoch = helper.current_epoch
        if current_epoch % 10 == 0:  # 每10轮保存一次
            try:
                fig = plt.figure(figsize=(16, 12))
                dendrogram(linkage_matrix, distance_sort=True, count_sort=True)
                plt.title(f'FEDDMC Hierarchical Clustering Tree - Epoch {current_epoch}')
                
                # 创建保存目录
                log_dir = getattr(helper.config, 'log_dir', './logs')
                tree_dir = os.path.join(log_dir, 'feddmc_tree')
                mkdirs(tree_dir)
                
                # 保存图像
                fig.savefig(os.path.join(tree_dir, f'agglom_epoch_{current_epoch}.png'))
                plt.close(fig)
                print(f"FEDDMC: 聚类树图像已保存到 {tree_dir}")
            except Exception as e:
                print(f"FEDDMC: 保存聚类树图像失败: {e}")
    
    # 构建二叉树进行异常值分析
    tree = Building_tree(linkage_matrix, len(agglomer.labels_))
    
    # 去除异常值，识别良性和恶意客户端
    benign_indices, malicious_indices, outlier_indices = Removing_outliers(tree, min_cluster_size)
    
    # 生成最终标签：0=良性，1=恶意，-1=异常值
    labels = np.ones(len(agglomer.labels_))  # 默认标记为恶意
    for value in benign_indices:
        labels[int(value)] = 0  # 标记为良性
    
    # ============== 多轮投票机制实现 ==============
    # 将当前轮的检测结果添加到历史记录中
    malicious_records.append(labels)
    
    # 维护滑动窗口：只保留最近window_size轮的检测结果
    if len(malicious_records) > window_size:
        malicious_records.pop(0)  # 移除最旧的记录
    
    print(f"FEDDMC: 当前轮检测结果: {labels}")
    print(f"FEDDMC: 历史记录窗口大小: {len(malicious_records)}/{window_size}")
    
    # 多轮投票：累加所有历史记录中的恶意标记
    if len(malicious_records) > 0:
        sum_records = np.sum(malicious_records, axis=0)
        print(f"FEDDMC: 累积恶意得分: {sum_records}")
        
        # 根据投票阈值确定最终的恶意客户端
        detected_malicious_clients = []
        benign_clients = []
        
        for i, accumulated_score in enumerate(sum_records):
            # 如果累积得分超过阈值*历史记录数，则认为是恶意客户端
            if accumulated_score > vote_threshold * len(malicious_records):
                detected_malicious_clients.append(i)
            else:
                benign_clients.append(i)
        
        print(f"FEDDMC: 多轮投票后检测到的恶意客户端索引: {detected_malicious_clients}")
        print(f"FEDDMC: 多轮投票后确认的良性客户端索引: {benign_clients}")
    else:
        # 如果没有历史记录，使用当前轮的结果
        benign_clients = []
        for i, label in enumerate(labels):
            if label == 0:  # 良性客户端
                benign_clients.append(i)
    
    # 容错机制：如果没有检测到良性客户端，使用所有客户端
    if len(benign_clients) == 0:
        print("FEDDMC警告: 多轮投票后未检测到良性客户端，使用所有客户端进行聚合")
        benign_clients = list(range(len(weight_accumulator_by_client)))
    
    print(f"FEDDMC: 最终用于聚合的客户端数量: {len(benign_clients)}/{len(weight_accumulator_by_client)}")
    print(f"FEDDMC: 最终用于聚合的客户端索引: {benign_clients}")
    
    # ============== 对选择的良性客户端进行平均聚合 ==============
    lr = 1
    for name, data in global_model.state_dict().items():
        if name == 'decoder.weight':
            continue
        
        # 计算良性客户端的平均更新
        total_update = torch.zeros_like(data)
        for client_idx in benign_clients:
            if name in weight_accumulator_by_client[client_idx]:
                client_update = weight_accumulator_by_client[client_idx][name]
                total_update += client_update
        
        # 应用平均更新
        avg_update = total_update / len(benign_clients) * lr
        avg_update = torch.tensor(avg_update, dtype=data.dtype)
        data.add_(avg_update.cuda())
    
    return True
