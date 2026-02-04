import csv
import numpy as np
from scipy.stats import ttest_ind
from typing import List, Tuple, Optional
import os


class FLConvergenceDetector:
    """
    联邦学习收敛检测器
    
    用于检测联邦学习过程中全局模型的收敛状态，基于验证集上的loss或accuracy变化趋势
    使用滑动窗口和统计检验方法进行在线收敛检测
    """
    
    def __init__(self, 
                 window_size: int = 200,
                 check_interval: int = 20,
                 patience: int = 10,
                 p_value_threshold: float = 0.05,
                 metric_type: str = 'loss'):
        """
        初始化收敛检测器
        
        Args:
            window_size: 滑动窗口大小，用于计算历史窗口和近期窗口
            check_interval: 检查间隔，每隔多少个epoch检查一次
            patience: 耐心值，需要连续多少次检查都"不显著"才确认收敛
            p_value_threshold: 统计显著性水平（p值阈值）
            metric_type: 指标类型，'loss' 或 'accuracy'
        """
        self.window_size = window_size
        self.check_interval = check_interval
        self.patience = patience
        self.p_value_threshold = p_value_threshold
        self.metric_type = metric_type
        
        # 数据存储结构
        self.epochs = []          # 存储epoch编号
        self.losses = []          # 存储loss值
        self.accuracies = []      # 存储accuracy值
        self.convergence_epoch = -1  # 收敛epoch，-1表示未收敛
        
        # 检测状态
        self.patience_counter = 0
        self.is_converged = False
        
    def add_metric(self, epoch: int, loss: float, accuracy: float) -> None:
        """
        添加新的指标数据
        
        Args:
            epoch: 当前epoch
            loss: 验证集loss
            accuracy: 验证集accuracy
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
    def get_metrics(self) -> Tuple[List[int], List[float], List[float]]:
        """
        获取所有指标数据
        
        Returns:
            epochs, losses, accuracies
        """
        return self.epochs, self.losses, self.accuracies
    
    def clear_data(self) -> None:
        """清空所有数据"""
        self.epochs.clear()
        self.losses.clear()
        self.accuracies.clear()
        self.convergence_epoch = -1
        self.patience_counter = 0
        self.is_converged = False
    
    def detect_convergence(self, current_epoch: Optional[int] = None) -> Tuple[bool, int, float]:
        """
        检测当前是否收敛
        
        Args:
            current_epoch: 当前epoch，如果为None则使用最新添加的epoch
            
        Returns:
            is_converged: 是否收敛
            convergence_epoch: 收敛epoch
            p_value: 最新的p值
        """
        if len(self.losses) < 2 * self.window_size:
            return False, -1, 1.0
        
        # 确定当前epoch
        if current_epoch is None:
            current_epoch = self.epochs[-1] if self.epochs else 0
        
        # 检查是否需要进行分析
        if current_epoch % self.check_interval != 0:
            return self.is_converged, self.convergence_epoch, 1.0 # 不需要分析时直接返回当前的默认参数值
        
        # 获取当前数据索引
        current_idx = len(self.losses) - 1
        
        # 确保有足够的数据
        if current_idx < 2 * self.window_size:
            return False, -1, 1.0
        
        # 选择分析指标
        if self.metric_type == 'accuracy':
            # 对于accuracy，我们希望它增加，所以检验"近期均值 > 历史均值"
            metric_data = self.accuracies
            alternative = 'greater'
        else:
            # 对于loss，我们希望它减少，所以检验"近期均值 < 历史均值"
            metric_data = self.losses
            alternative = 'less'
        
        # 获取历史窗口和近期窗口
        recent_window = metric_data[current_idx - self.window_size + 1: current_idx + 1]
        historical_window = metric_data[current_idx - 2 * self.window_size + 1: current_idx - self.window_size + 1]
        
        # 执行t检验
        t_stat, p_value = ttest_ind(recent_window, historical_window, 
                                   equal_var=False, alternative=alternative)
        
        # 判断是否显著
        if p_value > self.p_value_threshold:
            # 不显著，可能已收敛
            self.patience_counter += 1
            print(f"Epoch {current_epoch}: 指标变化不显著 (p={p_value:.4f})。耐心计数器: {self.patience_counter}/{self.patience}")
        else:
            # 显著，仍在学习
            self.patience_counter = 0
            print(f"Epoch {current_epoch}: 指标仍在显著变化 (p={p_value:.4f})。重置耐心计数器")
        
        # 检查是否达到收敛条件
        if self.patience_counter >= self.patience and not self.is_converged:
            self.is_converged = True
            self.convergence_epoch = current_epoch - (self.patience_counter * self.check_interval)
            print(f"\n{'='*60}")
            print(f"✅ 检测到收敛！在 Epoch {self.convergence_epoch} 附近")
            print(f"   连续 {self.patience} 次检查中，{self.metric_type}变化均不具备统计显著性 (p > {self.p_value_threshold})")
            print(f"{'='*60}")
        
        return self.is_converged, self.convergence_epoch, p_value
    
    def save_metrics_to_csv(self, filepath: str) -> None:
        """
        保存指标数据到CSV文件
        
        Args:
            filepath: 文件路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'loss', 'accuracy'])
            for epoch, loss, acc in zip(self.epochs, self.losses, self.accuracies):
                writer.writerow([epoch, loss, acc])
        print(f"指标数据已保存到: {filepath}")
    
    def load_metrics_from_csv(self, filepath: str) -> None:
        """
        从CSV文件加载指标数据
        
        Args:
            filepath: 文件路径
        """
        self.clear_data()
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                epoch, loss, acc = int(row[0]), float(row[1]), float(row[2])
                self.add_metric(epoch, loss, acc)
        print(f"从 {filepath} 加载了 {len(self.epochs)} 条指标数据")
    
    def get_convergence_summary(self) -> dict:
        """
        获取收敛检测摘要
        
        Returns:
            包含收敛信息的字典
        """
        return {
            'is_converged': self.is_converged,
            'convergence_epoch': self.convergence_epoch,
            'total_epochs': len(self.epochs),
            'patience_counter': self.patience_counter,
            'metric_type': self.metric_type,
            'window_size': self.window_size,
            'check_interval': self.check_interval,
            'patience': self.patience,
            'p_value_threshold': self.p_value_threshold
        }


# 使用示例函数
def example_usage():
    """
    使用示例
    """
    # 创建检测器
    detector = FLConvergenceDetector(
        window_size=200,
        check_interval=20,
        patience=10,
        p_value_threshold=0.05,
        metric_type='loss'
    )
    
    # 模拟添加数据
    for epoch in range(1000):
        # 模拟loss下降趋势
        loss = 2.0 * np.exp(-epoch / 300) + 0.1 + np.random.normal(0, 0.01)
        accuracy = 0.9 * (1 - np.exp(-epoch / 300)) + 0.05 + np.random.normal(0, 0.005)
        
        detector.add_metric(epoch, loss, accuracy)
        
        # 检测收敛
        is_converged, conv_epoch, p_value = detector.detect_convergence(epoch)
        
        if is_converged:
            print(f"在第 {epoch} 个epoch检测到收敛")
            break
    
    # 获取摘要
    summary = detector.get_convergence_summary()
    print("收敛检测摘要:", summary)
    
    # 保存数据
    detector.save_metrics_to_csv("convergence_analysis.csv")


if __name__ == "__main__":
    example_usage()
