from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.simple import SimpleNet


class EfficientNetB0(SimpleNet):
    """
    适用于 CIFAR-100 数据集的 EfficientNet-B0 模型。
    基于 timm 实现，但针对 32x32 输入图像进行了调整。
    """
    
    def __init__(self, num_classes=100, name=None, created_time=None, in_channels=3, pretrained=True):
        super(EfficientNetB0, self).__init__(name=name)
        
        # 从 timm 加载EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        # 调整 stem（第一个卷积层）以适应 CIFAR-100 的 32x32 输入
        # 原始步长=2 会将 32x32 快速降采样到 16x16，丢失过多信息
        original_conv_stem = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels,
            original_conv_stem.out_channels,
            kernel_size=3,
            stride=1,  # 从 2 改为 1，适应 CIFAR-100
            padding=1,
            bias=False
        )
        
        # 如果可能，从原始 stem 复制权重（中心部分）
        if in_channels == 3:
            with torch.no_grad():
                # 复制原始 3x3 核的中心部分
                self.backbone.conv_stem.weight.data = original_conv_stem.weight.data.clone()
        
        # 替换分类器头部以适应 CIFAR-100（100 类）
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        
        # 存储参数
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # 初始化新层
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化新层的权重。"""
        # 初始化新的分类器层
        nn.init.normal_(self.backbone.classifier.weight, 0, 0.01)
        nn.init.zeros_(self.backbone.classifier.bias)
    
    def switch_grads(self, enable=True):
        """启用或禁用所有参数的梯度计算。"""
        for param in self.parameters():
            param.requires_grad_(enable)
    
    def forward(self, x):
        """模型的前向传播。"""
        return self.backbone(x)
    
    def forward_features(self, x):
        """仅通过特征提取器进行前向传播（不包含分类器）。"""
        # 使用 backbone 的特征提取器提取特征
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        return x
    
    def forward_embedding(self, x):
        """前向传播获取嵌入（分类前的特征）。"""
        features = self.forward_features(x)
        # 全局平均池化
        x = self.backbone.global_pool(features)
        return x
    
    def features(self, x):
        """提取特征并返回展平表示。"""
        embedding = self.forward_embedding(x)
        return embedding.view(embedding.size(0), -1)

# 实现对比学习模型的Encoder部分
class SupConEfficientNetB0_backbone(SimpleNet):
    """
    适用于对比学习的 EfficientNet-B0 模型的 Encoder 部分。
    基于 timm 实现，但针对 32x32 输入图像进行了调整。
    """
    
    def __init__(self, name=None, created_time=None, in_channels=3):
        super(SupConEfficientNetB0_backbone, self).__init__(name=name)
        
        # 从 timm 加载EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        # 调整 stem（第一个卷积层）以适应 CIFAR-100 的 32x32 输入
        # 原始步长=2 会将 32x32 快速降采样到 16x16，丢失过多信息
        original_conv_stem = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels,
            original_conv_stem.out_channels,
            kernel_size=3,
            stride=1,  # 从 2 改为 1，适应 CIFAR-100
            padding=1,
            bias=False
        )
        
        # 如果可能，从原始 stem 复制权重（中心部分）
        if in_channels == 3:
            with torch.no_grad():
                # 复制原始 3x3 核的中心部分
                self.backbone.conv_stem.weight.data = original_conv_stem.weight.data.clone()
        
        # 移除分类器头部，因为这是Encoder部分
        self.backbone.classifier = nn.Identity()
        
        # 存储参数
        self.in_channels = in_channels
    
    def forward(self, x):
        """
        前向传播，输出归一化的特征向量。
        """
        out = self.backbone.conv_stem(x)
        out = self.backbone.bn1(out)
        out = self.backbone.blocks(out)
        out = self.backbone.conv_head(out)
        out = self.backbone.bn2(out)
        out = self.backbone.global_pool(out)
        out = torch.flatten(out, 1)    # 将数据展平
        out = F.normalize(out, dim=1)  # 归一化输出结果
        return out
    
    def switch_grads(self, enable=True):
        """启用或禁用所有参数的梯度计算。"""
        for param in self.parameters():
            param.requires_grad_(enable)

# 遵循 ResNet 模式的工厂函数
def EfficientNetB0_cifar100(name=None, created_time=None, num_classes=100, in_channels=3, pretrained=False):
    model_name = f'{name}_EfficientNet_B0_CIFAR100' if name else 'EfficientNet_B0_CIFAR100'
    return EfficientNetB0(num_classes=num_classes, name=model_name, created_time=created_time, in_channels=in_channels, pretrained=pretrained)


def EfficientNetB0_Custom(name=None, created_time=None, num_classes=10, in_channels=3, pretrained=False):
    """
    创建具有自定义类别数的 EfficientNet-B0 模型。
    
    参数:
        name: 模型名称标识
        created_time: 创建时间戳
        num_classes: 输出类别数
        in_channels: 输入通道数
        pretrained: 是否使用预训练权重（默认为 True）
    
    返回:
        EfficientNetB0 模型实例
    """
    model_name = f'{name}_EfficientNet_B0_Custom' if name else 'EfficientNet_B0_Custom'
    return EfficientNetB0(
        num_classes=num_classes,
        name=model_name,
        created_time=created_time,
        in_channels=in_channels,
        pretrained=pretrained
    )

def SupConEfficientNetB0(name=None, created_time=None, num_classes=100, in_channels=3):
    model_name = f'{name}_SupConEfficientNet_B0' if name else 'SupConEfficientNet_B0'
    return SupConEfficientNetB0_backbone(name=model_name, created_time=created_time, in_channels=in_channels)


# # 测试函数
# def test():
#     """使用 CIFAR-100 输入尺寸测试 EfficientNet-B0 模型。"""
#     print("测试 CIFAR-100 的 EfficientNet-B0...")
    
#     # 使用 pretrained=False 避免网络问题
#     model = EfficientNetB0_CIFAR100(pretrained=False)
#     x = torch.randn(1, 3, 32, 32)  # CIFAR-100 输入尺寸
    
#     print(f"输入形状: {x.shape}")
    
#     # 测试前向传播
#     y = model(x)
#     print(f"输出形状: {y.shape}")
#     print(f"模型名称: {model.name}")
    
#     # 测试特征提取
#     features = model.features(x)
#     print(f"特征形状: {features.shape}")
    
#     # 测试嵌入提取
#     embedding = model.forward_embedding(x)
#     print(f"嵌入形状: {embedding.shape}")
    
#     # 测试不同输入尺寸
#     print("\n测试不同批次大小:")
#     for batch_size in [1, 4, 8]:
#         x_batch = torch.randn(batch_size, 3, 32, 32)
#         y_batch = model(x_batch)
#         print(f"批次大小 {batch_size}: {x_batch.shape} -> {y_batch.shape}")
    
#     print("所有测试通过!")


# def test_with_pretrained():
#     """使用预训练权重测试 EfficientNet-B0 模型（如果网络可用）。"""
#     print("使用预训练权重测试 CIFAR-100 的 EfficientNet-B0...")
    
#     try:
#         model = EfficientNetB0_CIFAR100(pretrained=True)
#         x = torch.randn(1, 3, 32, 32)
#         y = model(x)
#         print(f"成功加载预训练模型!")
#         print(f"输入形状: {x.shape}")
#         print(f"输出形状: {y.shape}")
#     except Exception as e:
#         print(f"加载预训练模型失败: {e}")
#         print("如果网络不可用，这是预期的。")


# if __name__ == '__main__':
#     test()
#     print("\n" + "="*50 + "\n")
#     test_with_pretrained()