import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet


class BasicConv2d(nn.Module):
    """基础卷积块：包含卷积层、批量归一化和ReLU激活"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionA(nn.Module):
    """Inception A模块：用于35x35特征图的混合卷积模块"""
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        # 1x1分支
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        # 5x5分支（用1x1+5x5实现）
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        # 3x3双分支（用1x1+3x3+3x3实现）
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        # 池化分支
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    """Inception B模块：用于降采样的模块（35x35 -> 17x17）"""
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        # 3x3分支（步长为2）
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=1, padding=1)  # 修改步长为1

        # 3x3双分支（步长为2）
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)  # 修改步长为1

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)  # 修改步长为1

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    """Inception C模块：用于17x17特征图的混合卷积模块"""
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        # 1x1分支
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        # 7x7分支（用1x1+1x7+7x1实现）
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        # 7x7双分支
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        # 池化分支
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    """Inception D模块：用于降采样的模块（17x17 -> 8x8）"""
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        # 3x3分支（步长为2）
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=1, padding=1)  # 修改步长为1

        # 7x7x3分支（步长为2）
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)  # 修改步长为1

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)  # 修改步长为1
        
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    """Inception E模块：用于8x8特征图的混合卷积模块"""
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        # 1x1分支
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        # 3x3分支（分解为1x3和3x1）
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # 3x3双分支（分解为1x3和3x1）
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # 池化分支
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionV3(SimpleNet):
    """Inception V3网络：继承自SimpleNet基类，用于图像分类任务，针对CIFAR-100优化"""
    def __init__(self, num_classes=100, name=None, created_time=None, in_channels=3):
        super(InceptionV3, self).__init__(name)
        
        # 初始卷积层（适配CIFAR-100的32x32输入）
        self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)  # 32x32
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)  # 32x32
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)  # 32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 32x32，修改步长为1
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)  # 32x32
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1)  # 32x32
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 16x16，保留一个降采样

        # Inception A模块（适配较小特征图）
        self.Mixed_5b = InceptionA(192, pool_features=32)  # 256
        self.Mixed_5c = InceptionA(256, pool_features=64)  # 288
        # 减少一个A模块以适应小输入

        # Inception B模块（降采样：16x16 -> 16x16，修改步长为1）
        self.Mixed_6a = InceptionB(288)  # 768

        # Inception C模块（16x16特征图，减少模块数量）
        self.Mixed_6b = InceptionC(768, channels_7x7=128)  # 768
        self.Mixed_6c = InceptionC(768, channels_7x7=160)  # 768
        # 减少C模块数量以适应小输入

        # Inception D模块（降采样：16x16 -> 8x8，修改步长为1但保留一次降采样）
        self.Mixed_7a = InceptionD(768)  # 1280

        # Inception E模块（8x8特征图，减少模块数量）
        self.Mixed_7b = InceptionE(1280)  # 2048
        # 减少一个E模块以适应小输入

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)

    def switch_grads(self, enable=True):
        """开启或关闭梯度计算"""
        for i, p in self.named_parameters():
            p.requires_grad_(enable)

    def forward(self, x):
        # 输入预处理（适配CIFAR-100的32x32输入）
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        # Inception A模块
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        # Inception B模块
        x = self.Mixed_6a(x)

        # Inception C模块
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)

        # Inception D模块
        x = self.Mixed_7a(x)

        # Inception E模块
        x = self.Mixed_7b(x)

        # 全局平均池化和分类
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def features(self, x):
        """提取特征表示（不包含最后的分类层）"""
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

#  =======================实现对比学习模型================================
class SupConInceptionV3_backbone(SimpleNet):
    """Inception V3 Encoder模型：用于对比学习的特征提取器，不包含分类层"""
    def __init__(self, num_classes=100, name=None, created_time=None, in_channels=3):
        super(SupConInceptionV3_backbone, self).__init__(name)
        
        # 初始卷积层（适配CIFAR-100的32x32输入）
        self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)  # 32x32
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)  # 32x32
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)  # 32x32
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 32x32，修改步长为1
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)  # 32x32
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1)  # 32x32
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 16x16，保留一个降采样

        # Inception A模块（适配较小特征图）
        self.Mixed_5b = InceptionA(192, pool_features=32)  # 256
        self.Mixed_5c = InceptionA(256, pool_features=64)  # 288
        # 减少一个A模块以适应小输入

        # Inception B模块（降采样：16x16 -> 16x16，修改步长为1）
        self.Mixed_6a = InceptionB(288)  # 768

        # Inception C模块（16x16特征图，减少模块数量）
        self.Mixed_6b = InceptionC(768, channels_7x7=128)  # 768
        self.Mixed_6c = InceptionC(768, channels_7x7=160)  # 768
        # 减少C模块数量以适应小输入

        # Inception D模块（降采样：16x16 -> 8x8，修改步长为1但保留一次降采样）
        self.Mixed_7a = InceptionD(768)  # 1280

        # Inception E模块（8x8特征图，减少模块数量）
        self.Mixed_7b = InceptionE(1280)  # 2048
        # 减少一个E模块以适应小输入

        # 全局平均池化（不包含分类层和dropout）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 特征提取过程（与完整模型相同）
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        # Inception A模块
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        # Inception B模块
        x = self.Mixed_6a(x)

        # Inception C模块
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)

        # Inception D模块
        x = self.Mixed_7a(x)

        # Inception E模块
        x = self.Mixed_7b(x)

        # 全局平均池化和特征处理
        x = self.avgpool(x)
        x = torch.flatten(x, 1)    # ==JLY：将数据展平==
        x = F.normalize(x, dim=1)  # ==JLY：归一化输出结果==
        
        return x

def InceptionV3Net(name=None, created_time=None, num_classes=100, in_channels=3):
    """创建Inception V3模型实例，针对CIFAR-100优化"""
    return InceptionV3(name='{0}_InceptionV3'.format(name) if name else 'InceptionV3', created_time=created_time, num_classes=num_classes, in_channels=in_channels)

def SupConInceptionV3(name=None, created_time=None, num_classes=100, in_channels=3):
    """创建用于对比学习的Inception V3 Encoder模型实例，针对CIFAR-100优化"""
    return SupConInceptionV3_backbone(name='{0}_SupConInceptionV3'.format(name) if name else 'SupConInceptionV3', created_time=created_time, num_classes=num_classes, in_channels=in_channels)
#====================================================================================================
# def test():
#     """测试函数"""
#     net = InceptionV3Net()
#     # 注意：Inception V3通常期望299x299的输入，但这里用较小的输入进行测试
#     y = net(torch.randn(1, 3, 32, 32))  # 适配CIFAR-10等小图像数据集
#     print(f"输出形状: {y.size()}")
