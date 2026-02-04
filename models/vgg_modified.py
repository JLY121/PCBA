'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.simple import SimpleNet


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn(num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn(num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes)


# ======================= SupCon VGG19 backbone for contrastive learning =======================
class SupConVGG19_backbone(SimpleNet):
    """
    VGG19 编码器，用于对比学习：
    - 仅包含卷积特征提取部分（features），不包含全连接分类头
    - 输出为展平后的特征，并在通道维度上做 L2 归一化，适合作为对比学习特征
    """
    def __init__(self, num_classes=10, name=None, created_time=None, in_channels=3):
        super(SupConVGG19_backbone, self).__init__()
        if in_channels != 3:
            # 当前 VGG 特征提取结构写死为 3 通道输入，其他通道数暂不支持
            logger.warning(f"SupConVGG19_backbone currently assumes in_channels=3, got {in_channels}.")
        # 复用与 vgg19_bn 相同的卷积特征结构（配置 'E' + BatchNorm）
        self.features = make_layers(cfg['E'], batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, dim=1)
        return x


def SupConVGG19_bn(name=None, created_time=None, num_classes=10, in_channels=3):
    """
    创建用于对比学习的 VGG19-BN 编码器，与 vgg19_bn 共享卷积特征结构。
    仅输出归一化的特征向量，不包含分类层。
    """
    model_name = '{0}_SupConVGG19_bn'.format(name) if name is not None else 'SupConVGG19_bn'
    return SupConVGG19_backbone(num_classes=num_classes, name=model_name, created_time=created_time, in_channels=in_channels)


# ======================= SupCon VGG11 backbone for contrastive learning =======================
class SupConVGG11_backbone(SimpleNet):
    """
    VGG11-BN 编码器，用于对比学习：
    - 仅包含卷积特征提取部分（features），不包含全连接分类头
    - 输出为展平后的特征，并在通道维度上做 L2 归一化，适合作为对比学习特征
    """
    def __init__(self, num_classes=10, name=None, created_time=None, in_channels=3):
        super(SupConVGG11_backbone, self).__init__()
        if in_channels != 3:
            # 当前 VGG 特征提取结构写死为 3 通道输入，其他通道数暂不支持
            logger.warning(f"SupConVGG11_backbone currently assumes in_channels=3, got {in_channels}.")
        # 复用与 vgg11_bn 相同的卷积特征结构（配置 'A' + BatchNorm）
        self.features = make_layers(cfg['A'], batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, dim=1)
        return x


def SupConVGG11_bn(name=None, created_time=None, num_classes=10, in_channels=3):
    """
    创建用于对比学习的 VGG11-BN 编码器，与 vgg11_bn 共享卷积特征结构。
    仅输出归一化的特征向量，不包含分类层。
    """
    model_name = '{0}_SupConVGG11_bn'.format(name) if name is not None else 'SupConVGG11_bn'
    return SupConVGG11_backbone(num_classes=num_classes, name=model_name, created_time=created_time, in_channels=in_channels)
