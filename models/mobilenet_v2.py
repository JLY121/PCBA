import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.simple import SimpleNet


__all__ = [
    "MobileNetV2",
    "SupConMobileNetV2_backbone",
    "mobilenet_V2",
    "SupConMobileNetV2",
]


def _make_divisible(v: float, divisor: int, min_value: int = None) -> int:
    """
    与原始 MobileNetV2 中相同的通道数调整函数：
    将通道数调整为 divisor 的倍数，同时避免缩减过多。
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保新通道数不会比原值少太多
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """标准卷积块：Conv2d + BatchNorm2d + ReLU6"""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        bn_eps=1e-5,
        bn_momentum=0.1,
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            # 与 torchvision 默认 BatchNorm2d 行为对齐：eps=1e-5, momentum=0.1
            nn.BatchNorm2d(out_planes, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    倒残差结构 + Linear Bottleneck 实现：
      - 若 expand_ratio != 1: 1x1 卷积升维
      - 3x3 depth-wise 卷积（groups = 通道数）
      - 1x1 线性卷积降维（无激活）
      - stride=1 且输入输出通道相同则使用残差连接
    """

    def __init__(self, inp, oup, stride, expand_ratio, bn_eps=1e-5, bn_momentum=0.1):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], "stride must be 1 or 2"
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 1x1 卷积升维
            layers.append(
                ConvBNReLU(
                    inp,
                    hidden_dim,
                    kernel_size=1,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                )
            )
        # 3x3 depth-wise 卷积
        layers.append(
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
        )
        # 1x1 线性瓶颈（无激活）
        layers.append(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        )
        layers.append(nn.BatchNorm2d(oup, eps=bn_eps, momentum=bn_momentum))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(SimpleNet):
    """
    MobileNetV2 分类模型（不加载预训练权重）：
      - 结构参考 `mobilenetv2.py` 中的实现
      - 参数初始化方式参考 `vgg_modified.py` 中的 VGG 实现
    """

    def __init__(
        self,
        num_classes=10,
        width_mult=1.0,
        in_channels=3,
        round_nearest=8,
        dropout=0.2,
        bn_eps=1e-5,
        bn_momentum=0.1,
        stem_stride=1,
        name=None,
        created_time=None,
    ):
        super(MobileNetV2, self).__init__(name)

        # 配置：每个元素为 [t, c, n, s]
        inverted_residual_setting = [
            # t,  c,  n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        last_channel = 1280

        # 构建第一层卷积
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        # CIFAR10/32x32 下通常使用 stride=1 的 stem（否则下采样太早）
        features = [
            ConvBNReLU(
                in_channels,
                input_channel,
                stride=stem_stride,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
        ]

        # 堆叠倒残差模块
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        inp=input_channel,
                        oup=output_channel,
                        stride=stride,
                        expand_ratio=t,
                        bn_eps=bn_eps,
                        bn_momentum=bn_momentum,
                    )
                )
                input_channel = output_channel

        # 最后的 1x1 卷积
        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                stride=1,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
        )

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.last_channel, num_classes)

        # 参数初始化：与 VGG 相同方式初始化卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SupConMobileNetV2_backbone(SimpleNet):
    """
    MobileNetV2 编码器，用于对比学习：
      - 仅包含卷积特征提取部分和全局平均池化
      - 输出为 L2 归一化的特征向量
    """

    def __init__(
        self,
        num_classes=10,
        width_mult=1.0,
        in_channels=3,
        round_nearest=8,
        bn_eps=1e-5,
        bn_momentum=0.1,
        stem_stride=1,
        name=None,
        created_time=None,
    ):
        super(SupConMobileNetV2_backbone, self).__init__(name)

        inverted_residual_setting = [
            # t,  c,  n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        last_channel = 1280

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        features = [
            ConvBNReLU(
                in_channels,
                input_channel,
                stride=stem_stride,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        inp=input_channel,
                        oup=output_channel,
                        stride=stride,
                        expand_ratio=t,
                        bn_eps=bn_eps,
                        bn_momentum=bn_momentum,
                    )
                )
                input_channel = output_channel

        features.append(
            ConvBNReLU(
                input_channel,
                self.last_channel,
                kernel_size=1,
                stride=1,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
            )
        )

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 与 VGG 相同方式初始化卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=1)
        return x


def mobilenet_V2(name=None, created_time=None, num_classes=10, in_channels=3):
    """
    创建 MobileNetV2 分类模型实例。
    接口风格与 `vgg_modified.vgg19_bn` 等保持一致。
    """
    model_name = "{0}_MobileNetV2".format(name) if name is not None else "MobileNetV2"
    return MobileNetV2(
        num_classes=num_classes,
        in_channels=in_channels,
        # 与 torchvision 的 MobileNetV2 对齐：BN eps=1e-5；CIFAR 常用 stem_stride=1
        bn_eps=1e-5,
        bn_momentum=0.1,
        stem_stride=1,
        name=model_name,
        created_time=created_time,
    )


def SupConMobileNetV2(name=None, created_time=None, num_classes=10, in_channels=3):
    """
    创建用于对比学习的 MobileNetV2 编码器，仅输出归一化特征向量。
    """
    model_name = (
        "{0}_SupConMobileNetV2".format(name)
        if name is not None
        else "SupConMobileNetV2"
    )
    return SupConMobileNetV2_backbone(
        num_classes=num_classes,
        in_channels=in_channels,
        bn_eps=1e-5,
        bn_momentum=0.1,
        stem_stride=1,
        name=model_name,
        created_time=created_time,
    )


