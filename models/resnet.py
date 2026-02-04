
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10, name=None, created_time=None, in_channels=3):  #添加一个参数in_channels
        super(ResNet, self).__init__()
        self.in_planes = 32

        # 修改第一个卷积层以支持动态输入通道数
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def switch_grads(self, enable=True):
        for i, p in self.named_parameters():
            p.requires_grad_(enable)
    # 内部方法，构建残差块
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out5 = out5.view(out5.size()[0], -1)
        return out5

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out) # 这里与Chameleon略有区别
        out = out.view(out.size(0), -1)
        # out = torch.flatten(out, 1)  #为了与Chameleon保持一致，使用flatten实现展平操作
        out = self.linear(out)  # 得到指定的分类输出维度
        return out

    def forward_embedding(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward_embedding_tsne(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def first_activations(self, x):  # 获取第一个卷积层的输出
        x = self.relu(self.bn1(self.conv1(x)))
        return x

#  =======================实现对比学习模型================================
class SupConResNet_backbone(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10, name=None, created_time=None, in_channels=3):
        super(SupConResNet_backbone, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        #self.head = nn.Sequential(nn.Linear(256, 256),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(256, 128)
        #                       )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #logger.info(f'after layer4 data is {out}')
        out = self.avgpool(out)
        #logger.info(f'after avgpool data is {out}')
        out = torch.flatten(out, 1)    #==JLY：将数据展平==
        #logger.info(f'after flatten data is {out}')
        out = F.normalize(out, dim=1)  #==JLY：归一化输出结果==
        #logger.info(f'after normalize and out data is {out}')
        return out

def SupConResNet18(name=None, created_time=None, num_classes=10, in_channels=3):
    return SupConResNet_backbone(BasicBlock, [2,2,2,2],name='{0}_SupConResNet_18'.format(name), created_time=created_time, in_channels=in_channels)

def SupConResNet34(name=None, created_time=None, num_classes=10, in_channels=3):
    return SupConResNet_backbone(BasicBlock, [3,4,6,3],name='{0}_SupConResNet_34'.format(name), created_time=created_time, in_channels=in_channels)

def SupConResNet50(name=None, created_time=None, num_classes=10, in_channels=3):
    return SupConResNet_backbone(Bottleneck, [3,4,6,3],name='{0}_SupConResNet_50'.format(name), created_time=created_time, in_channels=in_channels)
    
def SupConResNet101(name=None, created_time=None, num_classes=10, in_channels=3):
    return SupConResNet_backbone(Bottleneck, [3,4,23,3],name='{0}_SupConResNet_101'.format(name), created_time=created_time, in_channels=in_channels)
#====================================================================================================
def ResNet18(name=None, created_time=None, num_classes=10, in_channels=3): #添加一个参数in_channels
    return ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(name), created_time=created_time, num_classes=num_classes, in_channels=in_channels) 

def ResNet34(name=None, created_time=None, num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [3,4,6,3],name='{0}_ResNet_34'.format(name), created_time=created_time, num_classes=num_classes, in_channels=in_channels)

def ResNet50(name=None, created_time=None, num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3,4,6,3],name='{0}_ResNet_50'.format(name), created_time=created_time, num_classes=num_classes, in_channels=in_channels)

def ResNet101(name=None, created_time=None, num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3,4,23,3],name='{0}_ResNet'.format(name), created_time=created_time, num_classes=num_classes, in_channels=in_channels)

def ResNet152(name=None, created_time=None, num_classes=10, in_channels=3):
    return ResNet(Bottleneck, [3,8,36,3],name='{0}_ResNet'.format(name), created_time=created_time, num_classes=num_classes, in_channels=in_channels)



def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

def layer2module(model, layer: str):
    if isinstance(model, ResNet):
        if layer == 'conv1.weight':
            return 'relu'
        elif 'conv1.weight' in layer:
            return layer.replace('conv1.weight', 'relu')
        elif 'conv2.weight' in layer:
            return layer.replace('.conv2.weight', '')
    elif isinstance(model, SimpleNet):
        module_name = None
        if 'conv' in layer:
            module_name = layer.split('.')[0]
        elif 'fc' in layer:
            module_name = layer.split('.')[0]
        return module_name