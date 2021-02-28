'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.normalize import Normalize
import math
from torch.autograd import Variable

import torch
from torch.autograd import Variable
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, pool_len=4, low_dim=128, width=1, n_label=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.base = int(64 * width)
        self.layer1 = self._make_layer(block, self.base, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.base * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, num_blocks[3], stride=2)

        print("ResNet MOE with Linear classifier ")
        self.linear_K = nn.ModuleList(
            [
                nn.Linear(self.base * 8 * block.expansion, low_dim)
                for  i in range(n_label)
            ]
        )

        self.n_label = n_label
        self.classifier = nn.Linear(self.base * 8 * block.expansion, low_dim)


        self.softmax = nn.Softmax(dim=1)

        self.l2norm = Normalize(2)
        self.pool_len = pool_len

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, gating=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.pool_len)
        x = out.view(out.size(0), -1)

        # if layer == 6:
        #     return x

        gating_logits = self.classifier(x)
        gating_logits = self.l2norm(gating_logits)

        # x = self.mlp(x)
        # x = self.l2norm(x)

        x_list = [self.l2norm( self.linear_K[i](x) ) for i in range(self.n_label)]
        if not gating:
            return torch.stack(x_list, dim=1)

        return torch.stack(x_list, dim=1), gating_logits


def ResNet18(pool_len = 4, low_dim=128):
    return ResNet(BasicBlock, [2,2,2,2], pool_len, low_dim)

def ResNet34(pool_len = 4, low_dim=128, n_label=10):
    return ResNet(BasicBlock, [3,4,6,3], pool_len, low_dim, width=1, n_label=n_label)

def ResNet50(pool_len = 4, low_dim=128, **kwargs):
    return ResNet(Bottleneck, [3,4,6,3], pool_len, low_dim, **kwargs)

def ResNet101(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,4,23,3], pool_len, low_dim)

def ResNet152(pool_len = 4, low_dim=128):
    return ResNet(Bottleneck, [3,8,36,3], pool_len, low_dim)


class ResNet34_cifar(nn.Module):
    """Encoder for MiCE"""
    def __init__(self, width=1, low_dim=128, n_label=10):
        super(ResNet34_cifar, self).__init__()
        self.encoder = ResNet34(low_dim=low_dim, n_label=n_label)
        self.encoder = nn.DataParallel(self.encoder)

    def forward(self, x, gating=False):
        return self.encoder(x, gating=gating)

