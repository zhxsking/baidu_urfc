# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, padding=1, dilation=1, expansion=2, need_downsample=False):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.need_downsample = need_downsample
        if self.need_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * expansion, stride=stride, padding=padding-1),
                nn.BatchNorm2d(planes * expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.need_downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

