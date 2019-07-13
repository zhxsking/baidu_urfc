# -*- coding: utf-8 -*-

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, need_downsample=False):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.need_downsample = need_downsample
        if self.need_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
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


class ResNet(nn.Module):

    def __init__(self, num_block, num_classes=1000, dilation=1):
        super(ResNet, self).__init__()

        self.dilation = dilation

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, num_block[0], stride=1, dilation=dilation)
        self.layer2 = self._make_layer(4*64, 128, num_block[0], stride=2, dilation=dilation)
        self.layer3 = self._make_layer(4*128, 256, num_block[0], stride=2, dilation=dilation)
        self.layer4 = self._make_layer(4*256, 512, num_block[0], stride=2, dilation=dilation)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, inplanes, planes, num_block, stride=1, dilation=1):
        layers = []
        layers.append(Bottleneck(inplanes, planes, stride=stride, dilation=dilation, need_downsample=True))
        for i in range(num_block-1):
            layers.append(Bottleneck(planes*4, planes, dilation=dilation, need_downsample=False))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(num_classes=9, dilation=1):
    return ResNet(num_block=[3, 4, 6, 3], num_classes=num_classes, dilation=dilation)


def resnet101(num_classes=9, dilation=1):
    return ResNet(num_block=[3, 4, 23, 3], num_classes=num_classes, dilation=dilation)


def resnet152(num_classes=9, dilation=1):
    return ResNet(num_block=[3, 8, 36, 3], num_classes=num_classes, dilation=dilation)

class mResnet50(nn.Module):
    def __init__(self, dilation=1):
        super().__init__()
        self.img_model = resnet50(dilation=dilation)
        self.img_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512*4, 9),
                )

    def forward(self, x_img, x_vis):
        x_img = self.img_model(x_img)
        return x_img, x_img


if __name__ == '__main__':
    net = resnet50(dilation=1)
    
#    from torchvision import models
#    net = models.resnet50(num_classes=9)
    
    from torchsummary import summary
    summary(net, (3, 100, 100))
