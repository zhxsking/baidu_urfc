# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels

from multimodal import DPN26, DPN92, MultiModalNet, FCViewer
from unet import UNet, UNet_p


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

class CNN(nn.Module):
    def __init__(self, in_depth1=3, in_depth2=7):
        """参数分别为输入图像深度、高、宽"""
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_depth1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_depth2, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 9)
        
    def forward(self, x_img, x_visit):
#        x_img = self.conv1(x_img)
        x_visit = self.conv2(x_visit)
        
#        x = torch.cat((x_img, x_visit), dim=1)
        x = self.conv3(x_visit)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
#        x = F.log_softmax(x, dim=1)
        return x

class mTESTNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained=None)
        
#        self.features = list(mdl.children())[:-3]
##        self.features.append(nn.AdaptiveAvgPool2d(1))
##        self.features.append(nn.Dropout(p=0.5))
#        self.features = nn.Sequential(*self.features)
        
        self.features = mdl
        self.features.conv0.conv = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.features.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.features.last_linear = nn.Linear(in_features=4032, out_features=64, bias=True)
        
#        self.fc = nn.Linear(mdl.last_linear.in_features, 9)
        

        
    def forward(self, x_img, x_visit):
        # N,7,26,24整型为N,1,56,78
#        x_visit = x_visit.reshape(x_visit.size(0), 1, 56, -1)
#        # pad为N,1,100,100
#        x_visit = nn.ConstantPad2d((11,11,22,22), 0)(x_visit)
#        
#        x = torch.cat((x_img, x_visit), dim=1)
        
        features = self.features(x_img)
        return features
#        out = F.relu(features, inplace=True)
#        out = F.adaptive_avg_pool2d(out, (1, 1))
#        out_fea = out
#        out = out.view(features.size(0), -1)
#        out = self.fc(out)
# 
#        return out, out_fea

class mResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        mdl = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = mdl.bn1
        self.relu = mdl.relu
        self.maxpool = mdl.maxpool
        
        self.layer1 = mdl.layer1
        self.layer2 = mdl.layer2
        self.layer3 = mdl.layer3
        self.layer4 = mdl.layer4
        
        self.avgpool = mdl.avgpool
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(mdl.fc.in_features, 9)
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_visit):
        # N,7,26,24整型为N,1,56,78
#        x_visit = x_visit.reshape(x_visit.size(0), 1, 56, -1)
#        # pad为N,1,100,100
#        x_visit = nn.ConstantPad2d((11,11,22,22), 0)(x_visit)
#        
#        x = torch.cat((x_img, x_visit), dim=1)
        
        x = self.conv1(x_img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_fea = x
        
        x = x.reshape(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x, x_fea

class mResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        mdl = models.resnext101_32x8d(pretrained=pretrained)
        mdl = models.resnext50_32x4d(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = mdl.bn1
        self.relu = mdl.relu
        self.maxpool = mdl.maxpool
        
        self.layer1 = mdl.layer1
        self.layer2 = mdl.layer2
        self.layer3 = mdl.layer3
        self.layer4 = mdl.layer4
        
        self.avgpool = mdl.avgpool
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(mdl.fc.in_features, 9)
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_visit):
        # N,7,26,24整型为N,1,56,78
        x_visit = x_visit.permute(0,2,1,3)
        x_visit = x_visit.reshape(x_visit.size(0), 1, 56, -1)
        # pad为N,1,100,100
        x_visit = nn.ConstantPad2d((11,11,22,22), 0)(x_visit)
        
        x = torch.cat((x_img, x_visit), dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_fea = x
        
        x = x.reshape(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x, x_fea

class mDenseNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        mdl = models.densenet121(pretrained=pretrained)
        
#        self.features_all = list(mdl.children())
        self.features = mdl.features
        self.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(mdl.classifier.in_features, 9)
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_visit):
        # N,7,26,24整型为N,1,56,78
        x_visit = x_visit.reshape(x_visit.size(0), 1, 56, -1)
        # pad为N,1,100,100
        x_visit = nn.ConstantPad2d((11,11,22,22), 0)(x_visit)
        
        x = torch.cat((x_img, x_visit), dim=1)
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out_fea = out
        out = out.view(features.size(0), -1)
        out = self.fc(out)
 
        return out, out_fea
    
class mDPN68Net(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['dpn68'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['dpn68'](num_classes=1000, pretrained=None)

#        self.features = list(mdl.children())
        self.img_model = mdl
        self.img_model.test_time_pool = False
        self.img_model.features.conv1_1 = nn.Sequential(
                nn.Conv2d(7, 10, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                )
        self.img_model.last_linear = nn.Conv2d(mdl.last_linear.in_channels, 9, kernel_size=(1, 1), stride=(1, 1))

        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
#        x_vis = self.vis_model(x_vis)
#        x_img = nn.ConstantPad2d((1,0,1,0), 0)(x_img) # pad为101*101
        
#        x_vis = x_vis.reshape(x_vis.size(0), 1, 56, -1)
#        # pad为N,1,100,100
#        x_vis = nn.ConstantPad2d((11,11,22,22), 0)(x_vis)

        x_vis = self.img_model(x_vis)
        
#        x = torch.cat((x_img, x_vis), dim=1)
#        out_fea = x
#        out = self.fc(x)
        return x_vis, x_vis

class mDPN92Net(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['dpn92']()
        else:
            mdl= pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained=None)

#        self.features = list(mdl.children())
        self.img_model = mdl
        self.img_model.test_time_pool = False
        self.img_model.features.conv1_1 = nn.Sequential(
                nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                )
        self.img_model.last_linear = nn.Conv2d(mdl.last_linear.in_channels, 9, kernel_size=(1, 1), stride=(1, 1))

        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
#        x_vis = self.vis_model(x_vis)
#        x_img = nn.ConstantPad2d((1,0,1,0), 0)(x_img) # pad为101*101
        
#        x_vis = x_vis.reshape(x_vis.size(0), 1, 56, -1)
#        # pad为N,1,100,100
#        x_vis = nn.ConstantPad2d((11,11,22,22), 0)(x_vis)

        x_vis = self.img_model(x_vis)
        
#        x = torch.cat((x_img, x_vis), dim=1)
#        out_fea = x
#        out = self.fc(x)
        return x_vis, x_vis

class mSENet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
        
        self.features = list(mdl.children())[:-2]
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        
        self.features[0].conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        
        self.fc = nn.Linear(mdl.last_linear.in_features, 9)
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_img, x_visit):
        # N,7,26,24整型为N,1,56,78
        x_visit = x_visit.reshape(x_visit.size(0), 1, 56, -1)
        # pad为N,1,100,100
        x_visit = nn.ConstantPad2d((11,11,22,22), 0)(x_visit)
#        x_visit = nn.ReflectionPad2d((11,11,22,22))(x_visit)
        
        x = torch.cat((x_img, x_visit), dim=1)
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out_fea = out
        out = out.view(features.size(0), -1)
        out = self.fc(out)
 
        return out, out_fea

class mUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        
        self.features = UNet(in_depth=4)
        
        if (pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_img, x_visit):
        # N,7,26,24整型为N,1,56,78
        x_visit = x_visit.permute(0,2,1,3) # 26,7,24
        x_visit = x_visit.reshape(x_visit.size(0), 1, 56, -1)
        # pad为N,1,100,100
        x_visit = nn.ConstantPad2d((11,11,22,22), 0)(x_visit)
#        x_visit = nn.ReflectionPad2d((11,11,22,22))(x_visit)
        
        x = torch.cat((x_img, x_visit), dim=1)
        x = nn.ConstantPad2d((6,6,6,6), 0)(x) # 100*100 -> 112*112
        
        out = self.features(x)
 
        return out, out

class mSS_UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        
        # 3*100*100 -> 32*32*32
        self.img_conv = nn.Sequential(
                conv3x3(3, 64, stride=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                Bottleneck(64, 64, stride=2, dilation=1, expansion=2, need_downsample=True),
                Bottleneck(64*2, 32, stride=2, padding=8, expansion=1, need_downsample=True),
                )
        
        # 7*26*24 -> 32*32*32
        self.vis_conv = nn.Sequential(
                nn.Conv2d(7, 32, kernel_size=3, padding=(4,5)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        
        self.features = UNet(in_depth=64)
        
        if (pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_img, x_vis):
        x_img = self.img_conv(x_img)
        x_vis = self.vis_conv(x_vis)
        x = torch.cat((x_img, x_vis), dim=1)
        
        x = self.features(x)
        
        return x, x

class mSS_UNet_p(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        
        # 3*100*100 -> 32*32*26
        self.img_conv = nn.Sequential(
                conv3x3(3, 64, stride=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                Bottleneck(64, 64, stride=2, dilation=1, expansion=2, need_downsample=True),
                Bottleneck(64*2, 32, stride=2, padding=8, expansion=1, need_downsample=True),
                )
        
        # 7*26*24 -> 32*26*26
        self.vis_conv = nn.Sequential(
                nn.Conv2d(7, 32, kernel_size=3, padding=(4,5)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        
        self.features = UNet_p(in_depth=64)
        
        if (pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_img, x_vis):
        x_img = self.img_conv(x_img)
        x_vis = self.vis_conv(x_vis)
        x = torch.cat((x_img, x_vis), dim=1)
        
        x = self.features(x)
        
        return x, x

class mSSNet50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
        
        # 3*100*100 -> 32*26*26
        self.img_conv = nn.Sequential(
                conv3x3(3, 64, stride=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                Bottleneck(64, 64, stride=2, dilation=1, expansion=2, need_downsample=True),
                Bottleneck(64*2, 32, stride=2, padding=2, expansion=1, need_downsample=True),
                )
        
        # 7*26*24 -> 32*26*26
        self.vis_conv = nn.Sequential(
                nn.Conv2d(7, 32, kernel_size=3, padding=(1,2)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        
        self.features = list(mdl.children())[1:-2]
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=mdl.last_linear.in_features, out_features=9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x_img, x_vis):
        x_img = self.img_conv(x_img)
        x_vis = self.vis_conv(x_vis)
        x = torch.cat((x_img, x_vis), dim=1)
        
        x = self.features(x)
        
        x = x.view(x.size(0), -1)
        fea = x
        x = self.fc(x)
        
        return x, fea

class mSSNet101(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
        
        # 3*100*100 -> 32*26*26
        self.img_conv = nn.Sequential(
                conv3x3(3, 64, stride=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                Bottleneck(64, 64, stride=2, dilation=1, expansion=2, need_downsample=True),
                Bottleneck(64*2, 32, stride=2, padding=2, expansion=1, need_downsample=True),
                )
        
        # 7*26*24 -> 32*26*26
        self.vis_conv = nn.Sequential(
                nn.Conv2d(7, 32, kernel_size=3, padding=(1,2)),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        
        self.features = list(mdl.children())[1:-2]
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=mdl.last_linear.in_features, out_features=9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_img = self.img_conv(x_img)
        x_vis = self.vis_conv(x_vis)
        x = torch.cat((x_img, x_vis), dim=1)
        
        x = self.features(x)
        
        x = x.view(x.size(0), -1)
        fea = x
        x = self.fc(x)
        
        return x, fea

class mSDNet50(nn.Module):
    '''sdnet'''
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
        
        self.features = list(mdl.children())[:-2]
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        self.features[0].conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        self.fc_img = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=mdl.last_linear.in_features, out_features=64, bias=True),
                )
        
        self.visit_model=DPN26()
        self.visit_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.visit_model.linear.in_features, out_features=64, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.visit_model(x_vis)
        
        features = self.features(x_img)
        x_img = F.relu(features, inplace=True)
        x_img = F.adaptive_avg_pool2d(x_img, (1, 1))
        x_img = x_img.view(features.size(0), -1)
        x_img = self.fc_img(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        
        out_fea = x
        
        out = self.fc(x)
 
        return out, out_fea

class mSDNet50_p(nn.Module):
    '''sdnet'''
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained=None)
        
        self.features = list(mdl.children())[:-2]
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        self.features[0].conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        self.fc_img = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=mdl.last_linear.in_features, out_features=256, bias=True),
                )
        
        self.visit_model=DPN26()
        self.visit_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.visit_model.linear.in_features, out_features=128, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(384, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.visit_model(x_vis)
        
        features = self.features(x_img)
        x_img = F.relu(features, inplace=True)
        x_img = F.adaptive_avg_pool2d(x_img, (1, 1))
        x_img = x_img.view(features.size(0), -1)
        x_img = self.fc_img(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        
        out_fea = x
        
        out = self.fc(x)
 
        return out, out_fea

class mSDNet101(nn.Module):
    '''sdnet'''
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
        
        self.features = list(mdl.children())[:-2]
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        self.features[0].conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                                        padding=3, bias=False)
        self.fc_img = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=mdl.last_linear.in_features, out_features=256, bias=True),
                )
        
        self.visit_model=DPN26()
        self.visit_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.visit_model.linear.in_features, out_features=64, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(320, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.visit_model(x_vis)
        
        x_img = self.features(x_img)
#        x_img = F.relu(features, inplace=True)
#        x_img = F.adaptive_avg_pool2d(x_img, (1, 1))
        x_img = x_img.view(x_img.size(0), -1)
        x_img = self.fc_img(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        
        out_fea = x
        
        out = self.fc(x)
 
        return out, out_fea

class mXNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained=None)

#        self.features = list(mdl.children())
        self.img_model = mdl
        self.img_model.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.img_model.last_linear = nn.Linear(in_features=mdl.last_linear.in_features, out_features=64, bias=True)
        
        self.vis_model=DPN26()
        self.vis_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.vis_model.linear.in_features, out_features=64, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.vis_model(x_vis)
        x_img = nn.ConstantPad2d((1,0,1,0), 0)(x_img) # pad为101*101
        x_img = self.img_model(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        out_fea = x
        out = self.fc(x)
        return out, out_fea

class mPOLYNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['polynet'](num_classes=1000, pretrained=None)

#        self.features = list(mdl.children())
        self.img_model = mdl
        self.img_model.stem.conv1[0].conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.img_model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.img_model.last_linear = nn.Linear(in_features=mdl.last_linear.in_features, out_features=64, bias=True)
        
        self.vis_model=DPN26()
        self.vis_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.vis_model.linear.in_features, out_features=64, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.vis_model(x_vis)
        x_img = nn.ConstantPad2d((1,0,1,0), 0)(x_img) # pad为101*101
        x_img = self.img_model(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        out_fea = x
        out = self.fc(x)
        return out, out_fea

class mNASNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained=None)

#        self.features = list(mdl.children())
        self.img_model = mdl
        self.img_model.conv0.conv = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.img_model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.img_model.last_linear = nn.Linear(in_features=mdl.last_linear.in_features, out_features=64, bias=True)
        
        self.vis_model=DPN26()
        self.vis_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.vis_model.linear.in_features, out_features=64, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.vis_model(x_vis)
        x_img = nn.ConstantPad2d((1,0,1,0), 0)(x_img) # pad为101*101
        x_img = self.img_model(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        out_fea = x
        out = self.fc(x)
        return out, out_fea

class mPNASNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained=None)

#        self.features = list(mdl.children())
        self.img_model = mdl
        self.img_model.conv_0.conv = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.img_model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.img_model.last_linear = nn.Linear(in_features=mdl.last_linear.in_features, out_features=64, bias=True)
        
        self.vis_model=DPN26()
        self.vis_model.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.vis_model.linear.in_features, out_features=64, bias=True),
                )
        
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, 9, bias=True),
                )
        
        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x_img, x_vis):
        x_vis = self.vis_model(x_vis)
        x_img = nn.ConstantPad2d((1,0,1,0), 0)(x_img) # pad为101*101
        x_img = self.img_model(x_img)
        
        x = torch.cat((x_img, x_vis), dim=1)
        out_fea = x
        out = self.fc(x)
        return out, out_fea

class MMNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet') #seresnext101
        else:
            img_model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained=None)
       
        self.visit_model=DPN26()
        
        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)

        self.img_fc = nn.Sequential(FCViewer(),
                                nn.Dropout(0.5),
                                nn.Linear(img_model.last_linear.in_features, 256))

        self.cls = nn.Linear(320,9) 

    def forward(self, x_img,x_vis):
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)

        x_vis=self.visit_model(x_vis)
        x_cat = torch.cat((x_img,x_vis),1)
        out_fea = x_cat
        x_cat = self.cls(x_cat)
        return x_cat, out_fea

class mDPN26(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.visit_model=DPN26()
        self.cls = nn.Linear(64, 9) 

    def forward(self, x_img, x_vis):
        x_vis = self.visit_model(x_vis)
        x_cat = self.cls(x_vis)
        return x_cat, x_vis



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    img_depth = 3
    img_height = 100
    img_width = 100
    visit_depth = 7
    visit_height = 26
    visit_width = 24
    net = mUNet().to(device)
#    net = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(device)
    
    from torchsummary import summary
    summary(net, [(img_depth, img_height, img_width), (visit_depth, visit_height, visit_width)])
    
    bs = 32
    test_x1 = torch.rand(bs, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(bs, visit_depth, visit_height, visit_width).to(device)

    out_x, out_fea = net(test_x1, test_x2)
    print(out_x.shape)
    print(out_fea.shape)

    torch.cuda.empty_cache()
