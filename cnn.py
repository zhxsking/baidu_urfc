# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels


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

class mResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        mdl = models.resnext101_32x8d(pretrained=pretrained)
        
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
        
        
#        x = F.log_softmax(x, dim=1)
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

class mSENet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
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
        
        x = torch.cat((x_img, x_visit), dim=1)
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out_fea = out
        out = out.view(features.size(0), -1)
        out = self.fc(out)
 
        return out, out_fea

class mPNASNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
        else:
            mdl= pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained=None)
        
        self.features = list(mdl.children())[:-3]
        self.features.append(nn.Dropout(p=0.5))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*self.features)
        
        self.features[0].conv = nn.Conv2d(4, 96, kernel_size=3, stride=2, bias=False)
        
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
        
        x = torch.cat((x_img, x_visit), dim=1)
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out_fea = out
        out = out.view(features.size(0), -1)
        out = self.fc(out)
 
        return out, out_fea


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
    net = mPNASNet(pretrained=False).to(device)
    
    from torchsummary import summary
    summary(net, [(img_depth, img_height, img_width), (visit_depth, visit_height, visit_width)])
    
    bs = 1
    test_x1 = torch.rand(bs, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(bs, visit_depth, visit_height, visit_width).to(device)

    out_x, out_fea = net(test_x1, test_x2)
    print(out_x)
    print(out_fea.shape)
