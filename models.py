# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels


class SeResNext50(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, pretrained=False):
        super().__init__()
        if pretrained:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d']()
        else:
            mdl= pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)
        
        self.model = list(mdl.children())[:-2]
        self.model.append(nn.AdaptiveAvgPool2d(1))
        self.model = nn.Sequential(*self.model)
        self.model[0].conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, 
                                        padding=1, bias=False)
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=mdl.last_linear.in_features, out_features=out_channels, bias=True),
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
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DPN68(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['dpn68']()
        else:
            mdl= pretrainedmodels.__dict__['dpn68'](pretrained=None)

        self.model = mdl
        self.model.test_time_pool = False
        self.model.features.conv1_1 = nn.Sequential(
                nn.Conv2d(in_channels, 10, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(10),
                nn.ReLU(inplace=True),
                )
        self.model.last_linear = nn.Conv2d(mdl.last_linear.in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.model(x)
        return x

class DPN92(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
#        print(pretrainedmodels.model_names)
        if pretrained:
            mdl= pretrainedmodels.__dict__['dpn92']()
        else:
            mdl= pretrainedmodels.__dict__['dpn92'](pretrained=None)

        self.model = mdl
        self.model.test_time_pool = False
        self.model.features.conv1_1 = nn.Sequential(
                nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                )
        self.model.last_linear = nn.Conv2d(mdl.last_linear.in_channels, 9, kernel_size=(1, 1), stride=(1, 1))

        if not(pretrained):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.model(x)
        return x


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
    bs = 1
    test_x1 = torch.rand(bs, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(bs, visit_depth, visit_height, visit_width).to(device)

    net1 = SeResNext50(3, 128).to(device)
    
    from torchsummary import summary
    summary(net1, (img_depth, img_height, img_width))
    
    out = net1(test_x1)
    print(out.shape)
    
    net2 = SeResNext50(7, 64).to(device)
    
#    from torchsummary import summary
#    summary(net2, (visit_depth, visit_height, visit_width))
    
    out = net2(test_x2)
    print(out.shape)


