# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
                nn.Conv2d(64+32, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 9)
        
    def forward(self, x_img, x_visit):
        x_img = self.conv1(x_img)
        x_visit = self.conv2(x_visit)
        
        x = torch.cat((x_img, x_visit), dim=1)
        x = self.conv3(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_depth = 3
    img_height = 100
    img_width = 100
    visit_depth = 7
    visit_height = 26
    visit_width = 24
    net = CNN().to(device)
    
#    from torchsummary import summary
#    summary(net, ((img_depth, img_height, img_width), (visit_depth, visit_height, visit_width)))
    
    test_x1 = torch.rand(1, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(1, visit_depth, visit_height, visit_width).to(device)

    out_x = net(test_x1, test_x2)
    print(out_x)
    