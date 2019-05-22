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
                nn.Conv2d(in_depth1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_depth2, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(256+128, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                )
#        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512*3*3, 1024)
        self.fc2 = nn.Linear(1024, 9)
        
    def forward(self, x_img, x_visit):
        x_img = self.conv1(x_img)
        x_visit = self.conv2(x_visit)
        
        x = torch.cat((x_img, x_visit), dim=1)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    
class CNN11(nn.Module):
    def __init__(self, in_depth, in_size, pretrained=True):
        """VGG11迁移学习"""
        super().__init__()

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, 3, padding=1),
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512*in_size*in_size, 4096),
            self.relu,
            nn.Linear(4096, 4),
            self.relu,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class CNN11_bn(nn.Module):
    def __init__(self, in_depth, in_size, pretrained=True):
        """VGG11迁移学习"""
        super().__init__()

        self.encoder = models.vgg11_bn(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, 3, padding=1),
                                   self.encoder[1],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[4],
                                   self.encoder[5],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[8],
            self.encoder[9],
            self.relu,
            self.encoder[11],
            self.encoder[12],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[15],
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.encoder[19],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[22],
            self.encoder[23],
            self.relu,
            self.encoder[25],
            self.encoder[26],
            self.relu,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512*in_size*in_size, 4096),
            self.relu,
            nn.Linear(4096, 4),
            self.relu,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
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
#    summary(net, (img_depth, img_height, img_width), (visit_depth, visit_height, visit_width))
    
    test_x1 = torch.rand(1, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(1, visit_depth, visit_height, visit_width).to(device)

    out_x = net(test_x1, test_x2)
    print(out_x)
    