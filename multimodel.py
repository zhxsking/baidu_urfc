# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels

from models import resnet50


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    img_depth = 3
    img_height = 100
    img_width = 100
    visit_depth = 7
    visit_height = 26
    visit_width = 24
    net = mResnet50(dilation=1).to(device)
    
    from torchsummary import summary
    summary(net, [(img_depth, img_height, img_width), (visit_depth, visit_height, visit_width)])
    
    bs = 1
    test_x1 = torch.rand(bs, img_depth, img_height, img_width).to(device)
    test_x2 = torch.rand(bs, visit_depth, visit_height, visit_width).to(device)

    out_x, out_fea = net(test_x1, test_x2)
    print(out_x.shape)
    print(out_fea.shape)
