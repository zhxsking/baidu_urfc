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
