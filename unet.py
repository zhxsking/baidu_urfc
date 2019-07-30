# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model_util import Bottleneck


class Down(nn.Module):
    """unet的下降部分模块"""
    def __init__(self, in_channel, out_channel, do_pool=True):
        super().__init__()
        self.conv = nn.Sequential(
                Bottleneck(in_channel, out_channel, expansion=2, need_downsample=True),
                Bottleneck(out_channel*2, int(out_channel/2), expansion=2, need_downsample=True),
#                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
#                nn.BatchNorm2d(out_channel),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#                nn.BatchNorm2d(out_channel),
#                nn.ReLU(inplace=True),
                )
        self.pool = nn.MaxPool2d(2)
        self.do_pool = do_pool
        
    def forward(self, x):
        if self.do_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x
    
class Down2(nn.Module):
    """unet的下降部分模块"""
    def __init__(self, in_channel, out_channel, do_pool=True):
        super().__init__()
        self.conv = nn.Sequential(
                Bottleneck(in_channel, out_channel, expansion=2, need_downsample=True),
                Bottleneck(out_channel*2, int(out_channel/2), expansion=2, need_downsample=True),
#                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
#                nn.BatchNorm2d(out_channel),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#                nn.BatchNorm2d(out_channel),
#                nn.ReLU(inplace=True),
                )
        self.pool = nn.MaxPool2d(2)
        self.do_pool = do_pool
        
    def forward(self, x_prev, x):
        if self.do_pool:
            x = self.pool(x)
        x = torch.cat((x_prev, x), dim=1)
        x = self.conv(x)
        return x
        
        return x    
    
class Up(nn.Module):
    """unet的上升部分模块"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
#                Bottleneck(in_channel, int(out_channel/2), expansion=2, need_downsample=True),
                Bottleneck(in_channel, out_channel, expansion=2, need_downsample=True),
                Bottleneck(out_channel*2, int(out_channel/2), expansion=2, need_downsample=True),
#                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
#                nn.BatchNorm2d(out_channel),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
#                nn.BatchNorm2d(out_channel),
#                nn.ReLU(inplace=True),
                )
        
    def forward(self, x_prev, x):
        x = self.up(x)
        x = torch.cat((x_prev, x), dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    """unet定义
    
    采用双线性上采样，实际网络与论文有所不同，
    论文中上卷积将图的深度也减半了，但上采样不改变深度
    """
    
    def __init__(self, in_depth):
        super().__init__()
#        self.conv = 
        self.down1 = Down(in_depth, 64, do_pool=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out_conv = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=9, bias=True))
        
    def forward(self, x):
#        x = nn.ConstantPad2d((6,6,6,6), 0)(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class UNet_n(nn.Module):
    """unet定义
    
    采用双线性上采样，实际网络与论文有所不同，
    论文中上卷积将图的深度也减半了，但上采样不改变深度
    """
    
    def __init__(self, in_depth):
        super().__init__()
#        self.conv = 
        self.down1 = Down(in_depth, 64, do_pool=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.down6 = Down2(64, 128)
        self.down7 = Down2(128, 256)
        self.down8 = Down2(256, 512)
        self.down9 = Down2(512, 512)
        self.out_conv = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=9, bias=True))
        
    def forward(self, x):
#        x = nn.ConstantPad2d((6,6,6,6), 0)(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x2 = self.down6(x1)
        x3 = self.down7(x2)
        x4 = self.down8(x3)
        x5 = self.down9(x4)
        x = self.out_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class UNet_p(nn.Module):
    """unet定义
    
    采用双线性上采样，实际网络与论文有所不同，
    论文中上卷积将图的深度也减半了，但上采样不改变深度
    """
    
    def __init__(self, in_depth):
        super().__init__()
#        self.conv = 
        self.down1 = Down(in_depth, 64, do_pool=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out_conv = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.AdaptiveAvgPool2d(1))
        self.out_conv_top = nn.Sequential(
                nn.Conv2d(512, 64, kernel_size=1),
                nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=128, out_features=9, bias=True))
        
    def forward(self, x):
#        x = nn.ConstantPad2d((6,6,6,6), 0)(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out_conv(x)
        x = x.view(x.size(0), -1)
        
        xx = self.out_conv_top(x5)
        xx = xx.view(xx.size(0), -1)
        
        x = torch.cat((x, xx),1)
        
        x = self.fc(x)
        return x
 

if __name__ == '__main__':
    unet = UNet_n(in_depth=3)
    if torch.cuda.is_available():
        unet = unet.cuda()
#    print(unet)

#    test_x = torch.FloatTensor(1, 1, 256, 256)
#    out_x = unet(test_x)
#    print(out_x.size())
    
    from torchsummary import summary
    summary(unet, (3,112,112))
        
        