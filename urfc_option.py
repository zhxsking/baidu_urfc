# -*- coding: utf-8 -*-

import torch

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        
        self.dir_img = r"D:\pic\URFC-baidu\train_image"
        self.dir_visit = r"D:\pic\URFC-baidu\train_visit"
        self.dir_visit_npy = self.dir_visit + "-npy-26-24-7"
        
        
        
        
        self.depth = 8 # 图片深度
        self.size = 7 # 图片大小size*size
        self.epochs = 20
        self.batchsize = 128
        self.lr = 1e-3
        self.weight_decay = 0.000
        self.early_stop_num = 15 # acc在多少个epoch下都不提升就提前结束训练
        self.workers = 0 # 多进程，可能会卡程序
        self.pretrained = True
        
        
        self.block_size = 1280 # 一次处理block_size*block_size个像素大小的块
        self.threshold = -0.7 # 阈值,ln(0.5)=-0.69
#        self.net_path = r"checkpoint\best-cnn.pkl"
#        self.net_path = r"checkpoint\best-unet.pkl"
        self.test_img_path = r"E:\pic\baidu\all-band-linear5-norm.tif"
        
        self.flag = 4 # 调参用，预测哪种模型，1为玉米，2为大豆，3为水稻，4为背景
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")