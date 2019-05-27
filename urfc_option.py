# -*- coding: utf-8 -*-

import torch

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        self.epochs = 50
        self.lr = 1e-3
        self.batchsize = 1024 # 128 1024
        self.weight_decay = 0.00
        self.early_stop_num = 150 # acc在多少个epoch下都不提升就提前结束训练
        self.workers = 4 # 4 多进程，可能会卡程序
        self.pretrained = False
        
        self.threshold = -0.7 # 阈值,ln(0.5)=-0.69
#        self.net_path = r"checkpoint\best-cnn.pkl"
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dir_npy_suffix = "_npy_26_24_7"
        self.dir_img = r"E:\pic\URFC-baidu\train_image"
        self.dir_visit = r"E:\pic\URFC-baidu\train_visit"
        self.dir_visit_test = r"E:\pic\URFC-baidu\test_visit"
        self.dir_visit_npy = self.dir_visit + self.dir_npy_suffix
        self.dir_visit_npy_test = self.dir_visit_test + self.dir_npy_suffix
        
        