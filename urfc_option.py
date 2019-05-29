# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        self.epochs = 20
        self.lr = 1e-3
        self.batchsize = 1024 # 256 1024
        self.weight_decay = 0.00
        self.early_stop_num = 150 # acc在多少个epoch下都不提升就提前结束训练
        self.workers = 0 # 4 多进程，可能会卡程序
        self.pretrained = False
        self.weight = torch.Tensor([0.76145, 0.81155, 0.91025, 0.96605, 0.9134, 0.862325, 0.912075, 0.934575, 0.928325])
        
        self.threshold = -0.7 # 阈值,ln(0.5)=-0.69
#        self.net_path = r"checkpoint\best-cnn.pkl"
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dir_npy_suffix = "_npy_26_24_7"
        self.dir_img = r"E:\pic\URFC-baidu\train_image"
        self.data_npy = r"E:\pic\URFC-baidu\data_npy"
        self.dir_visit = r"E:\pic\URFC-baidu\train_visit"
        self.dir_visit_test = r"E:\pic\URFC-baidu\test_visit"
        self.dir_visit_npy = self.dir_visit + self.dir_npy_suffix
        self.dir_visit_npy_test = self.dir_visit_test + self.dir_npy_suffix
        
        self.means = (-1.3326176e-09, -5.8395827e-10, -1.153197e-10)
        self.stds =(0.11115803, 0.09930103, 0.08884794)
        
        seed = 47
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()