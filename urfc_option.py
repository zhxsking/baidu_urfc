# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        self.epochs = 20
        self.lr = 1e-4
        self.batchsize = 1024 # 512 1024
        self.weight_decay = 0.1
        self.early_stop_num = 150 # acc在多少个epoch下都不提升就提前结束训练
        self.workers = 0 # 4 多进程，可能会卡程序
        self.pretrained = True
        
        self.weight = torch.Tensor([0.66145, 0.81155, 0.81025, 0.99605, 0.8634, 0.812325, 0.812075, 0.934575, 0.928325])
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.num_val = 500
        self.num_train = 1000
        
        
        self.dir_npy_suffix = "_npy_26_24_7"
        self.disk = "E"
        self.dir_img = self.disk + r":\pic\URFC-baidu\train_image"
        self.dir_img_val = self.disk + r":\pic\URFC-baidu\val_image"
        self.dir_img_test = self.disk + r":\pic\URFC-baidu\test_image"
        self.data_npy = self.disk + r":\pic\URFC-baidu\data_npy_" + str(self.num_train)
        self.dir_visit = self.disk + r":\pic\URFC-baidu\train_visit"
        self.dir_visit_test = self.disk + r":\pic\URFC-baidu\test_visit"
        self.dir_visit_npy = self.dir_visit + self.dir_npy_suffix
        self.dir_visit_npy_test = self.dir_visit_test + self.dir_npy_suffix
        
#        self.means = (-1.2232209e-09, -7.9797535e-10, -3.0236139e-10)
#        self.stds =(0.10928233, 0.098093316, 0.087956846)
        
        seed = 47
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()