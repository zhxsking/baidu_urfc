# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import imgaug as ia

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        self.epochs = 5
        self.lr = 3e-3 # 3e-3
        self.batchsize = 256
        self.weight_decay = 1e-4
        self.early_stop_num = 150 # acc在多少个epoch下都不提升就提前结束训练
        self.workers = 4 # 1 4 多进程，可能会卡程序
        self.pretrained = True
        
        self.weight = torch.Tensor([0.66145, 0.81155, 0.81025, 0.99605, 0.8634, 0.812325, 0.812075, 0.934575, 0.928325])
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.num_train = 500
        
        self.dir_npy_suffix = "_npy_26_24_7"
#        self.disk = r"E:\pic\URFC-baidu"
        self.disk = r"E:\pic\URFC-baidu-2"
        self.dir_img = self.disk + r"\train_image"
        self.dir_img_test = self.disk + r"\test_image"
        self.dir_visit = self.disk + r"\train_visit"
        self.dir_visit_test = self.disk + r"\test_visit"
        self.data_npy = self.disk + r"\data_npy_" + str(self.num_train)
        self.dir_visit_npy = self.dir_visit + self.dir_npy_suffix
        self.dir_visit_npy_test = self.dir_visit_test + self.dir_npy_suffix
        
        self.means = (0.4553296, 0.52905685, 0.6211398)
        self.stds =(0.14767659, 0.13119018, 0.1192783)
        
        seed = 47
        ia.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
#        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        