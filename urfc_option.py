# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import imgaug as ia

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        self.epochs = 20
        self.lr = 3e-3 # 3e-3
        self.batchsize = 64
        self.weight_decay = 3e-4
        self.early_stop_num = 150 # acc在多少个epoch下都不提升就提前结束训练
        self.workers = 4 # 1 4 多进程，可能会卡程序
        self.pretrained = True
#        self.weight = torch.Tensor([0.66145, 0.81155, 0.81025, 0.99605, 0.8634, 0.812325, 0.812075, 0.934575, 0.928325])
        self.weight = torch.Tensor([1, 1, 2, 1, 1, 1, 1, 1, 1])
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.num_train = 6000
        
        self.use_tta = True
        self.use_blend = True
        
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
        
        # [0.4762154, 0.5394282, 0.6343212] [0.18453903, 0.17017019, 0.15467317] 360000 TRAIN-OVER
        # [0.46370938, 0.52943575, 0.6256985] [0.1786646, 0.16450335, 0.15119104] val
        self.means = (0.4553296, 0.52905685, 0.6211398)
        self.stds =(0.14767659, 0.13119018, 0.1192783)
        
        self.seed = 47
        ia.seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
#        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

opt = Option()
'''
001 居住区
002 学校
003 工业园区
004 火车站
005 飞机场
006 公园
007 商业区
008 政务区
009 医院
'''


