# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool

from preprocess import imgProc
from cnn import mResNet18
from urfc_option import Option


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    imgs_train = np.load(join(opt.data_npy, "train-img.npy"))
    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
    visits_train = np.load(join(opt.data_npy, "train-visit.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_train = np.load(join(opt.data_npy, "train-label.npy"))
    labs_val = np.load(join(opt.data_npy, "val-label.npy"))
    
    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
    
    imgs_train = imgProc(imgs_train)
    imgs_val = imgProc(imgs_val)
    visits_train = torch.FloatTensor(visits_train.transpose(0,3,1,2))
    visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
    labs_train = torch.LongTensor(labs_train) - 1 # 网络输出从0开始，数据集标签从1开始
    labs_val = torch.LongTensor(labs_val) - 1
    
    imgs_test = imgProc(imgs_test)
    visits_test = torch.FloatTensor(visits_test.transpose(0,3,1,2))
    
    
    # In[]
    
    fea_train = imgs_train.cpu().numpy()
    fea_train = fea_train.reshape((fea_train.shape[0],-1))
    lab_train = labs_train.numpy()
    fea_val = imgs_val.cpu().numpy()
    fea_val = fea_val.reshape((fea_val.shape[0],-1))
    lab_val = labs_val.numpy()
    fea_test = imgs_test.cpu().numpy()
    fea_test = fea_test.reshape((fea_test.shape[0],-1))
    
    categorical_features_indices = np.where(fea_train.dtypes != np.float)[0]
    train_pool = Pool(fea_train, lab_train, cat_features=categorical_features_indices)
    val_pool = Pool(fea_val, lab_val, cat_features=categorical_features_indices)
    
    params = {
        'iterations': 500,
        'learning_rate': 0.1,
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'logging_level': 'Verbose',
        'use_best_model': True,
        'od_type': 'Iter', # early stop
        'od_wait': 40,
        'plot': True,
        }
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    
    
    # 预测数据集
    print('Start predicting...')
    y_pred = model.predict(fea_train)
    print('The acc of prediction is:', sum(lab_train==y_pred) / len(y_pred))
    
    y_pred_val = model.predict(fea_val)
    print('The acc of prediction is:', sum(lab_val==y_pred_val) / len(y_pred_val))
    
    y_pred_test = model.predict(fea_test)
    
    f = open(r"data/out-label-catboost.txt", "w+")
    cnt = 0
    for j in range(len(y_pred_test)):
        f.write("{} \t {}\n".format(str(cnt).zfill(6), str(y_pred_test[j]+1).zfill(3)))
        cnt += 1
    f.close()
    