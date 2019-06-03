# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import sys
import os
from os.path import join
import copy
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from preprocess import imgProc
from cnn import CNN, mResNet18
from urfc_dataset import UrfcDataset
from urfc_option import Option


def evalNet(net, loss_func, dataloader_val, device):
    """用验证集评判网络性能"""
    net.eval()
    acc_temp = 0
    loss_temp = 0
    with torch.no_grad():
        for cnt, (img, visit, out_gt) in enumerate(dataloader_val, 1):
            img = img.to(opt.device)
            visit = visit.to(opt.device)
            out_gt = out_gt.to(opt.device)
            out = net(img, visit)

            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            loss_temp += loss.item()
            acc_temp += (float(torch.sum(preds == out_gt.data)) / len(out_gt))
    return loss_temp / cnt, acc_temp / cnt


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    imgs_train = np.load(join(opt.data_npy, "train-img.npy"))
    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
    imgs_val_ori = np.load(join(opt.data_npy, "val-img-ori.npy"))
    visits_train = np.load(join(opt.data_npy, "train-visit.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_train = np.load(join(opt.data_npy, "train-label.npy"))
    labs_val = np.load(join(opt.data_npy, "val-label.npy"))
    
    imgs_train = imgProc(imgs_train)
    imgs_val = imgProc(imgs_val)
    imgs_val_ori = imgProc(imgs_val_ori)
    visits_train = torch.FloatTensor(visits_train.transpose(0,3,1,2))
    visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
    labs_train = torch.LongTensor(labs_train) - 1 # 网络输出从0开始，数据集标签从1开始
    labs_val = torch.LongTensor(labs_val) - 1
    
    dataloader_train = DataLoader(dataset=TensorDataset(imgs_train, visits_train, labs_train),
                                  batch_size=opt.batchsize,  num_workers=opt.workers, pin_memory=False)
    dataloader_val = DataLoader(dataset=TensorDataset(imgs_val, visits_val, labs_val),
                                  batch_size=opt.batchsize, num_workers=opt.workers, pin_memory=False)
    dataloader_val_ori = DataLoader(dataset=TensorDataset(imgs_val_ori, visits_val, labs_val),
                                  batch_size=opt.batchsize, num_workers=opt.workers, pin_memory=False)
    
    # 加载模型
    net = mResNet18().to(opt.device)
    state = torch.load(r"checkpoint\cnn-epoch-9.pkl", map_location=opt.device)
    net.load_state_dict(state['net'])
    
    # In[]
    
    since = time.time() # 记录时间
    net.eval()
    fea = torch.Tensor(1,2048).to(opt.device)
    with torch.no_grad():
        for (img, visit, lab) in tqdm(dataloader_val):
            img = img.to(opt.device)
            visit = visit.to(opt.device)
            _, out_fea = net(img, visit)
            out_fea = out_fea.squeeze()
            fea = torch.cat((fea, out_fea), dim=0)
    fea = fea[0:-1, :].cpu().numpy()
    lab = labs_val.numpy()
    time_elapsed = time.time() - since # 用时
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    # In[]
    X_train,X_test,y_train,y_test =train_test_split(fea,lab,test_size=0.2)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass', # 目标函数
        'metric': 'multi_logloss',  # 评估函数
        'num_class': 9,
        'num_trees': 100,
        'num_leaves': 31,   # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
#        'feature_fraction': 0.9, # 建树的特征选择比例
#        'bagging_fraction': 0.8, # 建树的样本采样比例
#        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
#        'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    
    print('Start training...')
    # 训练 cv and train
    gbm = lgb.train(params,lgb_train,num_boost_round=50,valid_sets=lgb_eval,early_stopping_rounds=10)
    
    print('Save model...')
    # 保存模型到文件
    gbm.save_model('model.txt')
    
    print('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_predy = np.argmax(y_pred, axis=1)
    # 评估模型
    print('The acc of prediction is:', sum(y_test==y_predy) / len(y_pred))
    
    f = open(r"data/out-label-lgb.txt", "w+")
    cnt = 0
    for j in range(len(y_predy)):
        f.write("{} \t {}\n".format(str(cnt).zfill(6), str(y_predy[j]).zfill(3)))
        cnt += 1
    f.close()
    