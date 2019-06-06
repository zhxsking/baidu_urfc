# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lightgbm as lgb

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
#    dataloader_train = DataLoader(dataset=TensorDataset(imgs_train, visits_train, labs_train),
#                                  batch_size=opt.batchsize,  num_workers=opt.workers)
#    dataloader_val = DataLoader(dataset=TensorDataset(imgs_val, visits_val, labs_val),
#                                  batch_size=opt.batchsize, num_workers=opt.workers)
#    dataloader_test = DataLoader(dataset=TensorDataset(imgs_test, visits_test),
#                                batch_size=opt.batchsize, num_workers=opt.workers)
#    
#    # 加载模型
#    net = mResNet18().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-ori.pkl", map_location=opt.device)
#    net.load_state_dict(state['net'])
    
    # In[]
    
#    since = time.time() # 记录时间
#    net.eval()
#    fea_train = torch.Tensor(1,net.fc.in_features).to(opt.device)
#    with torch.no_grad():
#        for (img, visit, _) in tqdm(dataloader_train):
#            img = img.to(opt.device)
#            visit = visit.to(opt.device)
#            _, out_fea = net(img, visit)
#            out_fea = out_fea.squeeze()
#            fea_train = torch.cat((fea_train, out_fea), dim=0)
#    fea_train = fea_train[0:-1, :].cpu().numpy()
#    lab_train = labs_train.numpy()
#    time_elapsed = time.time() - since # 用时
#    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#    
#    since = time.time() # 记录时间
#    net.eval()
#    fea_val = torch.Tensor(1,net.fc.in_features).to(opt.device)
#    with torch.no_grad():
#        for (img, visit, _) in tqdm(dataloader_val_ori):
#            img = img.to(opt.device)
#            visit = visit.to(opt.device)
#            _, out_fea = net(img, visit)
#            out_fea = out_fea.squeeze()
#            fea_val = torch.cat((fea_val, out_fea), dim=0)
#    fea_val = fea_val[0:-1, :].cpu().numpy()
#    lab_val = labs_val.numpy()
#    time_elapsed = time.time() - since # 用时
#    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#    
#    since = time.time() # 记录时间
#    net.eval()
#    fea_test = torch.Tensor(1,net.fc.in_features).to(opt.device)
#    with torch.no_grad():
#        for (img, visit) in tqdm(dataloader_test):
#            img = img.to(opt.device)
#            visit = visit.to(opt.device)
#            _, out_fea = net(img, visit)
#            out_fea = out_fea.squeeze()
#            fea_test = torch.cat((fea_test, out_fea), dim=0)
#    fea_test = fea_test[0:-1, :].cpu().numpy()
#    time_elapsed = time.time() - since # 用时
#    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    # In[]
    
    fea_train = imgs_train.cpu().numpy()
    fea_train = fea_train.reshape((fea_train.shape[0],-1))
    lab_train = labs_train.numpy()
    fea_val = imgs_val.cpu().numpy()
    fea_val = fea_val.reshape((fea_val.shape[0],-1))
    lab_val = labs_val.numpy()
    fea_test = imgs_test.cpu().numpy()
    fea_test = fea_test.reshape((fea_test.shape[0],-1))
    
    lgb_train = lgb.Dataset(fea_train, lab_train)
    lgb_eval = lgb.Dataset(fea_val, lab_val, reference=lgb_train)
    
    # 参数
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass', # 目标函数
        'metric': 'multi_logloss',  # 评估函数
        'num_class': 9,
        'num_iterations': 100,
        'num_leaves': 31,   # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9, # 建树的特征选择比例
        'bagging_fraction': 0.8, # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
#        "device": "gpu",
#        "gpu_platform_id": 0,
#        "gpu_device_id": 0,
    }
    
    # 训练
    print('Start training...')
    gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval,early_stopping_rounds=10)
    
    # 保存模型到文件
    gbm.save_model('model.txt')
    
    # 预测数据集
    print('Start predicting...')
    y_pred = gbm.predict(fea_train, num_iteration=gbm.best_iteration)
    y_predy = np.argmax(y_pred, axis=1)
    print('The acc of prediction is:', sum(lab_train==y_predy) / len(y_pred))
    
    y_pred_val = gbm.predict(fea_val, num_iteration=gbm.best_iteration)
    y_predy_val = np.argmax(y_pred_val, axis=1)
    print('The acc of prediction is:', sum(lab_val==y_predy_val) / len(y_pred_val))
    
    y_pred_test = gbm.predict(fea_test, num_iteration=gbm.best_iteration)
    y_predy_test = np.argmax(y_pred_test, axis=1)
    
    f = open(r"data/out-label-lgb.txt", "w+")
    cnt = 0
    for j in range(len(y_predy_test)):
        f.write("{} \t {}\n".format(str(cnt).zfill(6), str(y_predy_test[j]+1).zfill(3)))
        cnt += 1
    f.close()
    