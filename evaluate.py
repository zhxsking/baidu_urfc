# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torchvision.transforms as transforms
import sys

from urfc_utils import Logger, imgProc, aug_batch, aug_val_batch, aug_test_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet
from urfc_option import Option


def plotConfusionMatrix(cm):
    '''绘制混淆矩阵'''
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('confusion matrix')
    iters = np.reshape([[[i,j] for j in range(9)] for i in range(9)],(cm.size,2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]), horizontalalignment='center') # 显示对应的数字
    plt.ylabel('GT')
    plt.xlabel('Prediction')
    plt.show()
    
    
def evalNet(net, loss_func, dataloader_val, device):
    """用验证集评判网络性能"""
    net.eval()
    acc_temp = 0
    loss_temp = 0
    out_lab = []
    with torch.no_grad():
        for cnt, (img, visit, out_gt) in enumerate(dataloader_val, 1):
            img = img.to(device)
            visit = visit.to(device)
            out_gt = out_gt.to(device)
            out, _ = net(img, visit)

            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            loss_temp += loss.item()
            acc_temp += (float(torch.sum(preds == out_gt.data)) / len(out_gt))
            out_lab.append(preds.cpu().numpy().flatten().astype(np.uint8))
    return loss_temp / cnt, acc_temp / cnt, out_lab

def evalNet_TTA(net, loss_func, dataloader_val, device):
    """用验证集评判网络性能"""
    net.eval()
    acc_temp = 0
    loss_temp = 0
    out_lab = []
    with torch.no_grad():
        for cnt, (img_o, visit, out_gt) in enumerate(dataloader_val, 1):
            img_h, img_v = aug_test_batch(img_o)
            
            img_o = img_o.to(device)
            img_h = img_h.to(device)
            img_v = img_v.to(device)
            visit = visit.to(device)
            out_gt = out_gt.to(device)
            
            out_o, _ = net(img_o, visit)
            out_h, _ = net(img_h, visit)
            out_v, _ = net(img_v, visit)
            out = out_o + out_h + out_v
            
            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            loss_temp += loss.item()
            acc_temp += (float(torch.sum(preds == out_gt.data)) / len(out_gt))
            out_lab.append(preds.cpu().numpy().flatten().astype(np.uint8))
    return loss_temp / cnt, acc_temp / cnt, out_lab


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_val = np.load(join(opt.data_npy, "val-label.npy"))
    
    imgs_val = imgProc(imgs_val)
    visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
    labs_val = torch.LongTensor(labs_val) - 1
    
    dataloader_val = DataLoader(dataset=TensorDataset(imgs_val, visits_val, labs_val),
                                  batch_size=opt.batchsize, num_workers=opt.workers)
    
    # 加载模型
    print('Loading Model...')
    net = mSDNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn.pkl", map_location=opt.device)
    net.load_state_dict(state['net'])
    loss_func = nn.CrossEntropyLoss().to(opt.device)
    
    # 验证原始数据
#    loss_temp_val_ori, acc_temp_val_ori, out_lab = evalNet(net, loss_func, dataloader_val, opt.device)
    loss_temp_val_ori, acc_temp_val_ori, out_lab = evalNet_TTA(net, loss_func, dataloader_val, opt.device)

    # 绘制混淆矩阵
    out_lab_np = []
    for j in range(len(out_lab)):
        for i in range(len(out_lab[j])):
            out_lab_np.append(out_lab[j][i])
    out_lab_np = np.array(out_lab_np)
    labs_val_np = labs_val.cpu().numpy()
    cm = metrics.confusion_matrix(labs_val_np, out_lab_np)
    acc_all_val = metrics.accuracy_score(labs_val_np, out_lab_np)
    plotConfusionMatrix(cm)
    print('val acc {:.4f}'.format(acc_all_val))
    
    