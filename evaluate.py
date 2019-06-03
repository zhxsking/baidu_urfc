# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from preprocess import imgProc
from cnn import mResNet18
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
            img = img.to(opt.device)
            visit = visit.to(opt.device)
            out_gt = out_gt.to(opt.device)
            out, _ = net(img, visit)

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
    imgs_val_ori = np.load(join(opt.data_npy, "val-img-ori.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_val = np.load(join(opt.data_npy, "val-label.npy"))
    
    imgs_val = imgProc(imgs_val)
    imgs_val_ori = imgProc(imgs_val_ori)
    visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
    labs_val = torch.LongTensor(labs_val) - 1
    
    dataloader_val = DataLoader(dataset=TensorDataset(imgs_val, visits_val, labs_val),
                                  batch_size=opt.batchsize, num_workers=opt.workers)
    dataloader_val_ori = DataLoader(dataset=TensorDataset(imgs_val_ori, visits_val, labs_val),
                                  batch_size=opt.batchsize, num_workers=opt.workers)
    
    # 加载模型
    net = mResNet18().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-ori.pkl", map_location=opt.device)
    net.load_state_dict(state['net'])
    loss_func = nn.CrossEntropyLoss().to(opt.device)
    
    # 验证原始数据
    loss_temp_val_ori, acc_temp_val_ori, out_lab = evalNet(net, loss_func, dataloader_val_ori, opt.device)

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
    
    