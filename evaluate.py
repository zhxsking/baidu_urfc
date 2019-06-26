# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torchvision.transforms as transforms
import sys
from tqdm import tqdm

from multimodal import MultiModalNet

from urfc_dataset import UrfcDataset
from urfc_utils import Logger, Record, imgProc, aug_batch, aug_val_batch, get_tta_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet50, mSDNet50_p, mSDNet101, mPNASNet, MMNet
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
    
    
def evalNet(loss_func, dataloader_val, device, *nets):
    """用验证集评判网络性能"""
    sm = nn.Softmax(dim=1)
    acc_temp = Record()
    loss_temp = Record()
    labs_ori, labs_out = [], []
    with torch.no_grad():
        for (img, visit, out_gt) in tqdm(dataloader_val):
            img = img.to(device)
            visit = visit.to(device)
            out_gt = out_gt.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                out_tmp = net(img, visit)
                
                if isinstance(out_tmp, tuple):
                    out_tmp = out_tmp[0]
                
                out_tmp = sm(out_tmp)

                if (cnt==0):
                    out = out_tmp
                else:
                    out = out + out_tmp
            
            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            
            loss_temp.update(loss.item(), img.shape[0])
            acc_temp.update((float(torch.sum(preds == out_gt.data)) / len(out_gt)), len(out_gt))
            labs_ori.append(out_gt.cpu().numpy())
            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8))
    labs_ori_np = []
    labs_out_np = []
    for j in range(len(labs_ori)):
        for i in range(len(labs_ori[j])):
            labs_ori_np.append(labs_ori[j][i])
            labs_out_np.append(labs_out[j][i])            
    labs_ori_np = np.array(labs_ori_np)
    labs_out_np = np.array(labs_out_np)
    return loss_temp.avg, acc_temp.avg, labs_ori_np, labs_out_np

def evalNet_TTA(loss_func, dataloader_val, device, *nets):
    """用验证集评判网络性能, TTA"""
    sm = nn.Softmax(dim=1)
    acc_temp = Record()
    loss_temp = Record()
    labs_ori, labs_out = [], []
    with torch.no_grad():
        for (img_o, visit, out_gt) in tqdm(dataloader_val):
            img_tta = get_tta_batch(img_o) # 得到tta之后的图像
            
            img_o = img_o.to(device)
            visit = visit.to(device)
            out_gt = out_gt.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                out_o = net(img_o, visit)
                
                if isinstance(out_o, tuple):
                    out_o = out_o[0]
                
                out_o = sm(out_o)
                
                for i in range(len(img_tta)):
                    out_tta_tmp = net(img_tta[i].to(device), visit)
                    
                    if isinstance(out_tta_tmp, tuple):
                        out_tta_tmp = out_tta_tmp[0]
                    
                    out_tta_tmp = sm(out_tta_tmp)
                    if (i==0):
                        out_tta = out_tta_tmp
                    else:
                        out_tta = out_tta + out_tta_tmp
                
                out_tmp = out_o * (i+1) + out_tta

                if (cnt==0):
                    out = out_tmp
                else:
                    out = out + out_tmp
            
            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            
            loss_temp.update(loss.item(), img_o.shape[0])
            acc_temp.update((float(torch.sum(preds == out_gt.data)) / len(out_gt)), len(out_gt))
            labs_ori.append(out_gt.cpu().numpy())
            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8))
    labs_ori_np = []
    labs_out_np = []
    for j in range(len(labs_ori)):
        for i in range(len(labs_ori[j])):
            labs_ori_np.append(labs_ori[j][i])
            labs_out_np.append(labs_out[j][i])            
    labs_ori_np = np.array(labs_ori_np)
    labs_out_np = np.array(labs_out_np)
    return loss_temp.avg, acc_temp.avg, labs_ori_np, labs_out_np


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_val = np.load(join(opt.data_npy, "val-label.npy"))
    
    imgs_val = imgProc(imgs_val, opt.means, opt.stds)
    visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
    labs_val = torch.LongTensor(labs_val) - 1
    
    dataset_val = TensorDataset(imgs_val, visits_val, labs_val)
    dataloader_val = DataLoader(dataset=dataset_val, shuffle=False, 
                                batch_size=256, num_workers=opt.workers)
    
#    dataset_val = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt", aug=False)
#    dataloader_val = DataLoader(dataset=dataset_val, batch_size=128,
#                                shuffle=False, num_workers=opt.workers)
    
    # 加载模型
    print('Loading Model...')
    loss_func = nn.CrossEntropyLoss().to(opt.device)
    
    net0 = mSDNet50_p().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-50-p.pkl", map_location=opt.device)
    net0.load_state_dict(state['net'])
    
#    net1 = mSDNet50().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-50-59.pkl", map_location=opt.device)
#    net1.load_state_dict(state['net'])
    
    net2 = mSDNet101().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-101.pkl", map_location=opt.device)
    net2.load_state_dict(state['net'])
#    
#    net3 = MMNet().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-mmnet-59.pkl", map_location=opt.device)
#    net3.load_state_dict(state['net'])
#    
    net4 = mSDNet101().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-101.pkl", map_location=opt.device)
    net4.load_state_dict(state['net'])
    
#    netm = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(opt.device)
#    state = torch.load(r"checkpoint\multimodal_fold_0_model_best_loss.pth.tar", map_location=opt.device)
#    netm.load_state_dict(state['state_dict'])
#    
#    netm1 = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-mutimodel.pkl", map_location=opt.device)
#    netm1.load_state_dict(state['net'])
    
    #%% 验证原始数据
#    nets = [net0]
    nets = [net0]
    loss, acc, labs_ori_np, labs_out_np = evalNet(loss_func, dataloader_val, opt.device, *nets)
#    loss, acc, labs_ori_np, labs_out_np = evalNet_TTA(loss_func, dataloader_val, opt.device, *nets)
    
    #%%
#    device = opt.device
#    sm = nn.Softmax(dim=1)  # nn.functional.normalize
#    acc_temp = Record()
#    loss_temp = Record()
#    labs_ori, labs_out = [], []
#    with torch.no_grad():
#        for (img_o, visit, out_gt) in tqdm(dataloader_val):
#            img_tta = get_tta_batch(img_o)
#            
#            img_o = img_o.to(device)
#            visit = visit.to(device)
#            out_gt = out_gt.to(device)
#            
#            for cnt, net in enumerate(nets):
#                net.eval()
#                out_o = net(img_o, visit)
#                
#                if isinstance(out_o, tuple):
#                    out_o = out_o[0]
#                
##                out_o = out_o + 2*torch.mul(out_o, torch.le(out_o,0).float()) # 负的更负
#                
#                out_o = sm(out_o)
#                
#                for i in range(len(img_tta)):
#                    out_tta_tmp = net(img_tta[i].to(device), visit)
#                    
#                    if isinstance(out_tta_tmp, tuple):
#                        out_tta_tmp = out_tta_tmp[0]
#                    
##                    out_tta_tmp = out_tta_tmp + 2*torch.mul(out_tta_tmp, torch.le(out_tta_tmp,0).float())
#                    
#                    out_tta_tmp = sm(out_tta_tmp)
#                    
#                    if (i==0):
#                        out_tta = out_tta_tmp
#                    else:
#                        out_tta = out_tta + out_tta_tmp
#                
#                out_tmp = out_o * 7 + out_tta
#
#                if (cnt==0):
#                    out = out_tmp
#                else:
#                    out = out + out_tmp
#            
#            loss = loss_func(out, out_gt)
#            _, preds = torch.max(out, 1)
#            
#            loss_temp.update(loss.item(), img_o.shape[0])
#            acc_temp.update((float(torch.sum(preds == out_gt.data)) / len(out_gt)), len(out_gt))
#            labs_ori.append(out_gt.cpu().numpy())
#            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8))
#    labs_ori_np = []
#    labs_out_np = []
#    for j in range(len(labs_ori)):
#        for i in range(len(labs_ori[j])):
#            labs_ori_np.append(labs_ori[j][i])
#            labs_out_np.append(labs_out[j][i])            
#    labs_ori_np = np.array(labs_ori_np)
#    labs_out_np = np.array(labs_out_np)
#    loss, acc =  loss_temp.avg, acc_temp.avg

    #%% 绘制混淆矩阵, 计算acc
    cm = metrics.confusion_matrix(labs_ori_np, labs_out_np)
    acc_all_val = metrics.accuracy_score(labs_ori_np, labs_out_np)
    plotConfusionMatrix(cm)
    print('val acc {:.4f}'.format(acc_all_val))
    

    