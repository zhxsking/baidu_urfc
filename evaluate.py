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
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet50, mSDNet101, mPNASNet, MMNet
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
                out_tmp, _ = net(img, visit)

                if (cnt==0):
                    out = sm(out_tmp)
                else:
                    out = out + sm(out_tmp)
            
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
            img_h, img_v = get_tta_batch(img_o)
            
            img_o = img_o.to(device)
            img_h = img_h.to(device)
            img_v = img_v.to(device)
            visit = visit.to(device)
            out_gt = out_gt.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                out_o, _ = net(img_o, visit)
                out_h, _ = net(img_h, visit)
                out_v, _ = net(img_v, visit)
                out_tmp = sm(out_o) * 2 + sm(out_h) + sm(out_v)

                if (cnt==0):
                    out = sm(out_tmp)
                else:
                    out = out + sm(out_tmp)
            
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
    
    imgs_val = imgProc(imgs_val)
    visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
    labs_val = torch.LongTensor(labs_val) - 1
    
    dataset_val = TensorDataset(imgs_val, visits_val, labs_val)
    dataloader_val = DataLoader(dataset=dataset_val, shuffle=False, 
                                batch_size=opt.batchsize, num_workers=opt.workers)
    
#    dataset_val = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt", aug=False)
#    dataloader_val = DataLoader(dataset=dataset_val, batch_size=opt.batchsize,
#                                shuffle=False, num_workers=opt.workers)
    
    # 加载模型
    print('Loading Model...')
    net = mSDNet50().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-50.pkl", map_location=opt.device) # 0.6139 实测0.6394
    net.load_state_dict(state['net'])
    loss_func = nn.CrossEntropyLoss().to(opt.device)
    
    net1 = mPNASNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-pnasnet.pkl", map_location=opt.device) # 0.59 实测0.6077
    net1.load_state_dict(state['net'])
    
    net2 = mSDNet101().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-101.pkl", map_location=opt.device) # 0.6178 实测0.6526
    net2.load_state_dict(state['net'])
    
    net3 = MMNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-mmnet.pkl", map_location=opt.device) # 0.6200 实测0.6531
    net3.load_state_dict(state['net'])
    
#    net3 = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(opt.device)
#    state = torch.load(r"checkpoint\multimodal_fold_0_model_best_loss.pth.tar", map_location=opt.device)
#    net3.load_state_dict(state['state_dict'])
    
    #%% 验证原始数据
#    nets = [net]
    nets = [net, net2, net3]
    loss, acc, labs_ori_np, labs_out_np = evalNet(loss_func, dataloader_val, opt.device, *nets)
#    loss, acc, labs_ori_np, labs_out_np = evalNet_TTA(loss_func, dataloader_val, opt.device, *nets)
    
    #%%
#    net.eval()
##    net1.eval()
##    net2.eval()
##    net3.eval()
#    sm = nn.Softmax(dim=1)
#    acc_temp = Record()
#    loss_temp = Record()
#    labs_ori, labs_out = [], []
#    with torch.no_grad():
#        for cnt, (img, visit, out_gt) in tqdm(enumerate(dataloader_val, 1)):
#            img_h, img_v = get_tta_batch(img)
#            
#            img_h = img_h.to(opt.device)
#            img_v = img_v.to(opt.device)
#            
#            img = img.to(opt.device)
#            visit = visit.to(opt.device)
#            out_gt = out_gt.to(opt.device)
#            
#            out0, _ = net(img, visit)
##            out1, _ = net1(img, visit)
##            out2, _ = net2(img, visit)
##            out3, _ = net3(img, visit)
#            
##            out_o, _ = net3(img, visit)
##            out_h, _ = net3(img_h, visit)
##            out_v, _ = net3(img_v, visit)
##            out = out_o * 2 + out_h + out_v
#            
##            img_ = img.clone()
##            img_[:,0,:,:] = img[:,2,:,:]
##            img_[:,2,:,:] = img[:,0,:,:]
##            out3 = net3(img_, visit)
#            
##            out = (out0) + (out1) + (out2) + (out3)
##            out = sm(out0) + sm(out1)
##            out = sm(out0) + sm(out1) + sm(out2) + sm(out3)
#            out = sm(out0)
#
#            loss = loss_func(out, out_gt)
#            _, preds = torch.max(out, 1)
#            loss_temp.update(loss.item(), img.shape[0])
#            acc_temp.update((float(torch.sum(preds == out_gt.data)) / len(out_gt)), len(out_gt))
#            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8))
#            labs_ori.append(out_gt.cpu().numpy())
#    loss, acc =  loss_temp.avg, acc_temp.avg
#    labs_ori_np = []
#    labs_out_np = []
#    for j in range(len(labs_ori)):
#        for i in range(len(labs_ori[j])):
#            labs_ori_np.append(labs_ori[j][i])
#            labs_out_np.append(labs_out[j][i])            
#    labs_ori_np = np.array(labs_ori_np)
#    labs_out_np = np.array(labs_out_np)

    # 绘制混淆矩阵, 计算acc
    cm = metrics.confusion_matrix(labs_ori_np, labs_out_np)
    acc_all_val = metrics.accuracy_score(labs_ori_np, labs_out_np)
    plotConfusionMatrix(cm)
    print('val acc {:.4f}'.format(acc_all_val))
    

    