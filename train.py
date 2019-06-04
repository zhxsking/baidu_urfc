# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
from os.path import join
import copy
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from preprocess import imgProc
from cnn import mResNet, mDenseNet
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
            out, _ = net(img, visit)

            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            loss_temp += loss.item()
            acc_temp += (float(torch.sum(preds == out_gt.data)) / len(out_gt))
    return loss_temp / cnt, acc_temp / cnt


if __name__ == '__main__':
    __spec__ = None
    opt = Option()

    # 初始化保存目录
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
#    else:
#        shutil.rmtree('checkpoint')
#        os.makedirs('checkpoint')
    
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
                                  batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers)
    dataloader_val = DataLoader(dataset=TensorDataset(imgs_val, visits_val, labs_val),
                                  batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers)
    dataloader_val_ori = DataLoader(dataset=TensorDataset(imgs_val_ori, visits_val, labs_val),
                                  batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers)
    
#    dataset_train = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/train.txt")
#    dataloader_train = DataLoader(dataset=dataset_train, batch_size=opt.batchsize,
#                            shuffle=True, num_workers=opt.workers)   
#    dataset_val = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt")
#    dataloader_val = DataLoader(dataset=dataset_val, batch_size=opt.batchsize,
#                                shuffle=True, num_workers=opt.workers)
    
    # 定义网络及其他
#    net = CNN().to(opt.device)
    net = mResNet18(pretrained=True).to(opt.device)
    loss_func = nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.epochs//8)+1, eta_min=1e-08) # 动态改变lr
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # 动态改变lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5, patience=3, verbose=True)
    
    # 冻结层
#    for count, (name, param) in enumerate(net.named_parameters(), 1):
#        if 'layer' in name:
#            param.requires_grad = False
    
    # 初始化
    since = time.time() # 记录时间
    early_stop = 0
    loss_list_train = []
    acc_list_train = []
    loss_list_val = []
    acc_list_val = []
    loss_list_val_ori = []
    acc_list_val_ori = []
    best_acc = 0.0
    best_epoch = 1
    best_model = copy.deepcopy(net.state_dict())
    best_acc_ori = 0.0
    best_epoch_ori = 1
    best_model_ori = copy.deepcopy(net.state_dict())
    best_loss = 99.0
    best_epoch_loss = 1
    best_model_loss = copy.deepcopy(net.state_dict())
    
    # 训练
    print('Start Training...')
    for epoch in range(opt.epochs):
        loss_temp_train = 0.0
        acc_temp_train = 0.0
        net.train()
#        scheduler.step(epoch)
        for cnt, (img, visit, out_gt) in enumerate(dataloader_train, 1):
            img = img.to(opt.device)
            visit = visit.to(opt.device)
            out_gt = out_gt.to(opt.device)
            out, _ = net(img, visit)
            
#            sys.exit(0)
            
            loss = loss_func(out, out_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('\rbatch {}/{} interim loss is: {:.4f}'
                  .format(cnt, len(dataloader_train), loss.item()), end='\r')
            
            _, preds = torch.max(out, 1)
            loss_temp_train += loss.item()
            acc_temp_train += (float(torch.sum(preds == out_gt.data)) / len(out_gt))
        loss_temp_train /= cnt
        acc_temp_train /= cnt
        loss_list_train.append(loss_temp_train)
        acc_list_train.append(acc_temp_train)
        
        # 验证
        loss_temp_val, acc_temp_val = evalNet(net, loss_func, dataloader_val, opt.device)
        loss_list_val.append(loss_temp_val)
        acc_list_val.append(acc_temp_val)
        
        # 验证原始数据
        loss_temp_val_ori, acc_temp_val_ori = evalNet(net, loss_func, dataloader_val_ori, opt.device)
        loss_list_val_ori.append(loss_temp_val_ori)
        acc_list_val_ori.append(acc_temp_val_ori)
        
        scheduler.step(acc_temp_val_ori)
        
        # 更新最优模型
        if epoch > 0 and acc_temp_val >= best_acc:
            best_epoch = epoch + 1
            best_acc = acc_temp_val
            best_model = copy.deepcopy(net.state_dict())
            early_stop = 0
        elif epoch > 0:
            early_stop += 1
            if early_stop == opt.early_stop_num: break
        
        # 更新最优原始模型
        if epoch > 0 and acc_temp_val_ori >= best_acc_ori:
            best_epoch_ori = epoch + 1
            best_acc_ori = acc_temp_val_ori
            best_model_ori = copy.deepcopy(net.state_dict())
            early_stop = 0
        elif epoch > 0:
            early_stop += 1
            if early_stop == opt.early_stop_num: break
        
        # 更新最优loss模型
        if epoch > 0 and loss_temp_val_ori <= best_loss:
            best_epoch_loss = epoch + 1
            best_loss = loss_temp_val_ori
            best_model_loss = copy.deepcopy(net.state_dict())
            early_stop = 0
        elif epoch > 0:
            early_stop += 1
            if early_stop == opt.early_stop_num: break
        
        print('\repoch {}/{}, train val val-ori loss {:.4f} {:.4f} {:.4f}, acc {:.4f} {:.4f} {:.4f}'
              .format(epoch+1, opt.epochs, loss_temp_train, loss_temp_val, 
                      loss_temp_val_ori, acc_temp_train, acc_temp_val, acc_temp_val_ori))
        torch.save({'net':net.state_dict()}, r'checkpoint/cnn-epoch-{}.pkl'.format(epoch+1))
    # 保存最佳模型
    best_net_state = {
            'best_epoch': best_epoch,
            'best_acc': best_acc,
            'net': best_model,
            }
    torch.save(best_net_state, r'checkpoint/best-cnn.pkl')
    
    # 保存最佳原始模型
    best_net_state_ori = {
            'best_epoch': best_epoch_ori,
            'best_acc': best_acc_ori,
            'net': best_model_ori,
            }
    torch.save(best_net_state_ori, r'checkpoint/best-cnn-ori.pkl')
    
    # 保存最佳loss模型
    best_net_state_loss = {
            'best_epoch': best_epoch_loss,
            'best_loss': best_loss,
            'net': best_model_loss,
            }
    torch.save(best_net_state_loss, r'checkpoint/best-cnn-loss.pkl')
    
    # 保存最终模型以及参数
    time_elapsed = time.time() - since # 用时
    final_net_state = {
            'epoch': epoch+1,
            'time': time_elapsed,
            'loss_list_train': loss_list_train,
            'acc_list_train': acc_list_train,
            'loss_list_val': loss_list_val,
            'acc_list_val': acc_list_val,
            'loss_list_val_ori': loss_list_val_ori,
            'acc_list_val_ori': acc_list_val_ori,
            'optimizer': optimizer.state_dict(),
            'net': net.state_dict(),
            }
    torch.save(final_net_state, r'checkpoint/final-cnn.pkl')
    
    # 显示训练信息
    print('-' * 50)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc {:4f} in epoch {}, val-ori Acc {:4f} in epoch {}, Best loss {:4f} in epoch {}'
          .format(best_acc, best_epoch, best_acc_ori, best_epoch_ori, best_loss, best_epoch_loss))
    
    # 训练完显示loss及Acc曲线
    plt.figure()
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(loss_list_train)
    plt.plot(loss_list_val)
    plt.plot(loss_list_val_ori)
    plt.subplot(122)
    plt.title('Acc')
    plt.plot(acc_list_train)
    plt.plot(acc_list_val)
    plt.plot(acc_list_val_ori)
    plt.show()
    
    
    