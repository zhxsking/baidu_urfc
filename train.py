# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os 
import copy
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN
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
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # 初始化保存目录
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    else:
        shutil.rmtree('checkpoint')
        os.makedirs('checkpoint')
    
    # 加载数据
    dataset_train = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/train.txt")
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=opt.batchsize,
                            shuffle=True, num_workers=opt.workers)   
    dataset_val = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt")
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=opt.batchsize,
                                shuffle=True, num_workers=opt.workers)
    
    # 定义网络及其他
    net = CNN().to(opt.device)
    loss_func = nn.NLLLoss().to(opt.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # 动态改变lr
    
    # 初始化
    since = time.time() # 记录时间
    early_stop = 0
    loss_list_train = []
    acc_list_train = []
    loss_list_val = []
    acc_list_val = []
    best_acc = 0.0
    best_epoch = 1
    best_model = copy.deepcopy(net.state_dict())
    
    # 训练
    for epoch in range(opt.epochs):
        loss_temp_train = 0.0
        acc_temp_train = 0.0
        net.train()
        scheduler.step()
        for cnt, (img, visit, out_gt) in enumerate(dataloader_train, 1):
            img = img.to(opt.device)
            visit = visit.to(opt.device)
            out_gt = out_gt.to(opt.device)
            out = net(img, visit)
            
#            sys.exit(0)
            
            loss = loss_func(out, out_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#            print(loss.item())
            
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
        
        # 更新最优模型
        if epoch > 0 and acc_temp_val >= best_acc:
            best_epoch = epoch + 1
            best_acc = acc_temp_val
            best_model = copy.deepcopy(net.state_dict())
            early_stop = 0
        elif epoch > 0:
            early_stop += 1
            if early_stop == opt.early_stop_num: break
        
        print('epoch {}/{} done, train loss {:.4f}, train acc {:.4f}, val loss {:.4f}, val acc {:.4f}'
              .format(epoch+1, opt.epochs, loss_temp_train, acc_temp_train, loss_temp_val, acc_temp_val))
    # 保存最佳模型
    best_net_state = {
            'best_epoch': best_epoch,
            'best_acc': best_acc,
            'net': best_model,
            }
    torch.save(best_net_state, r'checkpoint/best-cnn.pkl')
    
    # 保存最终模型以及参数
    time_elapsed = time.time() - since # 用时
    final_net_state = {
            'epoch': epoch+1,
            'time': time_elapsed,
            'loss_list_train': loss_list_train,
            'acc_list_train': acc_list_train,
            'loss_list_val': loss_list_val,
            'acc_list_val': acc_list_val,
            'optimizer': optimizer.state_dict(),
            'net': net.state_dict(),
            }
    torch.save(final_net_state, r'checkpoint/final-cnn.pkl')
    
    # 显示训练信息
    print('-' * 50)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc {:4f} in epoch {}'.format(best_acc, best_epoch))
    
    # 训练完显示loss及Acc曲线
    plt.figure()
    plt.subplot(121)
    plt.plot(loss_list_train)
    plt.plot(loss_list_val)
    plt.subplot(122)
    plt.plot(acc_list_train)
    plt.plot(acc_list_val)
    plt.show()
    