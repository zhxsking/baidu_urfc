# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
from tqdm import tqdm
from boxx import g

from multimodal import MultiModalNet

from urfc_dataset import UrfcDataset
from urfc_utils import Logger, Record, imgProc, aug_batch, aug_val_batch, get_tta_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet50, mSDNet50_p, mSDNet101, mPNASNet, MMNet
from urfc_option import Option


def plotConfusionMatrix(cm, normalize=False, classes=None, cmap=plt.cm.Blues):
    '''绘制混淆矩阵'''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    if (classes != None):
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)
    thresh = cm.max() / 2.
    fmt = '.2f' if normalize else 'd'
    iters = np.reshape([[[i,j] for j in range(9)] for i in range(9)],(cm.size,2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black") # 显示对应的数字
    plt.ylabel('GT')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

def eval_net(loss_func, dataloader_val, device, *nets):
    """用验证集评判网络性能"""
    sm = nn.Softmax(dim=1)
    acc_temp = Record()
    loss_temp = Record()
    labs_ori, labs_out = [], []
    out_mat = []
    with torch.no_grad():
        for (img, visit, out_gt) in tqdm(dataloader_val):
            if isinstance(img, list):
                img_tta = img
            else:
                img_tta = (img,)

            visit = visit.to(device)
            out_gt = out_gt.to(device)
            
            out_mat_h = []
            for cnt, net in enumerate(nets):
                net.eval()
                
                for i in range(len(img_tta)):
                    out_tta_tmp = net(img_tta[i].to(device), visit)
                    
                    if isinstance(out_tta_tmp, tuple):
                        out_tta_tmp = out_tta_tmp[0]
                    
                    out_tta_tmp = sm(out_tta_tmp)
                    
                    # 投票法，每行最大值赋1，其他为0
#                    mat_tmp = torch.max(out_tta_tmp, 1)[0].repeat(out_tta_tmp.shape[1],1).transpose(1,0)
#                    out_tta_tmp = (mat_tmp == out_tta_tmp).float()
                    
                    out_mat_h.append(out_tta_tmp.cpu().numpy())
                    
                    if (i==0):
                        out_tta = out_tta_tmp
                    else:
                        out_tta = out_tta + out_tta_tmp
                
                out_tmp = out_tta
                
                if (cnt==0):
                    out = out_tmp
                else:
                    out = out + out_tmp
            
            loss = loss_func(out, out_gt)
            _, preds = torch.max(out, 1)
            
            loss_temp.update(loss.item(), img_tta[0].shape[0])
            acc_temp.update((float(torch.sum(preds == out_gt.data)) / len(out_gt)), len(out_gt))
            labs_ori.append(out_gt.cpu().numpy())
            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8))
            tt = np.array(out_mat_h).transpose((1,0,2)).reshape((len(out_gt), -1))
            out_mat.append(tt)
    labs_ori_np = []
    labs_out_np = []
    for j in range(len(labs_ori)):
        for i in range(len(labs_ori[j])):
            labs_ori_np.append(labs_ori[j][i])
            labs_out_np.append(labs_out[j][i])            
    labs_ori_np = np.array(labs_ori_np)
    labs_out_np = np.array(labs_out_np)
    g()
    return loss_temp.avg, acc_temp.avg, labs_ori_np, labs_out_np, out_mat


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    dataset_val = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt", aug=False, tta=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1024,
                                shuffle=False, num_workers=opt.workers, pin_memory=True)
    
    # 加载模型
    print('Loading Model...')
    loss_func = nn.CrossEntropyLoss().to(opt.device)
    
    net1 = mSENet().to(opt.device) # 2tta6652 4tta6666 7tta6664
    state = torch.load(r"checkpoint\best-cnn-senet-6617.pkl", map_location=opt.device)
    net1.load_state_dict(state['net'])
    
    net2 = mSENet().to(opt.device) # 2tta6397 4tta6406 7tta6408
    state = torch.load(r"checkpoint\best-cnn-senet-6386.pkl", map_location=opt.device)
    net2.load_state_dict(state['net'])
    
    
    #%% 验证原始数据
    nets = [net1]
#    nets = [net1, net2]
    loss, acc, labs_ori_np, labs_out_np, out_mat = eval_net(loss_func, dataloader_val, opt.device, *nets)
    
    #%% 绘制混淆矩阵, 计算acc
    cm = metrics.confusion_matrix(labs_ori_np, labs_out_np)
    acc_all_val = metrics.accuracy_score(labs_ori_np, labs_out_np)
    class_names = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
    plotConfusionMatrix(cm, normalize=True, classes = class_names)
    print('val acc: {:.4f}, loss: {:.4f}'.format(acc, loss))
    print('val acc all {:.4f}'.format(acc_all_val))
    
    #%%     
#    t = out_mat[0]
#    for i in range(1, len(out_mat)):
#        tmp = out_mat[i]
#        t = np.r_[t, tmp]
#    
#    
#    from catboost import CatBoostClassifier, Pool
#
#    model = CatBoostClassifier(
#            learning_rate = 0.2,
#            iterations = 1000,
#            eval_metric = 'Accuracy',
#            random_seed = 42,
#            logging_level = 'Verbose',
#            use_best_model = True,
#            task_type = 'GPU',
#            )
#    model.fit(t, labs_ori_np)
#    model.save_model(r"checkpoint\catboost_model.dump")
##    model = CatBoostClassifier()
##    model.load_model(r"checkpoint\catboost_model.dump")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    