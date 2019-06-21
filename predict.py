# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from urfc_utils import Logger, imgProc, aug_batch, aug_val_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet
from urfc_option import Option

def predict(net, dataloader_test, device):
    """用验证集评判网络性能"""
    net.eval()
    out_lab = []
    with torch.no_grad():
        for (img, visit) in tqdm(dataloader_test):
            img = img.to(device)
            visit = visit.to(device)
            out, _ = net(img, visit)
            _, preds = torch.max(out, 1)
            out_lab.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
    
    out_lab_np = []
    for j in range(len(out_lab)):
        for i in range(len(out_lab[j])):
            out_lab_np.append(out_lab[j][i])
    out_lab_np = np.array(out_lab_np)
    
    return out_lab_np

def predict_TTA(net, dataloader_test, device):
    """用验证集评判网络性能"""
    net.eval()
    out_lab = []
    with torch.no_grad():
        for (img_o, visit) in tqdm(dataloader_test):
            img_o = img_o.to(device)
            visit = visit.to(device)
            
            img_h = transforms.RandomHorizontalFlip(1)(img_o)
            img_v = transforms.RandomVerticalFlip(1)(img_o)
            
            out_o, _ = net(img_o, visit)
            out_h, _ = net(img_h, visit)
            out_v, _ = net(img_v, visit)
            out = out_o + out_h + out_v

            _, preds = torch.max(out, 1)
            out_lab.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
    
    out_lab_np = []
    for j in range(len(out_lab)):
        for i in range(len(out_lab[j])):
            out_lab_np.append(out_lab[j][i])
    out_lab_np = np.array(out_lab_np)
    
    return out_lab_np
    

if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载模型
    print('Loading Model...')
    net = mSDNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn.pkl", map_location=opt.device)
    net.load_state_dict(state['net'])
    
    # 加载数据
    print('Loading Data...')
    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
    imgs_test = imgProc(imgs_test)
    visits_test = torch.FloatTensor(visits_test.transpose(0,3,1,2))
    
    dataloader_test = DataLoader(dataset=TensorDataset(imgs_test, visits_test),
                                batch_size=opt.batchsize, num_workers=opt.workers)
    
    out_lab_np = predict(net, dataloader_test, opt.device)
#    out_lab_np = predict_TTA(net, dataloader_test, opt.device)
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    for i in range(10000):
        f.write("{} \t {}\n".format(str(i).zfill(6), str(out_lab_np[i]).zfill(3)))
    f.close()
    



