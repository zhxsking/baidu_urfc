# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from multimodal import MultiModalNet

from urfc_utils import Logger, imgProc, aug_batch, aug_val_batch, get_tta_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet50, mSDNet101, mPNASNet, MMNet
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
            img_h, img_v = get_tta_batch(img_o)
            
            img_o = img_o.to(device)
            img_h = img_h.to(device)
            img_v = img_v.to(device)
            visit = visit.to(device)
            
            out_o, _ = net(img_o, visit)
            out_h, _ = net(img_h, visit)
            out_v, _ = net(img_v, visit)
            out = out_o * 2 + out_h + out_v

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
    net = mSDNet50().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-50.pkl", map_location=opt.device)
    net.load_state_dict(state['net'])
    
    net1 = mPNASNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-pnasnet.pkl", map_location=opt.device)
    net1.load_state_dict(state['net'])
    
    net2 = mSDNet101().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-101.pkl", map_location=opt.device)
    net2.load_state_dict(state['net'])
    
    net3 = MMNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-mmnet.pkl", map_location=opt.device)
    net3.load_state_dict(state['net'])
    
#    net3 = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(opt.device)
#    state = torch.load(r"checkpoint\multimodal_fold_0_model_best_loss.pth.tar", map_location=opt.device)
#    net3.load_state_dict(state['state_dict'])
    
    # 加载数据
    print('Loading Data...')
    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
    imgs_test = imgProc(imgs_test)
    visits_test = torch.FloatTensor(visits_test.transpose(0,3,1,2))
    
    dataloader_test = DataLoader(dataset=TensorDataset(imgs_test, visits_test),
                                batch_size=opt.batchsize, num_workers=opt.workers)
    
    device = opt.device
    net.eval()
    net1.eval()
    net2.eval()
    net3.eval()
    
    sm = nn.Softmax()
    
    out_lab = []
    with torch.no_grad():
        for (img_o, visit) in tqdm(dataloader_test):
            img_h, img_v = get_tta_batch(img_o)
            
            img_o = img_o.to(device)
            img_h = img_h.to(device)
            img_v = img_v.to(device)
            visit = visit.to(device)
            
            out_o, _ = net(img_o, visit)
            out_h, _ = net(img_h, visit)
            out_v, _ = net(img_v, visit)
            
            out_o1, _ = net1(img_o, visit)
            out_h1, _ = net1(img_h, visit)
            out_v1, _ = net1(img_v, visit)
            
            out_o2, _ = net2(img_o, visit)
            out_h2, _ = net2(img_h, visit)
            out_v2, _ = net2(img_v, visit)
            
            out_o3, _ = net3(img_o, visit)
            out_h3, _ = net3(img_h, visit)
            out_v3, _ = net3(img_v, visit)
            
#            img_o_ = img_o.clone()
#            img_o_[:,0,:,:] = img_o[:,2,:,:]
#            img_o_[:,2,:,:] = img_o[:,0,:,:]
#            img_h_ = img_h.clone()
#            img_h_[:,0,:,:] = img_h[:,2,:,:]
#            img_h_[:,2,:,:] = img_h[:,0,:,:]
#            img_v_ = img_v.clone()
#            img_v_[:,0,:,:] = img_v[:,2,:,:]
#            img_v_[:,2,:,:] = img_v[:,0,:,:]
#            
#            out_o3 = net3(img_o_, visit)
#            out_h3 = net3(img_h_, visit)
#            out_v3 = net3(img_v_, visit) 
            
#            out = (sm(out_o3) * 2 + sm(out_h3) + sm(out_v3))
            
            out = (sm(out_o) * 2 + sm(out_h) + sm(out_v) + 
                   sm(out_o1) * 2 + sm(out_h1) + sm(out_v1) + 
                   sm(out_o2) * 2 + sm(out_h2) + sm(out_v2) +
                   sm(out_o3) * 2 + sm(out_h3) + sm(out_v3))

            _, preds = torch.max(out, 1)
            out_lab.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
    
    out_lab_np = []
    for j in range(len(out_lab)):
        for i in range(len(out_lab[j])):
            out_lab_np.append(out_lab[j][i])
    out_lab_np = np.array(out_lab_np)
    
#    out_lab_np = predict(net, dataloader_test, opt.device)
#    out_lab_np = predict_TTA(net, dataloader_test, opt.device)
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    for i in range(10000):
        f.write("{} \t {}\n".format(str(i).zfill(6), str(out_lab_np[i]).zfill(3)))
    f.close()
    



