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

def predict(dataloader_test, device, *nets):
    """预测输出"""
    sm = nn.Softmax(dim=1)
    labs_out = []
    with torch.no_grad():
        for (img, visit) in tqdm(dataloader_test):
            img = img.to(device)
            visit = visit.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                out_tmp, _ = net(img, visit)
                out_tmp = sm(out_tmp)

                if (cnt==0):
                    out = out_tmp
                else:
                    out = out + out_tmp
            
            _, preds = torch.max(out, 1)
            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
    labs_out_np = []
    for j in range(len(labs_out)):
        for i in range(len(labs_out[j])):
            labs_out_np.append(labs_out[j][i])            
    labs_out_np = np.array(labs_out_np)
    return labs_out_np

def predict_TTA(dataloader_test, device, *nets):
    """预测输出，TTA"""
    sm = nn.Softmax(dim=1)
    labs_out = []
    with torch.no_grad():
        for (img_o, visit) in tqdm(dataloader_test):
            img_tta = get_tta_batch(img_o) # 得到tta之后的图像
            
            img_o = img_o.to(device)
            visit = visit.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                out_o, _ = net(img_o, visit)
                out_o = sm(out_o)
                
                for i in range(len(img_tta)):
                    out_tta_tmp, _ = net(img_tta[i].to(device), visit)
                    out_tta_tmp = sm(out_tta_tmp)
                    if (i==0):
                        out_tta = out_tta_tmp
                    else:
                        out_tta = out_tta + out_tta_tmp
                
                out_tmp = out_o * i + out_tta

                if (cnt==0):
                    out = out_tmp
                else:
                    out = out + out_tmp
            
            _, preds = torch.max(out, 1)
            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
    labs_out_np = []
    for j in range(len(labs_out)):
        for i in range(len(labs_out[j])):
            labs_out_np.append(labs_out[j][i])            
    labs_out_np = np.array(labs_out_np)
    return labs_out_np
    

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
    
    net4 = MMNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-mmnet-实测6464.pkl", map_location=opt.device)
    net4.load_state_dict(state['net'])
    
    net5 = mSDNet50().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-sdnet-50-实测6264.pkl", map_location=opt.device)
    net5.load_state_dict(state['net'])
    
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
    
    # 预测
#    nets = [net]
#    nets = [net, net2, net3]
    nets = [net, net2, net3, net4, net5]
#    out_lab_np = predict(dataloader_test, opt.device, *nets)
    out_lab_np = predict_TTA(dataloader_test, opt.device, *nets)
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    for i in range(10000):
        f.write("{} \t {}\n".format(str(i).zfill(6), str(out_lab_np[i]).zfill(3)))
    f.close()
    



