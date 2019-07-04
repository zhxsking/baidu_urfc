# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

from multimodal import MultiModalNet

from urfc_dataset import UrfcDataset
from urfc_utils import Logger, imgProc, aug_batch, aug_val_batch, get_tta_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet50, mSDNet50_p, mSDNet101, mPNASNet, MMNet
from urfc_option import Option

def predict(dataloader_test, device, *nets):
    """预测输出"""
    sm = nn.Softmax(dim=1)
    labs_out = []
    with torch.no_grad():
        for (img, visit, _) in tqdm(dataloader_test):
            img = img.to(device)
            visit = visit.to(device)
            
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
        for (img_o, visit, _) in tqdm(dataloader_test):
            img_tta = get_tta_batch(img_o) # 得到tta之后的图像
            
            img_o = img_o.to(device)
            visit = visit.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                out_o = net(img_o, visit)
                
                if isinstance(out_o, tuple):
                    out_o = out_o[0]
                
#                out_o = out_o + 2*torch.mul(out_o, torch.le(out_o,0).float()) # 负的更负
                
                out_o = sm(out_o)
                
                for i in range(len(img_tta)):
                    out_tta_tmp = net(img_tta[i].to(device), visit)
                    
                    if isinstance(out_tta_tmp, tuple):
                        out_tta_tmp = out_tta_tmp[0]
                    
#                    out_tta_tmp = out_tta_tmp + 2*torch.mul(out_tta_tmp, torch.le(out_tta_tmp,0).float())
                    
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
    
#    net1 = mSDNet50().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-50.pkl", map_location=opt.device) # 实测0.6394
#    net1.load_state_dict(state['net'])
    
#    net2 = mSDNet101().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-101.pkl", map_location=opt.device) # 实测0.6526
#    net2.load_state_dict(state['net'])
#    
#    net3 = MMNet().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-mmnet.pkl", map_location=opt.device) # 实测0.6529
#    net3.load_state_dict(state['net'])
    
#    net4 = MMNet().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-mmnet-实测6464.pkl", map_location=opt.device)
#    net4.load_state_dict(state['net'])
    
#    net5 = mSDNet50().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-50-实测6264.pkl", map_location=opt.device)
#    net5.load_state_dict(state['net'])
    
#    net6 = mSDNet101().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-101-64.pkl", map_location=opt.device) # 实测0.6379
#    net6.load_state_dict(state['net'])
    
#    net7 = mSDNet50().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-50-59.pkl", map_location=opt.device) # 实测0.6394
#    net7.load_state_dict(state['net'])
    
#    net8 = MMNet().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-mmnet-59.pkl", map_location=opt.device) # 实测0.6531
#    net8.load_state_dict(state['net'])
    
#    net9 = mSDNet101().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-101-63.pkl", map_location=opt.device) # tta更低 实测0.6305
#    net9.load_state_dict(state['net'])
    
#    net10 = mSDNet50_p().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-sdnet-50-p.pkl", map_location=opt.device) # 实测0.6066
#    net10.load_state_dict(state['net'])
    
#    net11 = MMNet().to(opt.device)
#    state = torch.load(r"checkpoint\best-cnn-mmnet-6630.pkl", map_location=opt.device)
#    net11.load_state_dict(state['net'])
    
    net12 = mSENet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-senet-6617.pkl", map_location=opt.device) # semi实测0.66134
    net12.load_state_dict(state['net'])
    
#    netm = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(opt.device) # 实测0.6447
#    state = torch.load(r"checkpoint\multimodal_fold_0_model_best_loss.pth.tar", map_location=opt.device)
#    netm.load_state_dict(state['state_dict'])
    
#    netm1 = MultiModalNet("se_resnext101_32x4d","dpn26",0.5).to(opt.device) # 实测0.6688
#    state = torch.load(r"checkpoint\best-cnn-mutimodel.pkl", map_location=opt.device)
#    netm1.load_state_dict(state['net'])
    
    
    # 加载数据
    print('Loading Data...')
#    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
#    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
#    imgs_test = imgProc(imgs_test, opt.means, opt.stds)
#    visits_test = torch.FloatTensor(visits_test.transpose(0,3,1,2))
#    
#    dataloader_test = DataLoader(dataset=TensorDataset(imgs_test, visits_test),
#                                batch_size=512, num_workers=opt.workers)
    
    dataset_val = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt", aug=False)
    
    dataset_test = UrfcDataset(opt.dir_img_test, opt.dir_visit_npy_test, 
                               "data/test.txt", aug=False, mode='test')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1024,
                                shuffle=False, num_workers=opt.workers, pin_memory=True)
    
    # 预测
    nets = [net12]
#    nets = [net, net2, net3]
#    nets = [net2, net3, net8, netm1]
#     nets = [net2, net3, net4, net6, net8, netm, netm1]
    out_lab_np = predict(dataloader_test, opt.device, *nets)
#    out_lab_np = predict_TTA(dataloader_test, opt.device, *nets)
    
    for i in range(1,10):
        print(np.sum(out_lab_np==i), end=' ')
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    for i in range(100000):
        f.write("{} \t {}\n".format(str(i).zfill(6), str(out_lab_np[i]).zfill(3)))
    f.close()
    



