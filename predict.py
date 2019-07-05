# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
from tqdm import tqdm
from boxx import g

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
            if isinstance(img, list):
                img_tta = img
            else:
                img_tta = (img,)
            
            visit = visit.to(device)
            
            for cnt, net in enumerate(nets):
                net.eval()
                for i in range(len(img_tta)):
                    out_tta_tmp = net(img_tta[i].to(device), visit)
                    
                    if isinstance(out_tta_tmp, tuple):
                        out_tta_tmp = out_tta_tmp[0]
                    
#                    out_tta_tmp = out_tta_tmp + 2*torch.mul(out_tta_tmp, torch.le(out_tta_tmp,0).float())
                    
                    out_tta_tmp = sm(out_tta_tmp)
                    
                    if (i==0):
                        out_tta = out_tta_tmp
                        out_tta_o = out_tta_tmp
                    else:
                        out_tta = out_tta + out_tta_tmp
                
                out_tmp = out_tta

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
    g()
    return labs_out_np
    

if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载模型
    print('Loading Model...')
    net1 = mSENet().to(opt.device) # 2tta6652 4tta6666 7tta6664
    state = torch.load(r"checkpoint\best-cnn-senet-6617.pkl", map_location=opt.device) # semi实测0.66134 no tta
    net1.load_state_dict(state['net'])
    
    net2 = mSENet().to(opt.device) # 2tta6397 4tta6406 7tta6408
    state = torch.load(r"checkpoint\best-cnn-senet-6386.pkl", map_location=opt.device)
    net2.load_state_dict(state['net'])
    
    # 加载数据
    print('Loading Data...')  
    dataset_test = UrfcDataset(opt.dir_img_test, opt.dir_visit_npy_test, 
                               "data/test.txt", aug=False, mode='test',
                               tta=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1024,
                                shuffle=False, num_workers=opt.workers, pin_memory=True)
    
    # 预测
#    nets = [net1]
    nets = [net1, net2]
#    nets = [net2, net3, net8, netm1]
#     nets = [net2, net3, net4, net6, net8, netm, netm1]
    out_lab_np = predict(dataloader_test, opt.device, *nets)
    
    print('29981 22833 12971 1717 4099 15438 5429 3266 4266')
    for i in range(1,10):
        print(np.sum(out_lab_np==i), end=' ')
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    for i in range(len(dataset_test)):
        f.write("{} \t {}\n".format(str(i).zfill(6), str(out_lab_np[i]).zfill(3)))
    f.close()
    



