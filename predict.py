# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
from tqdm import tqdm
from boxx import g
from catboost import CatBoostClassifier

from multimodal import MultiModalNet

from urfc_dataset import UrfcDataset
from urfc_utils import Logger, imgProc, aug_batch, aug_val_batch, get_tta_batch
from cnn import mResNet18, mResNet, mDenseNet, mSENet, mDPN26, mSDNet50, mSDNet50_p, mSDNet101, mPNASNet, MMNet
from urfc_option import opt


def predict(dataloader_test, device, *nets):
    """预测输出"""
    sm = nn.Softmax(dim=1)
    labs_out = []
    out_mat = []
    with torch.no_grad():
        for (img, visit, _) in tqdm(dataloader_test):
            if isinstance(img, list):
                img_tta = img
            else:
                img_tta = (img,)
            
            visit = visit.to(device)
            
            out_mat_h = []
            for cnt, net in enumerate(nets):
                net.eval()
                for i in range(len(img_tta)):
                    out_tta_tmp = net(img_tta[i].to(device), visit)
                    
                    if isinstance(out_tta_tmp, tuple):
                        out_tta_tmp = out_tta_tmp[0]
                    
#                    out_tta_tmp = out_tta_tmp + 2*torch.mul(out_tta_tmp, torch.le(out_tta_tmp,0).float())
                    
                    out_tta_tmp = sm(out_tta_tmp)
                    
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
            
            _, preds = torch.max(out, 1)
            labs_out.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
            out_mat.append(np.array(out_mat_h).transpose((1,0,2)).reshape((visit.shape[0], -1)))
    labs_out_np = []
    for j in range(len(labs_out)):
        for i in range(len(labs_out[j])):
            labs_out_np.append(labs_out[j][i])            
    labs_out_np = np.array(labs_out_np)
    fea = out_mat[0]
    for i in range(1, len(out_mat)):
        tmp = out_mat[i]
        fea = np.r_[fea, tmp]
    g()
    return labs_out_np, fea
    

if __name__ == '__main__':
    __spec__ = None
    
    # 加载模型
    print('Loading Model...')
    net1 = mSENet().to(opt.device) # 2tta6652 4tta6666 7tta6664
    state = torch.load(r"checkpoint\best-cnn-senet-6617.pkl", map_location=opt.device) # semi实测0.66134 no tta
    net1.load_state_dict(state['net'])
    
    net2 = mSENet().to(opt.device) # 2tta6397 4tta6406 7tta6408
    state = torch.load(r"checkpoint\best-cnn-senet-6386.pkl", map_location=opt.device)
    net2.load_state_dict(state['net'])
    
    net3 = mResNet().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-resnet-26-7-24-6470.pkl", map_location=opt.device)
    net3.load_state_dict(state['net'])
    
    net4 = MMNet().to(opt.device) # 4tta6811
    state = torch.load(r"checkpoint\best-cnn-mmnet-6774.pkl", map_location=opt.device)
    net4.load_state_dict(state['net'])
    
    # 加载数据
    print('Loading Data...')  
    dataset_test = UrfcDataset(opt.dir_img_test, opt.dir_visit_npy_test, 
                               "data/test.txt", aug=False, mode='test', tta=opt.use_tta)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1024,
                                shuffle=False, num_workers=opt.workers, pin_memory=True)
    
    # 预测
#    nets = [net1]
    nets = [net1, net3, net4]
#    nets = [net2, net3, net8, netm1]
#     nets = [net2, net3, net4, net6, net8, netm, netm1]
    out_lab_np, fea = predict(dataloader_test, opt.device, *nets)
    
    #%%
    if opt.use_blend:
        # blending
        model = CatBoostClassifier()
        model.load_model(r"checkpoint\catboost_model-150.dump")
        y_pred = (model.predict(fea)).astype(np.uint8).squeeze() + 1
        res = y_pred
    else:
        res = out_lab_np
    
    print('29981 22833 12971 1717 4099 15438 5429 3266 4266')
    for i in range(1,10):
        print(np.sum(res==i), end=' ')
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    for i in range(len(dataset_test)):
        f.write("{} \t {}\n".format(str(i).zfill(6), str(res[i]).zfill(3)))
    f.close()
    
    torch.cuda.empty_cache()


