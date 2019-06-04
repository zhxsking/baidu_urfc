# -*- coding: utf-8 -*-

import torch
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
from tqdm import tqdm

from preprocess import imgProc
from cnn import mResNet, mDenseNet
from urfc_option import Option


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载模型
    net = mResNet18().to(opt.device)
    state = torch.load(r"checkpoint\best-cnn-ori.pkl", map_location=opt.device)
    net.load_state_dict(state['net'])
    
    # 加载数据
    print('Loading Data...')
    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
    imgs_test = imgProc(imgs_test)
    visits_test = torch.FloatTensor(visits_test.transpose(0,3,1,2))
    
    dataloader_test = DataLoader(dataset=TensorDataset(imgs_test, visits_test),
                                batch_size=opt.batchsize, num_workers=opt.workers)
    
    # 分batch进行预测
    net.eval()
    out_lab = []
    with torch.no_grad():
        for (img, visit) in tqdm(dataloader_test):
            img = img.to(opt.device)
            visit = visit.to(opt.device)
            out, _ = net(img, visit)
            _, preds = torch.max(out, 1)
            out_lab.append(preds.cpu().numpy().flatten().astype(np.uint8) + 1)
    
    # 输出预测文件
    f = open(r"data/out-label.txt", "w+")
    cnt = 0
    out_lab_np = []
    for j in range(len(out_lab)):
        for i in range(len(out_lab[j])):
            out_lab_np.append(out_lab[j][i])
            f.write("{} \t {}\n".format(str(cnt).zfill(6), str(out_lab[j][i]).zfill(3)))
            cnt += 1
    f.close()
    out_lab_np = np.array(out_lab_np)
    



