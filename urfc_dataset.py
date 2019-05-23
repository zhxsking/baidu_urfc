# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from urfc_option import Option


class UrfcDataset(Dataset):
    def __init__(self, dir_img, dir_visit, path_txt):
        super().__init__()
        self.dir_img = dir_img
        self.dir_visit = dir_visit
        
        data_list = list(pd.read_table(path_txt, header=None)[0])
        self.data_names = [a.split('\\')[-1] for a in data_list]
        
    def __getitem__(self, index):
        label_str = self.data_names[index][7:10]
        
        img = Image.open(join(self.dir_img, label_str, self.data_names[index] + ".jpg"))
        visit = np.load(join(self.dir_visit, self.data_names[index] + ".npy"))
        

        # 标准化
        means = (0.46832234, 0.53796417, 0.6216422)
        stds =(0.1810789, 0.16477963, 0.14735216)
        img_process = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
                ])
        img = img_process(img)
        
#        img = transforms.ToTensor()(img)
        visit = transforms.ToTensor()(visit)
        
        return img, visit, int(label_str)-1
    
    def __len__(self):
        return len(self.data_names)


if __name__ == '__main__':
    __spec__ = None

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    opt = Option()
    dataset = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    
    for cnt, (img, visit, lab) in enumerate(dataloader, 1):
        print(lab)
        img_show = img.detach().numpy()[0][0]
        visit_show = visit.detach().numpy()[0][0]
        
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_show, cmap='gray')
        plt.subplot(122)
        plt.imshow(visit_show, cmap='gray')
        plt.show()
        break
