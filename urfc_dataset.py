# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from imgaug import augmenters as iaa

from urfc_option import Option


class UrfcDataset(Dataset):
    def __init__(self, dir_img, dir_visit, path_txt):
        super().__init__()
        self.dir_img = dir_img
        self.dir_visit = dir_visit
        
        data_list = list(pd.read_csv(path_txt, header=None)[0])
        self.data_names = [a.split('\\')[-1] for a in data_list]
        
    def __getitem__(self, index):
        label_str = self.data_names[index][7:10]
        
        img = Image.open(join(self.dir_img, label_str, self.data_names[index] + ".jpg"))
        visit = np.load(join(self.dir_visit, self.data_names[index] + ".npy"))
        
        img = self.augumentor(img)
        
#        def subMean(x):
#            '''每张图减去均值，匀光'''
#            for i in range(3):
#                x[i,:,:] -= x[i,:,:].mean()
#            return x
        
        # 标准化
#        means = (0.46832234, 0.53796417, 0.6216422)
#        stds =(0.1810789, 0.16477963, 0.14735216)
#        means = (-1.3326176e-09, -5.8395827e-10, -1.153197e-10)
#        stds =(0.11115803, 0.09930103, 0.08884794)
        img_process = transforms.Compose([
                transforms.ToTensor(),
#                transforms.Lambda(subMean),
#                transforms.Normalize(means, stds),
                ])
        img = img_process(img)
        visit = transforms.ToTensor()(visit)
        
        return img, visit, int(label_str)-1
    
    def __len__(self):
        return len(self.data_names)
    
    def augumentor(self,image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

if __name__ == '__main__':
    __spec__ = None

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    opt = Option()
    dataset = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/val.txt")
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    
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
