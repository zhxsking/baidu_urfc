# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from imgaug import augmenters as iaa

from urfc_option import Option
from linear_p import linear_p


class UrfcDataset(Dataset):
    def __init__(self, dir_img, dir_visit, path_txt, aug=True, mode='train', tta=False):
        super().__init__()
        self.dir_img = dir_img
        self.dir_visit = dir_visit
        self.aug = aug
        self.mode = mode
        self.tta = tta
        self.data_list = list(pd.read_csv(path_txt, header=None)[0])
        
        # 定义操作表
        self.tfs_lib = {
            'h': transforms.RandomHorizontalFlip(1), # 水平翻转
            'v': transforms.RandomVerticalFlip(1), # 上下翻转
            'n90': transforms.RandomRotation((-90,-90)), # 顺时针旋转90度
            'p90': transforms.RandomRotation((90,90)), # 逆时针旋转90度
        }
        
    def __getitem__(self, index):
        img = plt.imread(join(self.data_list[index] + ".jpg"))
        visit = np.load(join(self.dir_visit, self.data_list[index].split('\\')[-1] + ".npy"))
        
        if self.mode == 'train':
            label_str = self.data_list[index].split('\\')[-1][7:10]
            lab = int(label_str)-1
        else:
            lab = self.data_list[index].split('\\')[-1]
        
        if self.aug:
            img = self.augumentor(img)
        
#        def subMean(x):
#            '''每张图减去均值，匀光'''
#            for i in range(3):
#                x[i,:,:] -= x[i,:,:].mean()
#            return x
        
        # 标准化
#        means = (0.4553296, 0.52905685, 0.6211398)
#        stds =(0.14767659, 0.13119018, 0.1192783)
#        img_process = transforms.Compose([
#                transforms.ToPILImage(),
#                transforms.ToTensor(),
##                transforms.Normalize(means, stds),
##                transforms.Lambda(subMean),
#                ])
#        img = img_process(img)
        
        if self.tta:
            img_o = self.transform(())(img)
            img_h = self.transform(('h'))(img)
            img_v = self.transform(('v'))(img)
            img = (img_o, img_h, img_v)
        else:
            img = self.transform(())(img)
            
        visit = transforms.ToTensor()(visit)
        
        return img, visit, lab
    
    def __len__(self):
        return len(self.data_list)
    
    def transform(self, tfs_str):
        tfs = []
        tfs.append(transforms.ToPILImage())
        for tf in tfs_str:
            tfs.append(self.tfs_lib[tf])
        tfs.append(transforms.ToTensor())
        return transforms.Compose(tfs)
    
    def augumentor(self,image):
#        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
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
#            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#            
#            iaa.Fliplr(0.5),
#            iaa.Flipud(0.5),
#            sometimes(iaa.Affine(
#                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#                rotate=(-15, 15),
#                shear=(-16, 16),
#                order=[0, 1],
#            )),
#            iaa.SomeOf((0, 4), [
#                iaa.OneOf([
#                    iaa.GaussianBlur((0, 2)),
#                    iaa.AverageBlur(k=(2, 5)),
#                    iaa.MedianBlur(k=(3, 5)),
#                ]),
#                iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),
#                sometimes(iaa.OneOf([
#                    iaa.EdgeDetect(alpha=(0, 0.7)),
#                    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
#                ])),
#                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#                iaa.Dropout((0.01, 0.1), per_channel=0.5),
#                iaa.OneOf([
#                    iaa.Fog(),
#                    iaa.Clouds(),
#                ]),
#                iaa.Add((-10, 10), per_channel=0.5),
#                iaa.Multiply((0.7, 1.3), per_channel=0.5),
#            ], random_order=True)
            
            ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

if __name__ == '__main__':
    __spec__ = None

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    opt = Option()
    dataset = UrfcDataset(opt.dir_img_test, opt.dir_visit_npy_test, "data/test.txt", aug=False, mode='test', tta=False)
#    dataset = UrfcDataset(opt.dir_img, opt.dir_visit_npy, "data/train-over.txt", aug=True)
    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)
    
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
