# -*- coding: utf-8 -*-

import numpy as np
from os.path import join
import os
import matplotlib.pyplot as plt
import cv2
from boxx import show
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import time
from imgaug import augmenters as iaa

from dehaze import deHaze
from linear_p import linear_p
from urfc_option import Option
from urfc_utils import imgProc


#def histq(img):
#    (r, g, b) = cv2.split(img)
#    
#    bH = cv2.equalizeHist(b)
#    gH = cv2.equalizeHist(g)
#    rH = cv2.equalizeHist(r)
#    # 合并每一个通道
#    result = cv2.merge((rH, gH, bH))
#    return result
#
#def func_images(img, random_state, parents, hooks):
#    img = (linear_p(img, 0.02) * 255).astype(np.uint8)
#    return img
#
#def aug_img(img):
#    '''对uint8图像或图像的一个batch（NHWC）进行aug'''
#    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#    aug_seq = iaa.Sequential([
#        iaa.Lambda(func_images=func_images),
#        iaa.Fliplr(0.5),
#        iaa.Flipud(0.5),
#        sometimes(iaa.Affine(
#            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#            rotate=(-15, 15),
#            shear=(-16, 16),
#            order=[0, 1],
#        )),
#        iaa.SomeOf((0, 5), [
#            iaa.OneOf([
#                iaa.GaussianBlur((0, 2)),
#                iaa.AverageBlur(k=(2, 5)),
#                iaa.MedianBlur(k=(3, 5)),
#            ]),
#            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),
#            sometimes(iaa.OneOf([
#                iaa.EdgeDetect(alpha=(0, 0.7)),
#                iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
#            ])),
#            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#            iaa.OneOf([
#                iaa.Dropout((0.01, 0.1), per_channel=0.5),
##                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),per_channel=0.2),
#            ]),
#            iaa.OneOf([
#                iaa.Fog(),
#                iaa.Clouds(),
#            ]),
##            iaa.Invert(0.05, per_channel=True),
#            iaa.Add((-10, 10), per_channel=0.5),
#            iaa.Multiply((0.7, 1.3), per_channel=0.5),
##            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
#        ], random_order=True)
#    ], random_order=True)
#    
#    if (img.ndim == 3):
#        # 对单幅图像进行aug
#        img_aug = aug_seq.augment_image(img)
#    elif (img.ndim == 4):
#        # 对一个batch进行aug
#        img_aug = aug_seq.augment_images(img)
#    else:
#        img_aug = []
#    return img_aug
#
#def aug_batch(batch):
#    '''对torch的一个batch（NCHW）进行aug'''
#    batch = (batch.permute(0,2,3,1).numpy()*255).astype(np.uint8)
#    batch_aug = aug_img(batch)
#    batch_aug = (batch_aug / 255.0).astype(np.float32).transpose(0,3,1,2)
#    batch_aug = torch.as_tensor(batch_aug, dtype=torch.float32)
#    return batch_aug

since = time.time() # 记录时间

###############################################################################
# 读取数据
#opt = Option()
#dirs = sorted(os.listdir(opt.dir_img))
#files = {}
#for dir in dirs:
#    path = join(opt.dir_img, dir)
#    files[int(dir)] = []
#    for file in os.listdir(path):
#        files[int(dir)].append(join(path, file))
#
## 匀光测试
#pic_num = 10
#pic_type = 1
#res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
#for i in range(pic_num):
#    for j in range(pic_num):
#        img = plt.imread(files[pic_type][pic_num*i+j]) / 255
#        
##        img = histq(img)
##        img = deHaze(img)
##        img -= np.mean(img)
##        img[:,:,0] -= np.mean(img[:,:,0])
##        img[:,:,1] -= np.mean(img[:,:,1])
##        img[:,:,2] -= np.mean(img[:,:,2])
##        img = deHaze(img)
#        img = linear_p(img, 0.02)
##        img[:,:,0] -= np.mean(img[:,:,0])
##        img[:,:,1] -= np.mean(img[:,:,1])
##        img[:,:,2] -= np.mean(img[:,:,2])
##        img = deHaze(img)
##        img = histq((img*255).astype(np.uint8))
##        r = (img[:,:,0]).copy()
##        g = (img[:,:,1]).copy()
##        b = (img[:,:,2]).copy()
##        r -= np.mean(r)
##        g -= np.mean(g)
##        b -= np.mean(b)
##        res[i*100:(i+1)*100, j*100:(j+1)*100, 0] = r
##        res[i*100:(i+1)*100, j*100:(j+1)*100, 1] = g
##        res[i*100:(i+1)*100, j*100:(j+1)*100, 2] = b
#        res[i*100:(i+1)*100, j*100:(j+1)*100, :] = img
##res = (res - np.mean(res)) / (np.std(res))
#res = (res - np.min(res)) / (np.max(res) - np.min(res))
##plt.subplot(121)
#plt.imshow(res)
#plt.imsave(r"data/test.jpg", res)
       
###############################################################################
# 查看测试数据
#opt = Option()
#files = sorted(os.listdir(opt.dir_img_test))
#
##pic_num = 30
#res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
#for i in range(pic_num):
#    for j in range(pic_num):
#        img = plt.imread(join(opt.dir_img_test, files[pic_num*i+j]))
#        img = histq(img) / 255
#        r = (img[:,:,0]).copy()
#        g = (img[:,:,1]).copy()
#        b = (img[:,:,2]).copy()
#        r -= np.mean(r)
#        g -= np.mean(g)
#        b -= np.mean(b)
#        res[i*100:(i+1)*100, j*100:(j+1)*100, 0] = r
#        res[i*100:(i+1)*100, j*100:(j+1)*100, 1] = g
#        res[i*100:(i+1)*100, j*100:(j+1)*100, 2] = b
##res = (res - np.mean(res)) / (np.std(res))
#res = (res - np.min(res)) / (np.max(res) - np.min(res))
#plt.subplot(122)
#plt.imshow(res)
#plt.imsave(r"data/test.jpg", res)

###############################################################################
# 检查数据
#opt = Option()
#imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
#imgs_val_ori = np.load(join(opt.data_npy, "val-img-ori.npy"))
#visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
#labs_val = np.load(join(opt.data_npy, "val-label.npy"))
#
#imgs_val = imgProc(imgs_val)
#imgs_val_ori = imgProc(imgs_val_ori)
#visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
#labs_val = torch.LongTensor(labs_val) - 1
#
#pic_num = 10
#res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
#for i in range(pic_num):
#    for j in range(pic_num):
#        img = imgs_val[pic_num*i+j,:].numpy().transpose((1,2,0))
#        res[i*100:(i+1)*100, j*100:(j+1)*100, :] = img
##res = (res - np.mean(res)) / (np.std(res))
#res = (res - np.min(res)) / (np.max(res) - np.min(res))
##plt.subplot(121)
#plt.imshow(res)

##%%
#opt = Option()
#path = r"D:\pic\URFC-baidu\train_image\003\000149_003.jpg"
#img = plt.imread(path)
##img = (img/255).astype(np.float32)
#
#sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#aug_seq = iaa.Sequential([
#    iaa.Fliplr(0.5),
#    iaa.Flipud(0.5),
#    sometimes(iaa.Affine(
#        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#        rotate=(-15, 15),
#        shear=(-16, 16),
#        order=[0, 1],
#    )),
#    iaa.SomeOf((0, 5), [
#        iaa.OneOf([
#            iaa.GaussianBlur((0, 2)),
#            iaa.AverageBlur(k=(2, 5)),
#            iaa.MedianBlur(k=(3, 5)),
#        ]),
#        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),
#        sometimes(iaa.OneOf([
#            iaa.EdgeDetect(alpha=(0, 0.7)),
#            iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
#        ])),
#        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#        iaa.OneOf([
#            iaa.Dropout((0.01, 0.1), per_channel=0.5),
#            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),per_channel=0.2),
#        ]),
#        iaa.OneOf([
#            iaa.Fog(),
#            iaa.Clouds(),
#        ]),
#        iaa.Invert(0.05, per_channel=True),
#        iaa.Add((-10, 10), per_channel=0.5),
#        iaa.Multiply((0.7, 1.3), per_channel=0.5),
#        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
#    ], random_order=True)
#], random_order=True)
#
##img_aug = aug_seq.augment_image(img)
##img_aug = linear_p(img, 0.02)
#aug = iaa.Fog()
#img_aug = aug.augment_image(img)
#
#time_elapsed = time.time() - since # 用时
#print(time_elapsed)
#plt.imshow(img_aug)

##%%
#opt = Option()
#imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
#visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
#labs_val = np.load(join(opt.data_npy, "val-label.npy"))
#
#imgs_val = imgProc(imgs_val)
#visits_val = torch.FloatTensor(visits_val.transpose(0,3,1,2))
#labs_val = torch.LongTensor(labs_val) - 1
#
#batch = imgs_val[0:100,:]
#batch_aug = aug_batch(batch)
##batch = (batch.permute(0,2,3,1).numpy()*255).astype(np.uint8)
##batch_aug = aug_img(batch)
##batch_aug = (batch_aug / 255.0).astype(np.float32).transpose(0,3,1,2)
##batch_aug = torch.as_tensor(batch_aug, dtype=torch.float32)
#
#time_elapsed = time.time() - since # 用时
#print(time_elapsed)
#
#pic_num = 10
#res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
#for i in range(pic_num):
#    for j in range(pic_num):
#        img = batch_aug[pic_num*i+j,:].numpy().transpose((1,2,0))
#        res[i*100:(i+1)*100, j*100:(j+1)*100, :] = img
##res = (res - np.mean(res)) / (np.std(res))
#res = (res - np.min(res)) / (np.max(res) - np.min(res))
#plt.imshow(res)


