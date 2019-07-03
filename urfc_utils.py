# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import torch
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from linear_p import linear_p


def imgProc(x, means, stds):
    '''image预处理，x为NHWC格式uint8类型的numpy矩阵，生成NCHW的tensor'''
#    for j in range(x.shape[0]):
#        for i in range(3):
#            x[j,:,:,i] = cv2.equalizeHist(x[j,:,:,i]) # 直方图均衡
    
    x = x.astype(np.float32) / 255.0 # 归一化
    
#    for j in range(x.shape[0]):
#            x[j,:,:,:] = linear_p(x[j,:,:,:], 0.02) # 拉伸
    
#    for j in range(x.shape[0]):
#        x[j,:,:,:] = deHaze(x[j,:,:,:]) # 去雾
    
    x = x.transpose(0,3,1,2)
    x = torch.as_tensor(x, dtype=torch.float32)
    
    # 每张图减去均值，匀光
#    for j in range(x.shape[0]):
#        for i in range(3):
#            x[j,i,:,:] -= x[j,i,:,:].mean()
#    
    # 标准化
#    mean = torch.as_tensor(means, dtype=torch.float32)
#    std = torch.as_tensor(stds, dtype=torch.float32)
#    x = x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return x

def func_images(img, random_state, parents, hooks):
    img = (linear_p(img, 0.02) * 255).astype(np.uint8)
    return img

def aug_img(img):
    '''对uint8图像或图像的一个batch（NHWC）进行aug'''
#    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    aug_seq = iaa.Sequential([
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
#            iaa.Dropout((0.01, 0.1), per_channel=0.5),
#            iaa.OneOf([
#                iaa.Fog(),
#                iaa.Clouds(),
#            ]),
##            iaa.Invert(0.05, per_channel=True),
#            iaa.Add((-10, 10), per_channel=0.5),
#            iaa.Multiply((0.7, 1.3), per_channel=0.5),
##            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
#        ], random_order=True)
    ], random_order=True)
    
    if (img.ndim == 3):
        # 对单幅图像进行aug
        img_aug = aug_seq.augment_image(img)
    elif (img.ndim == 4):
        # 对一个batch进行aug
        img_aug = aug_seq.augment_images(img)
    else:
        img_aug = []
    return img_aug

def aug_val_img(img):
    '''对uint8图像或图像的一个batch（NHWC）进行aug'''
    aug_seq = iaa.Lambda(func_images=func_images)
    
    if (img.ndim == 3):
        # 对单幅图像进行aug
        img_aug = aug_seq.augment_image(img)
    elif (img.ndim == 4):
        # 对一个batch进行aug
        img_aug = aug_seq.augment_images(img)
    else:
        img_aug = []
    return img_aug

def aug_batch(batch):
    '''对torch的一个batch（NCHW）进行aug'''
    batch = (batch.permute(0,2,3,1).numpy()*255).astype(np.uint8)
    batch_aug = aug_img(batch)
    batch_aug = (batch_aug / 255.0).astype(np.float32).transpose(0,3,1,2)
    batch_aug = torch.as_tensor(batch_aug, dtype=torch.float32)
    return batch_aug

def aug_val_batch(batch):
    '''对torch的一个batch（NCHW）进行aug'''
    batch = (batch.permute(0,2,3,1).numpy()*255).astype(np.uint8)
    batch_aug = aug_val_img(batch)
    batch_aug = (batch_aug / 255.0).astype(np.float32).transpose(0,3,1,2)
    batch_aug = torch.as_tensor(batch_aug, dtype=torch.float32)
    return batch_aug

def get_tta_batch(batch):
    '''对torch的一个batch（NCHW）进行aug'''
    transform_h = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor(),
    ])
    transform_v = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(1),
        transforms.ToTensor(),
    ])
    transform_h_v = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),
        transforms.ToTensor(),
    ])
    transform_n90 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((-90,-90)), # 负数代表顺时针旋转
        transforms.ToTensor(),
    ])
    transform_p90 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((90,90)),
        transforms.ToTensor(),
    ])
    transform_h_n90 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomRotation((-90,-90)), # 负数代表顺时针旋转
        transforms.ToTensor(),
    ])
    transform_v_n90 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(1),
        transforms.RandomRotation((-90,-90)), # 负数代表顺时针旋转
        transforms.ToTensor(),
    ])
    
    batch_h = batch.clone()
    batch_v = batch.clone()
    batch_h_v = batch.clone()
    batch_n90 = batch.clone()
    batch_p90 = batch.clone()
    batch_h_n90 = batch.clone()
    batch_v_n90 = batch.clone()
    for i in range(batch.shape[0]):
        batch_h[i,:] = transform_h(batch_h[i,:])
        batch_v[i,:] = transform_v(batch_v[i,:])
        
        batch_h_v[i,:] = transform_h_v(batch_h_v[i,:])
        batch_n90[i,:] = transform_n90(batch_n90[i,:])
        batch_p90[i,:] = transform_p90(batch_p90[i,:])
        batch_h_n90[i,:] = transform_h_n90(batch_h_n90[i,:])
        batch_v_n90[i,:] = transform_v_n90(batch_v_n90[i,:])

    return batch_h, batch_v, batch_h_v, batch_n90, batch_p90, batch_h_n90, batch_v_n90

class Record(object):
    '''记录loss、acc等信息'''
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class Logger(object):
    '''保存日志信息'''
    def __init__(self, lr=0, bs=0, wd=0, num_train=0):
        self.lr = lr
        self.bs = bs
        self.wd = wd
        self.num_train = num_train
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)
        self.file.write('\n--------------------{}--------------------\n'
                        .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.file.write('lr {}, batchsize {}, wd {}, num-train {}\n'
                        .format(self.lr, self.bs, self.wd, self.num_train))
        self.file.flush()

    def write(self, msg):
        self.file.write(msg)
        self.file.write('\n')
        self.file.flush()
    
    def close(self):
        self.file.write('---------------------------------------------\n')
        self.file.close()

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_img, self.next_visit, self.next_target = next(self.loader)
        except StopIteration:
            self.next_img = None
            self.next_visit = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.cuda(non_blocking=True)
            self.next_visit = self.next_visit.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
#            self.next_input = self.next_input.float()
#            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.next_img
        visit = self.next_visit
        target = self.next_target
        self.preload()
        return img, visit, target



