# -*- coding: utf-8 -*-
# 数据预处理，参考了https://github.com/czczup/UrbanRegionFunctionClassification
import numpy as np
import pandas as pd
import os
from os.path import join, pardir
import shutil
import stat
import datetime
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import Augmentor
import torch
import cv2

from urfc_option import Option
from dehaze import deHaze
from linear_p import linear_p


def imgProc(x):
    '''image预处理，x为NHWC格式uint8类型的numpy矩阵，生成NCHW的tensor'''
#    for j in range(x.shape[0]):
#        for i in range(3):
#            x[j,:,:,i] = cv2.equalizeHist(x[j,:,:,i]) # 直方图均衡
    
    x = x.astype(np.float32) / 255.0 # 归一化
    
    for j in range(x.shape[0]):
            x[j,:,:,:] = linear_p(x[j,:,:,:], 0.02) # 拉伸
    
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
#    means = [x[:,i,:,:].mean() for i in range(3)]
#    stds = [x[:,i,:,:].std() for i in range(3)]
#    mean = torch.as_tensor(means, dtype=torch.float32)
#    std = torch.as_tensor(stds, dtype=torch.float32)
#    x = x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return x

def deleteFile(filePath):
    '''删除非空文件夹'''
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(join(fileList[0],name), stat.S_IWRITE)
                os.remove(join(fileList[0],name))
        shutil.rmtree(filePath)

def imgDataClean(dir_img, ratio_b=0.6, ratio_w=0.9):
    '''清洗图片数据，删除大部分黑色图像'''
    # 初始化文件夹
    dir_img_cleaned = join(dir_img, pardir, "img_cleaned")
    if not os.path.exists(dir_img_cleaned):
        os.mkdir(dir_img_cleaned)
    
    # 读取数据
    dirs = sorted(os.listdir(dir_img))
    files = {}
    print('Clean Data...')
    for dir in tqdm(dirs):
        files[int(dir)] = []
        dir_img_cleaned_00 = join(dir_img_cleaned, dir)
        if not os.path.exists(dir_img_cleaned_00):
            os.mkdir(dir_img_cleaned_00)
        
        # 删除output文件夹
        if os.path.exists(join(dir_img, dir, "output")):
            deleteFile(join(dir_img, dir, "output"))
        for file in os.listdir(join(dir_img, dir)):
            path = join(dir_img, dir, file)
            img = plt.imread(path)
            
            # 图片黑色和白色部分占比大于ratio则删除
            if ((sum(sum(sum(img==0))) / (100*100*3)) > ratio_b or 
                (sum(sum(sum(img==255))) / (100*100*3)) > ratio_w):
#                shutil.copy(path, dir_img_cleaned_00)
                shutil.move(path, join(dir_img_cleaned_00, file))
                continue
            files[int(dir)].append(path)

def imgData2val(dir_img, dir_img_val):
    '''分一部分数据作为验证集'''
    # 初始化文件夹
    if not os.path.exists(dir_img_val):
        os.mkdir(dir_img_val)
    
    # 读取数据
    dirs = sorted(os.listdir(dir_img))
    files = {}
    for dir in dirs:
        # 删除output文件夹
        if os.path.exists(join(dir_img, dir, "output")):
            deleteFile(join(dir_img, dir, "output"))
        
        path = join(dir_img, dir)
        files[int(dir)] = []
        for file in os.listdir(path):
            files[int(dir)].append(join(path, file))
    
    # 每一类采样200作为验证集，剩下的为训练集
    valid_data = {}
    for i in dirs:
        valid_data[int(i)] = random.sample(files[int(i)], 200)
    
    # 移动验证集数据
    for i in tqdm(dirs):
        dir_img_val_00 = join(dir_img_val, i)
        if not os.path.exists(dir_img_val_00):
            os.mkdir(dir_img_val_00)
        for item in valid_data[int(i)]:
#            shutil.copy(item, dir_img_val_00)
            shutil.move(item, join(dir_img_val_00, item.split('\\')[-1]))

def _imgAug(dir_img, crop_w, crop_h, num_img, multi_threaded=False):
    '''图片数据增强'''
    p = Augmentor.Pipeline(dir_img)
    # 增强操作
    p.crop_by_size(1, width=crop_w, height=crop_h, centre=False)
    p.flip_left_right(0.5)
    p.flip_top_bottom(0.5)
#    p.random_erasing(0.5, rectangle_area=0.5) # 随机遮挡
    p.rotate(0.5, max_left_rotation=20, max_right_rotation=20)
    p.rotate_random_90(0.5) # 随机旋转90、180、270度，注意图片需为方的
    p.zoom_random(0.3, percentage_area=0.8) # 随机放大
    p.random_distortion(0.3,grid_height=5,grid_width=5,magnitude=2) # 弹性扭曲
    p.shear(0.3, max_shear_left=5, max_shear_right=5) # 随机错切（斜向一边）
#    p.skew(0.3, magnitude=0.3) # 透视形变
    p.sample(num_img, multi_threaded=multi_threaded) # 多线程提速但占内存，输出大图慎用多线程防死机

def imgsAug(dir_img, crop_w, crop_h, num_img, multi_threaded=False):
    print('Image Augment...')
    for i in range(1,10):
        # 删除output文件夹
        if os.path.exists(join(dir_img, str(i).zfill(3), "output")):
            deleteFile(join(dir_img, str(i).zfill(3), "output"))
        _imgAug(join(dir_img, str(i).zfill(3)), crop_w, crop_h, num_img, multi_threaded=multi_threaded)

def getSampleTxt(dir_img, path_txt, aug=True):
    '''将增广后的数据写入txt'''
    
    print('Get Sample Txt...')
    # 读取数据
    dirs = sorted(os.listdir(dir_img))
    files = {}
    for dir in dirs:
        if aug:
            path = join(dir_img, dir, "output")
        else:
            # 删除output文件夹
            if os.path.exists(join(dir_img, dir, "output")):
                deleteFile(join(dir_img, dir, "output"))
            path = join(dir_img, dir)
        files[int(dir)] = []
        for file in os.listdir(path):
            files[int(dir)].append(join(path, file))
    
    #各类比例
    nums = [len(files[i+1]) for i in range(9)]
    pert = [(sum(nums) - nums[i]) / sum(nums)  for i in range(9)]
    print(pert)
    
    # 初始化保存目录
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # 写入txt
    f = open(path_txt, "w+")
    for i in range(1, 10):
        for item in files[i]:
            f.write(item[0:-4] + "\n")
    f.close()

def imgs2npy(data_npy):
    '''将增广的图片集转换为一个npy文件'''
    
    print('Image to npy...')
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
    
    # 训练集
    data_list = list(pd.read_csv("data/train.txt", header=None)[0])
    
    names = [a.split('\\')[-1] for a in data_list]
    if 'original' in names[0]: # 文件名包含original则说明是增广数据
        labels = [int(a.split('\\')[-1][0:3]) for a in data_list]
    else:
        labels = [int(a.split('\\')[-2]) for a in data_list]
    
    imgs = []
    for file in tqdm(data_list):
        img = plt.imread(file + ".jpg")
        imgs.append(img)
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.uint8)
    np.save(join(data_npy, "train-img.npy"), imgs)
    np.save(join(data_npy, "train-label.npy"), labels)
    
    # 验证集
    data_list = list(pd.read_csv("data/val.txt", header=None)[0])
    labels = [int(a.split('\\')[-2]) for a in data_list]
    imgs = []
    for file in tqdm(data_list):
        img = plt.imread(file + ".jpg")
        imgs.append(img)
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.uint8)
    np.save(join(data_npy, "val-img.npy"), imgs)
    np.save(join(data_npy, "val-label.npy"), labels)
    
def visit2array(table):
    '''将visit数据转换为矩阵'''
    
    date2position = {}
    datestr2dateint = {}
    str2int = {}
    # 用字典查询代替类型转换，可以减少一部分计算时间
    for i in range(24):
        str2int[str(i).zfill(2)] = i
    
    # 访问记录内的时间从2018年10月1日起，共182天
    # 将日期按日历排列
    for i in range(182):
        date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
        date_int = int(date.__str__().replace("-", ""))
        date2position[date_int] = [i%7, i//7]
        datestr2dateint[str(date_int)] = date_int
    
    strings = table[1]
    init = np.zeros((26, 24, 7))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            # y - 第几周
            # z - 几点钟
            # x - 第几天
            # value - 到访的总人数
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst: # 统计到访的总人数
                init[y][str2int[visit]][x] += 1
    return init.astype(np.float32)

def visits2npys(dir_visit, dir_visit_npy):
    '''将visit数据转换为npy文件'''
    
    # 初始化保存目录
    if not os.path.exists(dir_visit_npy):
        os.makedirs(dir_visit_npy)
    else:
        return
    
    visit_names = os.listdir(dir_visit)
    
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit, visit_name)
        visit_table = pd.read_table(path_visit, header=None)
        visit_array = visit2array(visit_table)
        path_visit_npy = join(dir_visit_npy, visit_name.split('.')[0] + ".npy")
        np.save(path_visit_npy, visit_array)

def visits2npy(dir_visit_npy, data_npy):
    '''将visit数据转换为一个npy文件'''
    
    print('Visit to npy...')
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
    
    data_list = list(pd.read_csv("data/train.txt", header=None)[0])
    visit_names = [a.split('\\')[-1] for a in data_list]
    if 'original' in visit_names[0]: # 文件名包含original则说明是增广数据
        visit_names = [a[13:23] for a in visit_names]
    
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy, visit_name + ".npy")
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "train-visit.npy"), visit_arrays)
    
    data_list = list(pd.read_csv("data/val.txt", header=None)[0])
    visit_names = [a.split('\\')[-1] for a in data_list]
    if 'original' in visit_names[0]: # 文件名包含original则说明是增广数据
        visit_names = [a[13:23] for a in visit_names]
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy, visit_name + ".npy")
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "val-visit.npy"), visit_arrays)

def testData2npy(dir_img_test, dir_visit_npy_test, data_npy):
    '''将测试集数据转换为一个npy文件'''
    print('Test Data to npy...')
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
    
    # 读取数据
    img_names = sorted(os.listdir(dir_img_test))
    
    imgs = []
    for img_name in tqdm(img_names):
        img = plt.imread(join(dir_img_test, img_name))
        imgs.append(img)
    imgs = np.array(imgs)
    np.save(join(data_npy, "test-img.npy"), imgs)
    
    # 读取数据
    visit_names = sorted(os.listdir(dir_visit_npy_test))
    
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy_test, visit_name)
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "test-visit.npy"), visit_arrays)



if __name__ == '__main__':
    opt = Option()
    since = time.time() # 记录时间
#    imgDataClean(opt.dir_img)
#    imgData2val(opt.dir_img, opt.dir_img_val)
    
#    imgsAug(opt.dir_img, 100, 100, opt.num_train, multi_threaded=True)
    getSampleTxt(opt.dir_img, "data/train.txt", aug=False)
    getSampleTxt(opt.dir_img_val, "data/val.txt", aug=False)
    imgs2npy(opt.data_npy)
#    visits2npys(opt.dir_visit, opt.dir_visit_npy)
    visits2npy(opt.dir_visit_npy, opt.data_npy)
    
    # 生成测试集数据
#    visits2npys(opt.dir_visit_test, opt.dir_visit_npy_test)
    testData2npy(opt.dir_img_test, opt.dir_visit_npy_test, opt.data_npy)
    
    time_elapsed = time.time() - since # 用时
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    
    