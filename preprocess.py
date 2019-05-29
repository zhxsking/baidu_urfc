# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from os.path import join
import datetime
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import Augmentor

from urfc_option import Option


def _imgAug(dir_img, crop_w, crop_h, num_img, multi_threaded=False):
    '''图片数据增强'''
    p = Augmentor.Pipeline(dir_img)
    # 增强操作
    p.crop_by_size(1, width=crop_w, height=crop_h, centre=False)
    p.flip_left_right(0.5)
    p.flip_top_bottom(0.5)
#    p.random_erasing(0.5, rectangle_area=0.5) # 随机遮挡
    p.rotate(0.5, max_left_rotation=10, max_right_rotation=10)
    p.rotate_random_90(0.5) # 随机旋转90、180、270度，注意图片需为方的
    p.zoom_random(0.3, percentage_area=0.5) # 随机放大
    p.random_distortion(0.3,grid_height=5,grid_width=5,magnitude=5) # 弹性扭曲
    p.shear(0.3, max_shear_left=5, max_shear_right=5) # 随机错切（斜向一边）
    p.skew(0.3, magnitude=0.3) # 透视形变
    p.sample(num_img, multi_threaded=multi_threaded) # 多线程提速但占内存，输出大图慎用多线程防死机

def imgsAug(dir_img, crop_w, crop_h, num_img, multi_threaded=False):
    for i in range(1,10):
        _imgAug(join(dir_img, str(i).zfill(3)), crop_w, crop_h, num_img, multi_threaded=multi_threaded)

def getSampleTxt(dir_img):
    '''将数据分为训练集和验证集，写入txt'''
    
    # 读取数据
    dirs = sorted(os.listdir(dir_img))
    files = {}
    for dir in dirs:
        path = join(dir_img, dir)
        files[int(dir)] = []
        for file in os.listdir(path):
            files[int(dir)].append(join(path, file))
    
    #各类比例
    nums = [len(files[i+1]) for i in range(9)]
    pert = [(sum(nums) - nums[i]) / sum(nums)  for i in range(9)]
    print(pert)
    
    # 每一类采样200作为验证集，剩下的为训练集
    valid_data = {}
    train_data = {}
    for i in range(1, 10):
        valid_data[i] = random.sample(files[i], 200)
        train_data[i] = list(set(files[i]) - set(valid_data[i]))
    
    # 初始化保存目录
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # train写入txt
    f = open("data/train.txt", "w+")
    for i in range(1, 10):
        for item in train_data[i]:
            f.write(item.split('.')[0] + "\n")
    f.close()
    
    # val写入txt
    f = open("data/val.txt", "w+")
    for i in range(1, 10):
        for item in valid_data[i]:
            f.write(item.split('.')[0] + "\n")
    f.close()

def getAugSampleTxt(dir_img, num_val):
    '''将增广后的数据分为训练集和验证集，写入txt'''
    
    # 读取数据
    dirs = sorted(os.listdir(dir_img))
    files = {}
    for dir in dirs:
        path = join(dir_img, dir, "output")
        files[int(dir)] = []
        for file in os.listdir(path):
            files[int(dir)].append(join(path, file))
    
    #各类比例
    nums = [len(files[i+1]) for i in range(9)]
    pert = [(sum(nums) - nums[i]) / sum(nums)  for i in range(9)]
    print(pert)
    
    # 每一类采样num_val个作为验证集，剩下的为训练集
    valid_data = {}
    train_data = {}
    for i in range(1, 10):
        valid_data[i] = random.sample(files[i], num_val)
        train_data[i] = list(set(files[i]) - set(valid_data[i]))
    
    # 初始化保存目录
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # train写入txt
    f = open("data/train.txt", "w+")
    for i in range(1, 10):
        for item in train_data[i]:
            f.write(item.split('.')[0] + "\n")
    f.close()
    
    # val写入txt
    f = open("data/val.txt", "w+")
    for i in range(1, 10):
        for item in valid_data[i]:
            f.write(item.split('.')[0] + "\n")
    f.close()

def imgs2npy(data_npy):
    '''将图片集转换为一个npy文件'''
    
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
     
    data_list = list(pd.read_csv("data/train.txt", header=None)[0])
    labels = [int(a.split('\\')[-1][7:10]) for a in data_list]
    imgs = []
    for file in tqdm(data_list):
        img = plt.imread(file + ".jpg")
        imgs.append(img)
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.uint8)
    np.save(join(data_npy, "train-img.npy"), imgs)
    np.save(join(data_npy, "train-label.npy"), labels)
    
    data_list = list(pd.read_csv("data/val.txt", header=None)[0])
    labels = [int(a.split('\\')[-1][7:10]) for a in data_list]
    imgs = []
    for file in tqdm(data_list):
        img = plt.imread(file + ".jpg")
        imgs.append(img)
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.uint8)
    np.save(join(data_npy, "val-img.npy"), imgs)
    np.save(join(data_npy, "val-label.npy"), labels)

def augImgs2npy(data_npy):
    '''将增广的图片集转换为一个npy文件'''
    
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
    
    # D:\pic\URFC-baidu\train_image\002\output\002_original_001922_002.jpg_db4c590f-dd4c-46d9-b4cf-7a434b08e3e0.jpg
    # 训练集
    data_list = list(pd.read_csv("data/train.txt", header=None)[0])
    labels = [int(a.split('\\')[-1][0:3]) for a in data_list]
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
    labels = [int(a.split('\\')[-1][0:3]) for a in data_list]
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


if __name__ == '__main__':
    opt = Option()
    imgsAug(opt.dir_img, 100, 100, opt.num_aug, multi_threaded=True)
    getAugSampleTxt(opt.dir_img, opt.num_val)
    augImgs2npy(opt.data_npy)
#    visits2npys(opt.dir_visit, opt.dir_visit_npy)
#    visits2npys(opt.dir_visit_test, opt.dir_visit_npy_test)
    visits2npy(opt.dir_visit_npy, opt.data_npy)
    
    
    