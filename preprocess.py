# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import shutil
from os.path import join
import datetime
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from urfc_option import Option


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
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy, visit_name + ".npy")
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "train-visit.npy"), visit_arrays)
    
    data_list = list(pd.read_csv("data/val.txt", header=None)[0])
    visit_names = [a.split('\\')[-1] for a in data_list]
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy, visit_name + ".npy")
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "val-visit.npy"), visit_arrays)

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
    


if __name__ == '__main__':
    opt = Option()
    getSampleTxt(opt.dir_img)
    visits2npys(opt.dir_visit, opt.dir_visit_npy)
    visits2npys(opt.dir_visit_test, opt.dir_visit_npy_test)
    imgs2npy(opt.data_npy)
    visits2npy(opt.dir_visit_npy, opt.data_npy)
#    imgs_train = np.load(join(opt.data_npy, "train-img.npy"))
#    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
#    visits_train = np.load(join(opt.data_npy, "train-visit.npy"))
#    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
#    labs_train = np.load(join(opt.data_npy, "train-label.npy"))
#    labs_val = np.load(join(opt.data_npy, "val-label.npy"))
    
    
    