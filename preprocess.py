# -*- coding: utf-8 -*-
# 数据预处理，参考了https://github.com/czczup/UrbanRegionFunctionClassification
import numpy as np
import pandas as pd
import os
from os.path import join
import shutil
import stat
import datetime
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.ensemble import IsolationForest

from urfc_option import Option


def deleteFile(filePath):
    '''删除非空文件夹'''
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(join(fileList[0],name), stat.S_IWRITE)
                os.remove(join(fileList[0],name))
        shutil.rmtree(filePath)

def imgDataClean(dir_img, ratio_b=0.3, ratio_w=0.9):
    '''清洗图片数据，记录大部分黑或大部分白的图像'''
    print('Clean Data...')
    
    # 初始化保存目录
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # 读取数据
    dirs = (os.listdir(dir_img))
    f = open(r"data/bad-files.txt", "w+")
    for dir in tqdm(dirs):
        for file in os.listdir(join(dir_img, dir)):
            path = join(dir_img, dir, file)
#            img = plt.imread(path)
            img = Image.open(path)
            img_gray = img.convert('L')
            img_gray = np.array(img_gray)
            
            # 图片黑色和白色部分占比大于ratio则删除
            if ((np.sum(img_gray==0) / (10000)) > ratio_b or 
                (np.sum(img_gray==255) / (10000)) > ratio_w):
                f.write(path + "\n")
                continue
    f.close()

def imgDataClean_iforest(dir_img):
    '''利用孤立森林去除异常点'''
    # 读取数据
    dirs = (os.listdir(dir_img))
    files = []
    for dir in dirs:
        path = join(dir_img, dir)
        for file in os.listdir(path):
            files.append(join(path, file))
    
    bad_files = list(pd.read_csv("data/bad-files.txt", header=None)[0])
    
    good_files = list(set(files)-set(bad_files))
    
    files = good_files.copy()
    clf = IsolationForest(behaviour='new', verbose=1, contamination='auto', n_jobs=-1)
    
    all_bad_files = []
    for i in range(5):
        imgs = []
        for file in tqdm(files):
            img = plt.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgs.append(img)
        imgs = np.array(imgs)
        x = imgs.reshape(imgs.shape[0], -1)
        
        clf.fit(x)
        y_pred = clf.predict(x)
        
        idx = (np.where(y_pred == -1))[0]
        
        bad_files = [files[i] for i in idx]
        all_bad_files = list(set(all_bad_files + bad_files))
        
        files = list(set(files)-set(bad_files))
        
        txt_name = "data/bad-files-iforest-" + str(i) + ".txt"
        f = open(txt_name, "w+")
        for item in all_bad_files:
            f.write(item)
        f.close()

def getSampleTxt(dir_img, num_train):
    '''将数据写入txt'''
    print('Get Sample Txt...')
    
    # 初始化保存目录
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # 读取数据
    dirs = (os.listdir(dir_img))
    files = []
    for dir in dirs:
        path = join(dir_img, dir)
        for file in os.listdir(path):
            files.append(join(path, file))
    
    bad_files = list(pd.read_csv("data/bad-files.txt", header=None)[0])
    bad_files_iforest = list(pd.read_csv("data/bad-files-iforest-9.txt", header=None)[0])
        
#    good_files = list(set(files)-set(bad_files))
    good_files = list(set(files)-set(bad_files)-set(bad_files_iforest))
        
    f = open("data/val.txt", "w+")
    valid_data = {}
    train_data = {}
    for i in range(1, 10):
        tmp = [a for a in good_files if int(a.split('\\')[-2]) == i]
        valid_data[i] = random.sample(tmp, 200)
        train_data[i] = list(set(tmp) - set(valid_data[i]))
        for item in valid_data[i]:
            f.write(item[0:-4] + "\n")
    f.close()
    
    f = open("data/train.txt", "w+")
    for i in range(1, 10):
        for item in train_data[i]:
            f.write(item[0:-4]+ "\n")
    f.close()
    
    train_data_over = {}
    for i in range(1, 10):
        if (num_train <= len(train_data[i])):
            train_data_over[i] = random.sample(train_data[i], num_train)
        else:
            train_data_over[i] = train_data[i]
            for j in range(num_train-len(train_data[i])):
                train_data_over[i].append(train_data[i][random.randint(0, len(train_data[i])-1)])
    
    f = open("data/train-over.txt", "w+")
    for i in range(1, 10):
        for item in train_data_over[i]:
            f.write(item[0:-4] + "\n")
    f.close()

def imgs2npy(data_npy, get_ori=False):
    '''将图片集转换为一个npy文件'''
    
    print('Image to npy...')
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
    
    if get_ori:
        # 训练集
        data_list = list(pd.read_csv("data/train.txt", header=None)[0])
        labels = [int(a.split('\\')[-2]) for a in data_list]
        
        imgs = []
        for file in tqdm(data_list):
            img = plt.imread(file + ".jpg")
            imgs.append(img)
        imgs = np.array(imgs)
        labels = np.array(labels, dtype=np.uint8)
        np.save(join(data_npy, "train-img.npy"), imgs)
        np.save(join(data_npy, "train-label.npy"), labels)
    
    # 过采样训练集
    data_list = list(pd.read_csv("data/train-over.txt", header=None)[0])
    labels = [int(a.split('\\')[-2]) for a in data_list]
    
    imgs = []
    for file in tqdm(data_list):
        img = plt.imread(file + ".jpg")
        imgs.append(img)
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.uint8)
    np.save(join(data_npy, "train-over-img.npy"), imgs)
    np.save(join(data_npy, "train-over-label.npy"), labels)
    
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

def visits2npy(dir_visit_npy, data_npy, get_ori=False):
    '''将visit数据转换为一个npy文件'''
    
    print('Visit to npy...')
    # 初始化保存目录
    if not os.path.exists(data_npy):
        os.makedirs(data_npy)
    
    if get_ori:
        data_list = list(pd.read_csv("data/train.txt", header=None)[0])
        visit_names = [a.split('\\')[-1] for a in data_list]
        visit_arrays = []
        for visit_name in tqdm(visit_names):
            path_visit = join(dir_visit_npy, visit_name + ".npy")
            visit_array = np.load(path_visit)
            visit_arrays.append(visit_array)
        visit_arrays = np.array(visit_arrays)
        np.save(join(data_npy, "train-visit.npy"), visit_arrays)
    
    data_list = list(pd.read_csv("data/train-over.txt", header=None)[0])
    visit_names = [a.split('\\')[-1] for a in data_list]
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy, visit_name + ".npy")
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "train-over-visit.npy"), visit_arrays)
    
    data_list = list(pd.read_csv("data/val.txt", header=None)[0])
    visit_names = [a.split('\\')[-1] for a in data_list]
    visit_arrays = []
    for visit_name in tqdm(visit_names):
        path_visit = join(dir_visit_npy, visit_name + ".npy")
        visit_array = np.load(path_visit)
        visit_arrays.append(visit_array)
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "val-visit.npy"), visit_arrays)

def visits2npy_simple_fea(dir_visit_npy, data_npy):
    '''visit数据简单特征提取'''
    
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
        
        nums_day = np.zeros((7), dtype=np.float32)
        nums_week = np.zeros((26), dtype=np.float32)
        nums_hour = np.zeros((24), dtype=np.float32)
        for day in range(7):
            nums_day[day] = visit_array[:,:,day].sum()
        for week in range(26):
            nums_week[week] = visit_array[week,:,:].sum()
        for hour in range(24):
            nums_hour[hour] = visit_array[:,hour,:].sum()
        
        visit_arrays.append(np.hstack((nums_day,nums_week,nums_hour)))
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
        
        nums_day = np.zeros((7), dtype=np.float32)
        nums_week = np.zeros((26), dtype=np.float32)
        nums_hour = np.zeros((24), dtype=np.float32)
        for day in range(7):
            nums_day[day] = visit_array[:,:,day].sum()
        for week in range(26):
            nums_week[week] = visit_array[week,:,:].sum()
        for hour in range(24):
            nums_hour[hour] = visit_array[:,hour,:].sum()
        
        visit_arrays.append(np.hstack((nums_day,nums_week,nums_hour)))
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

def testData2npy_simple_fea(dir_img_test, dir_visit_npy_test, data_npy):
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
        
        nums_day = np.zeros((7), dtype=np.float32)
        nums_week = np.zeros((26), dtype=np.float32)
        nums_hour = np.zeros((24), dtype=np.float32)
        for day in range(7):
            nums_day[day] = visit_array[:,:,day].sum()
        for week in range(26):
            nums_week[week] = visit_array[week,:,:].sum()
        for hour in range(24):
            nums_hour[hour] = visit_array[:,hour,:].sum()
        
        visit_arrays.append(np.hstack((nums_day,nums_week,nums_hour)))
    visit_arrays = np.array(visit_arrays)
    np.save(join(data_npy, "test-visit.npy"), visit_arrays)



if __name__ == '__main__':
    opt = Option()
    since = time.time() # 记录时间
#    imgDataClean(opt.dir_img)
#    imgDataClean_iforest(opt.dir_img)
    
    getSampleTxt(opt.dir_img, opt.num_train)
#    imgs2npy(opt.data_npy, get_ori=True)
#    visits2npys(opt.dir_visit, opt.dir_visit_npy)
#    visits2npy(opt.dir_visit_npy, opt.data_npy, get_ori=True)
    
    # 生成测试集数据
#    visits2npys(opt.dir_visit_test, opt.dir_visit_npy_test)
#    testData2npy(opt.dir_img_test, opt.dir_visit_npy_test, opt.data_npy)
    
    time_elapsed = time.time() - since # 用时
    print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
    
    