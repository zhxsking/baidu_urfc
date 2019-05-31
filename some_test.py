# -*- coding: utf-8 -*-

import numpy as np
from os.path import join
import os
import matplotlib.pyplot as plt
import cv2
from boxx import show

from urfc_option import Option


def histq(img):
    (r, g, b) = cv2.split(img)
    
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((rH, gH, bH))
    return result

# 读取数据
opt = Option()
dirs = sorted(os.listdir(opt.dir_img))
files = {}
for dir in dirs:
    path = join(opt.dir_img, dir)
    files[int(dir)] = []
    for file in os.listdir(path):
        files[int(dir)].append(join(path, file))

# 匀光测试
pic_num = 10
pic_type = 5
res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
for i in range(pic_num):
    for j in range(pic_num):
        img = plt.imread(files[pic_type][pic_num*i+j])
        img = histq(img) / 255
        r = (img[:,:,0]).copy()
        g = (img[:,:,1]).copy()
        b = (img[:,:,2]).copy()
        r -= np.mean(r)
        g -= np.mean(g)
        b -= np.mean(b)
        res[i*100:(i+1)*100, j*100:(j+1)*100, 0] = r
        res[i*100:(i+1)*100, j*100:(j+1)*100, 1] = g
        res[i*100:(i+1)*100, j*100:(j+1)*100, 2] = b
#res = (res - np.mean(res)) / (np.std(res))
res = (res - np.min(res)) / (np.max(res) - np.min(res))
plt.subplot(121)
plt.imshow(res)
#plt.imsave(r"data/test.jpg", res)
       
 
# 读取测试数据
opt = Option()
files = sorted(os.listdir(opt.dir_img_test))

#pic_num = 30
res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
for i in range(pic_num):
    for j in range(pic_num):
        img = plt.imread(join(opt.dir_img_test, files[pic_num*i+j]))
        img = histq(img) / 255
        r = (img[:,:,0]).copy()
        g = (img[:,:,1]).copy()
        b = (img[:,:,2]).copy()
        r -= np.mean(r)
        g -= np.mean(g)
        b -= np.mean(b)
        res[i*100:(i+1)*100, j*100:(j+1)*100, 0] = r
        res[i*100:(i+1)*100, j*100:(j+1)*100, 1] = g
        res[i*100:(i+1)*100, j*100:(j+1)*100, 2] = b
#res = (res - np.mean(res)) / (np.std(res))
res = (res - np.min(res)) / (np.max(res) - np.min(res))
plt.subplot(122)
plt.imshow(res)
#plt.imsave(r"data/test.jpg", res)