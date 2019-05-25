# -*- coding: utf-8 -*-

import numpy as np
from os.path import join
import os
import matplotlib.pyplot as plt

from urfc_option import Option


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
pic_type = 1
res = np.zeros((pic_num*100, pic_num*100, 3), dtype=np.float64)
for i in range(pic_num):
    tmp = []
    for j in range(pic_num):
        img = plt.imread(files[pic_type][pic_num*i+j]) / 255
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

plt.imshow(res)
plt.imsave(r"data/test.jpg", res)
        
        
