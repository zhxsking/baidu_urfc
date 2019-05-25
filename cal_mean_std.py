# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# 计算数据集的均值及方差
path_list = list(pd.read_table("data/train.txt", header=None)[0])

r, g, b = [], [], []
for path in tqdm(path_list):
    img = plt.imread(path + '.jpg')
    img = img.astype(np.float32) / 255
    r.append(img[:,:,0] - np.mean(img[:,:,0]))
    g.append(img[:,:,1] - np.mean(img[:,:,1]))
    b.append(img[:,:,2] - np.mean(img[:,:,2]))
means = [np.mean(r), np.mean(g), np.mean(b)]
stds = [np.std(r), np.std(g), np.std(b)]
print('rgb mean std:')
print(means, stds)
