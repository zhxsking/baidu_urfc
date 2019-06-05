# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:53:18 2019

@author: zhxsking
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def linear_p(img, p=0.02):
    """对图像进行linear p 处理，一般p=0.02"""
    dim = 1
    if img.ndim > 2: # 判断是否为灰度图
        dim = img.shape[2]
    
    res = np.zeros(img.shape).astype(np.float16)
    for i in range(dim):
        if img.ndim > 2:
            tmp = img[:,:,i].copy()
        else:
            tmp = img.copy()
        # 归一化
        if tmp.dtype=='uint8':
            tmp = (tmp / 255).astype(np.float16)
        elif tmp.dtype=='uint16':
            tmp = (tmp / 65535).astype(np.float16)
        # 统计直方图
        (n, bins) = np.histogram(tmp.ravel(), bins=100, normed=False)
        n_ = n / tmp.size
        prob = n_.cumsum()
        # 直方图累计概率小于p的置0
        ind1 = np.where(prob <= p)
        if len(ind1[0]): # 空情况排除
            tmp[tmp < bins[ind1[0][-1]]] = 0
        # 直方图累计概率大于p的置1
        ind2 = np.where(prob >= 1 - p)
        if len(ind2[0]): # 空情况排除
            tmp[tmp > bins[ind2[0][0]]] = 1
        # 掐头去尾
        s = tmp.copy().ravel()
        s.sort()
        s = np.delete(s, np.where(s==0))
        s = np.delete(s, np.where(s==1))
        # 概率中间部分拉伸到0到1
        if len(s): # 空情况排除
            tmp = (tmp - s[1]) / (s[-1] - s[1])
        tmp[tmp<0] = 0;
        tmp[tmp>1] = 1;
        
        if img.ndim > 2:
            res[:,:,i] = tmp
        else:
            res = tmp
        
    return res
    
if __name__ == '__main__':
    p = 0.02
    path = r'E:\pic\cat.jpg'
    img = mpimg.imread(path)
    res = linear_p(img, p)
    
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(res)

