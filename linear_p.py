# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def linear_p(img, p=0.02):
    """对图像进行linear p 处理，一般p=0.02"""
    dim = 1
    if img.ndim > 2: # 判断是否为灰度图
        dim = img.shape[2]
    
    res = np.zeros(img.shape).astype(np.float32)
    for i in range(dim):
        if img.ndim > 2:
            tmp = img[:,:,i].copy()
        else:
            tmp = img.copy()
        
        # 归一化
        if tmp.dtype=='uint8':
            tmp = (tmp / 255).astype(np.float32)
        elif tmp.dtype=='uint16':
            tmp = (tmp / 65535).astype(np.float32)
            
        # 统计直方图
        (n, bins) = np.histogram(tmp.ravel(), bins=100)
        n_ = n / tmp.size
        prob = n_.cumsum()
        
        # 直方图累计概率小于p的置0
        ind1 = np.where(prob <= p)
        if len(ind1[0]): # 空情况排除
            tmp[tmp < bins[ind1[0][-1]]] = 2
        tmp_min = np.min(tmp) # 选出拉伸区间内的最小值
        tmp[tmp==2] = 0
            
        # 直方图累计概率大于p的置1
        ind2 = np.where(prob >= 1 - p)
        if len(ind2[0]): # 空情况排除
            tmp[tmp > bins[ind2[0][0]]] = -1
        tmp_max = np.max(tmp) # 选出拉伸区间内的最大值
        tmp[tmp==-1] = 1
        
        # 概率中间部分拉伸到0到1
        if (tmp_max != tmp_min): # 空情况排除
            tmp = (tmp - tmp_min) / (tmp_max - tmp_min)
        tmp[tmp<0] = 0;
        tmp[tmp>1] = 1;
        
        if img.ndim > 2:
            res[:,:,i] = tmp
        else:
            res = tmp
        
    return res
    
if __name__ == '__main__':
    p = 0.02
    path = r"D:\pic\URFC-baidu\train_image\004\000380_004.jpg"
#    path = r'D:\pic\other\cat-black1.jpg'
    img = plt.imread(path)
    import time
    since = time.time()
    res = linear_p(img, p)
    stop = time.time()
    print(stop-since)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(res)

