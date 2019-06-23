# -*- coding: utf-8 -*-

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
import hyperopt

from urfc_utils import Logger, imgProc, aug_batch, aug_val_batch
from cnn import mResNet18
from urfc_option import Option


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    imgs_train = np.load(join(opt.data_npy, "train-over-img.npy"))
    visits_train = np.load(join(opt.data_npy, "train-over-visit.npy"))
    labs_train = np.load(join(opt.data_npy, "train-over-label.npy")) - 1
    
    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_val = np.load(join(opt.data_npy, "val-label.npy")) - 1
    
    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
    
    # 打乱数据
    imgs_train, visits_train, labs_train = shuffle(imgs_train, visits_train, labs_train)
    imgs_val, visits_val, labs_val = shuffle(imgs_val, visits_val, labs_val)
    
    # 预处理
#    imgs_train = imgProc(imgs_train)
#    imgs_val = imgProc(imgs_val)
#    imgs_test = imgProc(imgs_test)
#    
#    imgs_train = imgs_train.numpy()
#    imgs_val = imgs_val.numpy()
#    imgs_test = imgs_test.numpy()
    
    #%%
    from skimage import feature as ft
#    features = ft.hog(imgs_val,  # input image
#                      orientations=ori,  # number of bins
#                      pixels_per_cell=ppc, # pixel per cell
#                      cells_per_block=cpb, # cells per blcok
#                      block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
#                      transform_sqrt = True, # power law compression (also known as gamma correction)
#                      feature_vector=True, # flatten the final vectors
#                      visualise=False) # return HOG map
    fea_train = []
    for i in tqdm(range(imgs_train.shape[0])):
        fea_train.append(ft.hog(imgs_train[i,:]).astype(np.float32))
    fea_train = np.array(fea_train)
        
    fea_val = []
    for i in tqdm(range(imgs_val.shape[0])):
        fea_val.append(ft.hog(imgs_val[i,:]).astype(np.float32))
    fea_val = np.array(fea_val)
    
    a=imgs_val[0,:]
    features = ft.hog(a) # return HOG map

    
    #%% 
    v1 = np.sum(visits_train, axis=1)
    v2 = np.sum(visits_train, axis=2)
    v3 = np.sum(visits_train, axis=3)
    v1_ = v1.reshape((v1.shape[0],-1))
    v2_ = v2.reshape((v2.shape[0],-1))
    v3_ = v3.reshape((v3.shape[0],-1))
    fea_train = np.c_[v1_, v2_, v3_]
    
    v1 = np.sum(visits_val, axis=1)
    v2 = np.sum(visits_val, axis=2)
    v3 = np.sum(visits_val, axis=3)
    v1_ = v1.reshape((v1.shape[0],-1))
    v2_ = v2.reshape((v2.shape[0],-1))
    v3_ = v3.reshape((v3.shape[0],-1))
    fea_val = np.c_[v1_, v2_, v3_]
    
    fea_train1 = np.c_[fea_train, visits_train.reshape((visits_train.shape[0],-1))]
    fea_val1 = np.c_[fea_val, visits_val.reshape((visits_val.shape[0],-1))]
    
    #%%
    fea_train = visits_train
    fea_train = fea_train.reshape((fea_train.shape[0],-1))

    fea_val = visits_val
    fea_val = fea_val.reshape((fea_val.shape[0],-1))
    
    fea_test = visits_test
    fea_test = fea_test.reshape((fea_test.shape[0],-1))
    
    #%% 调参
#    def f(params):
#        model = CatBoostClassifier(
#                learning_rate = 0.2,
#                l2_leaf_reg = params['l2'],
#                depth = params['depth'],
#                random_strength = params['random_strength'],
#                
#                iterations = 200,
#                eval_metric = 'Accuracy',
#                random_seed = 42,
#                use_best_model = True,
#                logging_level='Silent',
#                task_type='GPU',
#                )
#        model.fit(fea_train, labs_train, eval_set=(fea_val, labs_val))
#        y_pred_val = model.predict(fea_val)
#        acc = sum(labs_val==y_pred_val.squeeze()) / len(y_pred_val)
#        return -acc
#    
#    params_space = {
#            'l2': hyperopt.hp.uniform('l2', 0, 5),
#            'depth': hyperopt.hp.choice('depth', range(1,8)),
#            'random_strength': hyperopt.hp.uniform('random_strength', 0, 10),
#            }
#    trials = hyperopt.Trials()
#
#    best = hyperopt.fmin(
#            f,
#            space = params_space,
#            algo = hyperopt.tpe.suggest,
#            max_evals = 50,
#            trials = trials,
#            )

    #%% 训练
    print('Start training...')
    model = CatBoostClassifier(
            learning_rate = 0.2,
#            l2_leaf_reg = 2,
##            depth = 6,
#            random_strength = 3,
            
#            class_weights = [1,4],
            iterations = 210,
            eval_metric = 'Accuracy',
            random_seed = 42,
            logging_level = 'Verbose',
            use_best_model = True,
    #        od_type = 'Iter', # early stop
    #        od_wait = 140,
            task_type = 'GPU',
            )
    model.fit(fea_train, labs_train, eval_set=(fea_val, labs_val))
    
    fea_imp = model.feature_importances_
    fea_imp_sort = sorted(fea_imp, reverse=True) 
    fea_idx = (-fea_imp).argsort()
    
    fea_imp[fea_idx[850]]
    print((np.array(fea_imp_sort)==0).argmax(axis=0)) # 第一个0的位置
    
    fea_train_top100 = fea_train1[:,fea_idx[0:928]]
    fea_val_top100 = fea_val1[:,fea_idx[0:928]]
    
    fea_train_img_top850 = fea_train_top100
    fea_val_img_top850 = fea_val_top100
    
    fea_train_vis_top928 = fea_train_top100
    fea_val_vis_top928 = fea_val_top100
    
    fea_train2 = np.c_[fea_train_img_top850, fea_train_vis_top928]
    fea_val2 = np.c_[fea_val_img_top850, fea_val_vis_top928]
    
    model_top = CatBoostClassifier(
            learning_rate = 0.2,
            l2_leaf_reg = 2,
            random_strength = 3,
            iterations = 210,
            eval_metric = 'Accuracy',
            random_seed = 42,
            logging_level = 'Verbose',
            use_best_model = True,
            task_type = 'GPU',
            )
    model_top.fit(fea_train_top100, labs_train, eval_set=(fea_val_top100, labs_val))
    
    p = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    fea_train_new = p.fit_transform(fea_train_top100)
    fea_val_new = p.fit_transform(fea_val_top100)
    
    model_new = CatBoostClassifier(
            learning_rate = 0.2,
#            l2_leaf_reg = 2,
#            random_strength = 3,
            iterations = 210,
            eval_metric = 'Accuracy',
            random_seed = 42,
            logging_level = 'Verbose',
            use_best_model = True,
            task_type = 'GPU',
            )
    model_new.fit(fea_train_new, labs_train, eval_set=(fea_val_new, labs_val))
    
    eval_metrics = model.eval_metrics(Pool(fea_val, labs_val), ['AUC'])
    ee=[]
    for e in eval_metrics:
        ee.append(eval_metrics[e])
    ee = np.array(ee)
#    
#    # 预测数据集
#    print('Start predicting...')
#    y_pred_val = model.predict(fea_val)
#    print('The acc of prediction is:', sum(labs_val==y_pred_val.squeeze()) / len(y_pred_val))
#    
#    y_pred_test = model.predict(fea_test)
#    y_pred_test = (y_pred_test.squeeze()).astype(np.uint8)
#    
#    f = open(r"data/out-label-catboost.txt", "w+")
#    cnt = 0
#    for j in range(len(y_pred_test)):
#        f.write("{} \t {}\n".format(str(cnt).zfill(6), str(y_pred_test[j]+1).zfill(3)))
#        cnt += 1
#    f.close()
    