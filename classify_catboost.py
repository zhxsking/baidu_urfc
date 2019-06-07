# -*- coding: utf-8 -*-

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
from sklearn.utils import shuffle
import hyperopt

from preprocess import imgProc
from cnn import mResNet18
from urfc_option import Option


if __name__ == '__main__':
    __spec__ = None
    opt = Option()
    
    # 加载数据
    print('Loading Data...')
    imgs_train = np.load(join(opt.data_npy, "train-img.npy"))
    imgs_val = np.load(join(opt.data_npy, "val-img.npy"))
    visits_train = np.load(join(opt.data_npy, "train-visit.npy"))
    visits_val = np.load(join(opt.data_npy, "val-visit.npy"))
    labs_train = np.load(join(opt.data_npy, "train-label.npy")) - 1
    labs_val = np.load(join(opt.data_npy, "val-label.npy")) - 1
    imgs_test = np.load(join(opt.data_npy, "test-img.npy"))
    visits_test = np.load(join(opt.data_npy, "test-visit.npy"))
    
    # 打乱数据
    imgs_train, visits_train, labs_train = shuffle(imgs_train, visits_train, labs_train)
    imgs_val, visits_val, labs_val = shuffle(imgs_val, visits_val, labs_val)
    
    # 预处理
    imgs_train = imgProc(imgs_train)
    imgs_val = imgProc(imgs_val)
    imgs_test = imgProc(imgs_test)
    
    #%% 
    fea_train = visits_train
    fea_train = fea_train.reshape((fea_train.shape[0],-1))

    fea_val = visits_val
    fea_val = fea_val.reshape((fea_val.shape[0],-1))
    
    fea_test = visits_test
    fea_test = fea_test.reshape((fea_test.shape[0],-1))
    
    #%% 调参
    def f(params):
        model = CatBoostClassifier(
                learning_rate = params['lr'],
                l2_leaf_reg = int(params['l2']),
                iterations = 200,
                eval_metric = 'Accuracy',
                random_seed = 42,
                use_best_model = True,
                logging_level='Silent',
                task_type='GPU',
                )
        model.fit(fea_train, labs_train, eval_set=(fea_val, labs_val))
        y_pred_val = model.predict(fea_val)
        acc = sum(labs_val==y_pred_val.squeeze()) / len(y_pred_val)
        return -acc
    
    params_space = {
            'lr': hyperopt.hp.uniform('lr', 1e-4, 1),
            'l2': hyperopt.hp.uniform('l2', 0, 5),
            }
    trials = hyperopt.Trials()

    best = hyperopt.fmin(
            f,
            space = params_space,
            algo = hyperopt.tpe.suggest,
            max_evals = 50,
            trials = trials,
            )



    #%% 训练
    print('Start training...')
    model = CatBoostClassifier(
            iterations = 150,
            learning_rate = 0.2,
            eval_metric = 'Accuracy',
            random_seed = 42,
            logging_level = 'Verbose',
            use_best_model = True,
    #        od_type = 'Iter', # early stop
    #        od_wait = 140,
            task_type = 'GPU',
            )
    model.fit(fea_train, labs_train, eval_set=(fea_val, labs_val))
    
    eval_metrics = model.eval_metrics(Pool(fea_val, labs_val), ['AUC'])
    ee=[]
    for e in eval_metrics:
        ee.append(eval_metrics[e])
    ee = np.array(ee)
    
    # 预测数据集
    print('Start predicting...')
    y_pred_val = model.predict(fea_val)
    print('The acc of prediction is:', sum(labs_val==y_pred_val.squeeze()) / len(y_pred_val))
    
    y_pred_test = model.predict(fea_test)
    y_pred_test = (y_pred_test.squeeze()).astype(np.uint8)
    
    f = open(r"data/out-label-catboost.txt", "w+")
    cnt = 0
    for j in range(len(y_pred_test)):
        f.write("{} \t {}\n".format(str(cnt).zfill(6), str(y_pred_test[j]+1).zfill(3)))
        cnt += 1
    f.close()
    