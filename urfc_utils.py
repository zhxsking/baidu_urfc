# -*- coding: utf-8 -*-

from datetime import datetime


class Logger(object):
    def __init__(self, lr=0, bs=0, wd=0, num_train=0):
        self.lr = lr
        self.bs = bs
        self.wd = wd
        self.num_train = num_train
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)
        self.file.write('\n--------------------{}--------------------\n'
                        .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.file.write('lr {:.4f}, batchsize {:.4f}, wd {:.4f}, num-train {:.4f}\n'
                        .format(self.lr, self.bs, self.wd, self.num_train))
        self.file.flush()

    def write(self, msg):
        self.file.write(msg)
        self.file.write('\n')
        self.file.flush()
    
    def close(self):
        self.file.write('---------------------------------------------\n')
        self.file.close()





