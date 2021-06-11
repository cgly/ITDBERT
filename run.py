


import numpy as np
from os import path
import os
import time
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler,DataLoader
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import f1_score
from importlib import import_module
import argparse
import csv
import random

# import sys
# #sys.path.append('add absolute path')
from ITDBERT.utils import *


if __name__=='__main__':
    trainFile=r'data\THNS24_2010.csv'
    testFile=r'data\THNS24_2011.csv'
    model_name = "Transformer"   # TextCNN, RCNN, BiLSTM_Att, Transformer,
    num_epoch=10
    SAVE_PATH = r'save_model\/' +model_name+ '.pth'  # 定义模型保存路径
    ###############锁定系统路径########################
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    print(os.getcwd())
    seed_everything()
    ###################DataLoader##############

    train_loader,test_loader=build_dataloader(trainFile,testFile,65,256)
    ########################load Config########################

    x = import_module('model.' + model_name)
    config = x.Config()
    print(model_name+" Config loaded")
    #########################load Model###########################
    model = x.Model(config).to(config.device)
    print("Model loaded")
    print(model)
    optimizer = optim.Adam(model.parameters())
    ########################train############################



    best_acc = 0.0

    for epoch in range(1, num_epoch):  # 10个epoch
        train(model, config.device, train_loader, optimizer, epoch)
        acc = test(model, config.device, test_loader)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
    ########################test############################
    best_model = x.Model(config).to(config.device)
    best_model.load_state_dict(torch.load(SAVE_PATH))
    testfinal(best_model, config.device,train_loader)


