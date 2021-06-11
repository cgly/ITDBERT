import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler,DataLoader
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import gensim
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import os
import random


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

#锁定随机种子，保证实验结果
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


#加载数据
def load_certData(train_path,test_path):
    csv_file = open(r'E:\code\zhuheCode\act-Classification\wasa_data\data\30-7data\THNS24_2010.csv')
    csv_file = open(train_path)
    read_line = csv.reader(csv_file)
    data_pylist = []
    ans_pylist = []
    for line in read_line:
        line = [int(i) for i in line]
        data_pylist.append(line[1:])
        ans_pylist.append(0 if int(line[0]) == 0 else 1)
    train_cert = np.array(data_pylist)
    train_ans = np.array(ans_pylist)

    csv_file = open(r'E:\code\zhuheCode\act-Classification\wasa_data\data\30-7data\THNS24_2011.csv')
    csv_file = open(test_path)
    read_line = csv.reader(csv_file)
    data_pylist = []
    ans_pylist = []
    for line in read_line:
        line = [int(i) for i in line]
        data_pylist.append(line[1:])
        ans_pylist.append(0 if int(line[0]) == 0 else 1)
    test_cert = np.array(data_pylist)
    test_ans = np.array(ans_pylist)
    return (train_cert, train_ans), (test_cert, test_ans)
#构建数据加载器
def build_dataloader(train_path,test_path,max_len=65,batch_size=256):
    MAX_LEN = max_len  # 65覆盖max length 97%的数据
    BATCH_SIZE = batch_size

    # 借助Keras加载imdb数据集
    (x_train, y_train), (x_test, y_test) = load_certData(train_path,test_path)
    print(x_test)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
    print(x_train.shape, x_test.shape)

    # 转化为TensorDataset
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

    # 转化为 DataLoader
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    return train_loader,test_loader



def train(model, DEVICE, train_loader, optimizer, epoch):  # 训练模型
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_ = model(x)
        loss = criterion(y_, y)  # 得到loss
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:  # 打印loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, DEVICE, test_loader):  # 测试模型
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)

def testfinal(model, DEVICE, test_loader):  # 测试模型
    pre_list = []
    y_list = []

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
    test_loss = 0.0
    acc = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]  # .max() 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
        for i in y.cuda().data.cpu().numpy():
            y_list.append(i)
        for i in pred.cuda().data.cpu().numpy():
            pre_list.append(i)
    #############P,R,F1#####################
    print(classification_report(y_list, pre_list, labels=None,
                                target_names=None, sample_weight=None, digits=2))
    #############混淆矩阵####################
    confusion = metrics.confusion_matrix(y_list, pre_list)
    print(confusion)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)