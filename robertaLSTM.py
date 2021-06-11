import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler,DataLoader
from transformers import RobertaModel,RobertaTokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

#加载处理MaskedLM需要的序列处理工具
from ITDBERT.preTrain.MaskedLM.MaskedUtils import *

class Config(object):

    """配置参数"""
    def __init__(self):
        # 训练集
        #self.train_path = r'/act-Classification/data/30-7data/THNS24_2010.csv'
        # 测试集
        #self.test_path = r'/act-Classification/data/30-7data/THNS24_2011.csv'
        # dataset
        self.datasetpkl =r'preTrain/MaskedLM/data/datasetTHNS.pkl'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                        # 类别数
        self.n_vocab = 600 #600                                               # 词表大小，在运行时赋值
        self.num_epochs = 200                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 65                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 768          # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64
        self.reberta_path = r"preTrain/MaskedLM/models/42_H1_24_42"
        # bert的 tokenizer
        self.rebertaTokenizer = RobertaTokenizer.from_pretrained(self.reberta_path)
        self.rebertaModel = RobertaModel.from_pretrained(self.reberta_path)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab-1 )
        #
        self.tokenizer = config.rebertaTokenizer
        self.model=config.rebertaModel
        for param in self.model.parameters():
            param.requires_grad = True


        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

        self.fc2=nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 2)
        self.fc5 = nn.Linear(768, config.num_classes)
    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.model(context, attention_mask=mask)  #, output_all_encoded_layers=False

        '''
        方法1：[cls]+linear
        '''
        out = self.fc2(text_cls)
        out = F.relu(out)
        out = self.fc3(out)
        # out=F.softmax(out)
        return out



        '''
        mothed2：embedding vector+LSTM
        '''
        # H, _ = self.lstm(encoder_out)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 65, 256]
        # M = self.tanh1(H)  # [128, 65, 256]
        # # M = torch.tanh(torch.matmul(H, self.u))
        # alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 65, 1]
        # out = H * alpha  # [128, 65, 256]
        # out = torch.sum(out, 1)  # [128, 256]
        # out = F.relu(out)
        # out = self.fc1(out)
        # #out = self.fc(out)  # [128, 64]
        # return out










def train(model, DEVICE, train_loader, optimizer, epoch):  # 训练模型
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(train_loader):
        #x:torch.Size([256, 65])-->tensor([256,1])

        #x, y = x.to(DEVICE), y.to(DEVICE)
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
        #x, y = x.to(DEVICE), y.to(DEVICE)
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


def evaluate( model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        #report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        print(acc)
        print(confusion)
        return acc, loss_total / len(data_iter),  confusion
    return acc, loss_total / len(data_iter)



if __name__=='__main__':

    config = Config()

    train_data, test_data = bulid_dataset(config)

    train_iter = bulid_iterator(train_data, config)
    test_iter = bulid_iterator(test_data, config)

    model = Model(config).to(config.device)
    print(model)
    optimizer = optim.Adam(model.parameters())

    best_acc = 0.0
    PATH = r'preTrain/MaskedLM/save_model/42rebertaH1_model.pth'  # 定义模型保存路径

    for epoch in range(1, 200):  # 10个epoch
        train(model, config.device, train_iter, optimizer, epoch)
        acc = test(model, config.device, test_iter)
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

    # 检验保存的模型
    best_model = Model(config).to(config.device)
    best_model.load_state_dict(torch.load(PATH))
    evaluate(best_model,test_iter,True)
