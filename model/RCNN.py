# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec

class Config(object):

    """配置参数"""
    def __init__(self):
        self.embedding_pretrained = True
        self.freeze=True
        #load word2vec
        wvModelPath = r"preTrain/word2vec/models/330cert42Add_skip.model"
        wordModel = Word2Vec.load(wvModelPath)
        word_list = wordModel.wv.index2word
        self.pretrained_weight = np.array([wordModel[word] for word in word_list])  # 预训练词向量

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2#len(self.class_list)                         # 类别数
        self.n_vocab = 168                                                # 词表大小
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 65                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 128
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if not config.embedding_pretrained:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_weight))
            self.embedding.weight.requires_grad = config.freeze
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
