# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec

class Config(object):

    """配置参数"""
    def __init__(self):#, dataset, embedding
        self.embedding_pretrained = True
        self.freeze = True
        # load word2vec
        wvModelPath=r"preTrain/word2vec/models/330cert42Add_skip.model"
        wordModel = Word2Vec.load(wvModelPath)
        word_list = wordModel.wv.index2word
        self.pretrained_weight = np.array([wordModel[word] for word in word_list])  # 预训练词向量

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2#len(self.class_list)                         # 类别数
        self.n_vocab = 168                                               # 词表大小
        self.num_epochs = 10     #手动设置                                       # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 65                                              # 序列处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 128
        self.hidden_size = 512                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64


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
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        # x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
