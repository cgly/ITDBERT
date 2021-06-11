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
        #
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2#len(self.class_list)                         # 类别数
        self.n_vocab = 168                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 65                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 128     #or 1                                      # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if not config.embedding_pretrained :
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_weight))
            self.embedding.weight.requires_grad = config.freeze
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
