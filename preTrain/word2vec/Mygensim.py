from gensim.models import word2vec
import multiprocessing
import jieba
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# import logging
import os
import csv

path = get_tmpfile("w2v_model.model") #创建临时文件
model_path=(r'')
sentences_path=(r'')

sentences = word2vec.LineSentence(sentences_path)
#sg=0 CBOW
#sg=1 skip-gram
model = Word2Vec(sentences, sg=1, size=128,  window=10,  min_count=3,  negative=3, sample=0.001, hs=1, workers=8)

model.save(model_path)



# #model = Word2Vec(sentences, size=200, window=5, min_count=1,workers=8)
# #模型储存与加载1

# model=Word2Vec.load(path)
# vec = model.wv['8']
# print("行为-8-的向量表示为：")
# print(vec)
# print("与-8-语义相近的行为是：")
# for key in model.similar_by_word('8',topn=3):
#         print(key)




