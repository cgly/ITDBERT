# ITDBERT: An insider threat detection Framework based on sequence semantics
This is a PyTorch implementation of [ITDBERT: Temporal-semantic Representation for Insider Threat Detection](https://ieeexplore.ieee.org/document/9631538).
Extracting user behaviour semantics with Temporal information embedded 
## Required packages
- python 3.7 
- numpy
- sklearn
- torch 1.7.0+cu110
- Keras 2.3.0

## Usage

```
# Prd-Train and insider threat detection：
# MaskedLM +(emb/CLS)
python trans_corpus.py
python robertaLSTM.py

# word2vec + (TextCNN/BiLSTM_ATT/RCNN/Transformer)
python Mygensim.py
python run.py

#Autoregressive models(deeplog)
python ActModel_train.py
```

## Visualize the embedding results of word2vec via tsne
![image](https://github.com/cgly/ITDBERT/blob/main/tsne%20.png)

## Files
- data: the dataset after data preprocessing
- deeplog: the implementation of  deeplog model
- model: models architecture and hyperparameter setting
- preTrain: pretrain model 
- save_model: save place
- 42cert2Corpus.py: Preprocessing of cert datasets
- robertaLSTM.py: run MaskedLM+LSTM for insider threat detection
- run.py: run training models for insider threat detection
- utils.py: utils for training
