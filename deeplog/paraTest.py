import torch
import torch.nn as nn
import time
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name,window_size):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    hdfs = []
    # hdfs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            line = [n for n in line if n != '']
            line = tuple(map(lambda n: n, map(int, line)))
            line = list(line) #+ [-1] * (window_size + 1 - len(line))
            hdfs.append(tuple(line))
            # hdfs.append(tuple(ln))
    #print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

res_mat=[]
res_mat.append("容忍度,候选数,ACC,FP, FN, P, R, F1")
def predict(torl_num,num_candidates):
    # Hyperparameters
    num_classes = 168
    input_size = 1
    model_path = r'E:\code\zhuheCode\act-Classification\model\1.31Adam_batch_size=1024_epoch=300.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=5, type=int)
    #parser.add_argument('-num_candidates', default=5, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    #num_candidates = args.num_candidates


    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    # test_normal_loader = generate(r'E:\code\祝贺论文相关\act-Classification\data\httpreduce_n.csv',window_size)
    # test_abnormal_loader = generate(r'E:\code\祝贺论文相关\act-Classification\data\httpreduce_an.csv',window_size)
    test_normal_loader = generate(r'E:\code\zhuheCode\act-Classification\data\httpreduce_n.csv',window_size)
    test_abnormal_loader = generate(r'E:\code\zhuheCode\act-Classification\data\httpreduce_an.csv',window_size)
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            error_count=0
            for i in range(window_size,len(line) - window_size):
                # seq = line[i - window_size:i] + line[i +1:i + window_size +1]
                # label = line[i]
                # for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                # if label not in predicted:
                #     FP += 1
                #     break
                #在正常数据中 检测不符合正常数据建模规律的数据 FP为实际为真 预测为假
                #实际情况中，序列长度不一致 不应该设定固定的阈值

                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    error_count+=1
                if error_count>=torl_num:
                    FP+=1
                    break
    with torch.no_grad():
        for line in test_abnormal_loader:
            error_count = 0
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                # if label not in predicted:
                #     TP += 1
                #     break
                if label not in predicted:
                    error_count+=1
                if error_count>=torl_num:
                    TP+=1
                    break
    elapsed_time = time.time() - start_time
    #print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    TN=len(test_normal_loader)-FP
    FN = len(test_abnormal_loader) - TP
    ACC=100*(TP+TN)/(FP+FN+TP+TN)
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('容忍度: {},候选数: {} '.format(torl_num,num_candidates))
    print('Acc: {}, false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(ACC,FP, FN, P, R, F1))
    #print('Finished Predicting')
    res_mat.append([torl_num,num_candidates,ACC,FP, FN, P, R, F1])
    print(res_mat)

if __name__=='__main__':
    for t in range(1,5):
        for c in range(1,10):
            predict(t,c)
    print(res_mat)