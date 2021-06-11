import csv
import numpy
import pandas as pd
from tqdm import tqdm
import os
import copy
## 数据清洗：

'''
上次清洗的cert4.2不带有子行为信息，所以这次重新清洗数据;将清洗记录保存：
1. 恶意行为抽取

1. ShowFirst5Col 展示前五行
2. dropoutInfo（删除不使用的属性
3. 清洗
    1. timeProcess（将不同的日志形式翻转
    2. httpReduce（将HTTP数据按一定规则缩减
4. select_item_useFirstChar按首字母抽取排序
    1. 子行为抽取
5. mergeData合并排序结果
6. orderedCert2Seq 将排序后的数据转化为带名称和tag的序列
7. 用户数据——>gensim corpus   所有用户的数据——>train data  采样数据——>test data
'''

act_dict = {
    'Logon': 1,
    'Logoff': 2,
    'http': 3,
    'email': 4,
    'file': 5,
    'Connect': 6,
    'Disconnect': 7
}

def time_convet(timestr,action):
    time_list=timestr.split(":")
    hour=int(time_list[0])
    minute=int(time_list[1])
    offset=60*hour+minute
    res=(action-1)*1440+offset+1
    return res

def ShowFirst5Col(path,isshowFirst5):
    pd.set_option('display.max_columns', None)#使pandas的输出显示所有列
    if isshowFirst5:
        data=pd.read_csv(path,nrows=10)
    else:
        data=pd.read_csv(path)
    print((data))

#删除不需要的列
def saveDf2csv(path,newPath):
    data = pd.read_csv(path)
    data = data.drop(['id','pc','content','filename'], axis=1)
    data.to_csv(newPath)


#发现42没有多少子行为可以加 所以找到以前的42数据，将异常用户的历史数据全部清除掉 再训练词向量
def delMulseq(MilFile_Path,oriFile_Path,tarFile_Path):
    with open(MilFile_Path, 'r+') as read:
        reader = csv.reader(read)
        MilUser = set()
        for line in reader:
            MilUser.add(line[0])
    with open(oriFile_Path, 'r+') as read:
        with open(tarFile_Path, 'w', newline='') as write:
            reader = csv.reader(read)
            csv_writer = csv.writer(write)
            count=0
            for line in reader:
                if line[0] not in MilUser:
                    csv_writer.writerow(line)
                else:
                    count+=1
            print(count)

def httpreduce(oriFile_Path,tarFile_Path,timestamp,mul_sum):
    #http 2881-4320
    new_seq=[]
    with open(oriFile_Path, 'r+') as read:
        with open(tarFile_Path, 'w', newline='') as write:
            reader = csv.reader(read)
            csv_writer = csv.writer(write)
            for line in reader:
                line=[i for i in line if i!='']
                new_seq=[]
                for j,item in enumerate(line):
                    if j<3:
                        new_seq.append(item)
                        continue
                    item=int(item)
                    if item>=2881 and item<=4320:
                        #把http数据缩减后添加到new_seq中
                        if ishttpStart:
                            cur_http=item
                            new_seq.append(item)
                            ishttpStart=0
                        else:
                            if item-cur_http>=timestamp:
                                cur_http = cur_http+timestamp
                                new_seq.append(cur_http)
                    else:
                        if int(line[j-1])>=2881 and int(line[j-1])<=4320 and int(line[j-1])!=cur_http:
                            new_seq.append(int(line[j-1]))
                        ishttpStart=1
                        new_seq.append(item)
                csv_writer.writerow(new_seq)

def get24corpus(oriFile_Path,tarFile_Path):
    with open(oriFile_Path, 'r+') as read:
        with open(tarFile_Path, 'w', newline='') as write:
            reader = csv.reader(read)
            csv_writer = csv.writer(write)
            for line in reader:
                seq=map(lambda x:int(int(x)/60),line)
                csv_writer.writerow(list(seq))

def recordMil(file_path,target_path):
    milRes=[]
    print(os.listdir(file_path))
    for file in os.listdir(file_path):
        cur_file=file_path+'\\'+file
        with open(cur_file, 'r+') as f:
            reader = csv.reader(f)
            for i in reader:
                #10/23/2010 01:34:19
                date = i[2]
                date=date[6:10]+date[5]+date[:5]+date[10:]
                print(date)
                action = [date, i[3]]
                if i[0]=='device' or i[0]=='logon':
                    action.append(i[5])
                else:
                    action.append(i[0])
                milRes.append(action)
    with open(target_path,'w',newline='') as w:
        writer=csv.writer(w)
        for i in milRes:
            writer.writerow(i)
    return milRes

def milAct2Num(file_path,target_path):
    res=[]
    with open(file_path, 'r+') as f:
        reader = csv.reader(f)
        for i in reader:
            date=i[0].split(' ')[0]
            name=i[1]
            num=time_convet(i[0].split(' ')[1],act_dict[i[2]])
            action=[date,name,num]
            res.append(action)
    with open(target_path,'w',newline='') as w:
        writer=csv.writer(w)
        for i in res:
            writer.writerow(i)

def mergeUser(file_path,target_path):
    with open(file_path, 'r+') as f:
        with open(target_path, 'w', newline='') as w:
            reader = csv.reader(f)
            writer = csv.writer(w)
            cur_date='2010/6/28'
            cur_name='AAF0535'
            action=[cur_date,cur_name]
            for i in reader:
                if cur_date==i[0] and cur_name==i[1]:
                    action.append(i[2])
                else:
                    writer.writerow(action)
                    cur_date = i[0]
                    cur_name = i[1]
                    action = [cur_date, cur_name,i[2]]
            writer.writerow(action)

'''
2021年4月7日11:05:18
'''
def testData(file_path,target_path):
    action=[]
    res=[]
    count=0
    with open(file_path, 'r+') as f:
        with open(target_path, 'r+') as w:
            reader = csv.reader(f)
            writer = csv.reader(w)
            for i in reader:
                action.append([i[0],i[1]])
            for j in writer:
                res.append([j[1],j[0]])
            for k in action:
                if k in res:
                    count+=1
                    print(k)
            #ret = [i for i in res if i not in action]
            #ret=list(set(action) ^ set(res))
            print(count)

def compareData(file_path,tarFilePath,save_path):
    count=0
    with open(tarFilePath, 'r+') as f:
        with open(file_path, 'r+') as w:
            with open(save_path,'w',newline='') as s:
                save=csv.writer(s)
                dayBehavior = csv.reader(f)
                milBehavior = csv.reader(w)
                for _ in range(986):
                    day_lsit=[]
                    day=next(dayBehavior)
                    day=[i for i in day if i!='']
                    #print(day)
                    mil=next(milBehavior)
                    mil = [i for i in mil if i != '']
                    if day[0] != mil[1] or day[1] != mil[0]:
                        break
                    for index,i in enumerate(day):
                        if index<3:
                            day_lsit.append(i)
                            continue
                        if isHTTp(int(i)):
                            print("ok")
                            pass
                        if i in mil[2:]:
                            i='-'+i
                            day_lsit.append(i)
                            count+=1
                        else:
                            day_lsit.append(i)
                    save.writerow(day_lsit)
    print(count)
def sameDetect(file_path):
    count = 0
    with open(file_path, 'r+') as f:
        reader = csv.reader(f)
        for line in reader:
            count_set=set()
            for i,data in enumerate(line):
                count_set.add(data)
            count+=(len(line)-len(count_set))
    print(count)

#7323

'''
现在需要对带标签的行的http进行合并 
合并规则：当合并的标签中出现一个恶意的HTTp 那将这个标签置为恶意
'''
def tag_httpreduce(oriFile_Path,tarFile_Path,timestamp,mul_sum):
    #http 2881-4320
    new_seq=[]
    with open(oriFile_Path, 'r+') as read:
        with open(tarFile_Path, 'w', newline='') as write:
            reader = csv.reader(read)
            csv_writer = csv.writer(write)
            for line in reader:
                line=[i for i in line if i!='']
                new_seq=[]
                for j,item in enumerate(line):
                    if j<3:
                        new_seq.append(item)
                        continue
                    item=int(item)
                    if item>=2881 and item<=4320:
                        #把http数据缩减后添加到new_seq中
                        if ishttpStart:
                            cur_http=item
                            new_seq.append(item)
                            ishttpStart=0
                        else:
                            if item-cur_http>=timestamp:
                                cur_http = cur_http+timestamp
                                new_seq.append(cur_http)
                    else:
                        if int(line[j-1])>=2881 and int(line[j-1])<=4320 and int(line[j-1])!=cur_http:
                            new_seq.append(int(line[j-1]))
                        ishttpStart=1
                        new_seq.append(item)
                csv_writer.writerow(new_seq)


def isHTTp(num):
    num=int(num)
    if num>= 2881 and num <= 4320:
        return True
    else:
        return False

def tag_compareData(file_path,tarFilePath,save_path):
    count=0
    with open(tarFilePath, 'r+') as f:
        with open(file_path, 'r+') as w:
            with open(save_path,'w',newline='') as s:
                save=csv.writer(s)
                dayBehavior = csv.reader(f)
                milBehavior = csv.reader(w)
                for i in range(986):
                    day_lsit=[]

                    day=next(dayBehavior)
                    day=[i for i in day if i!='']
                    #print(day)
                    mil=next(milBehavior)
                    mil = [i for i in mil if i != '']

                    if day[0] != mil[1] or day[1] != mil[0]:
                        break
                    day_copy=copy.deepcopy(day)
                    for index,data in enumerate(mil):
                        if index<2:
                            continue

                        cur_num=int(data)
                        if isHTTp(cur_num):
                            day_num = list(map(int, day[3:]))
                            day_num = list(map(lambda x: abs(abs(x) - cur_num), day_num))
                            ind=day_num.index(min(day_num))+3
                            day_copy[ind]='-'+str(abs(int(day_copy[ind])))
                        else:
                            day_num = list(map(int, day[3:]))
                            ind = day_num.index(cur_num) + 3
                            day_copy[ind]='-'+str(abs(int(day_copy[ind])))
                            #str(abs(int(day_copy[day_num.index(data)+3])))
                        count+=1
                    save.writerow(day_copy)
        print(count)
def selectMilItem(file_path,tarFilePath):
    with open(file_path, 'r+') as w:
        with open(tarFilePath, 'w+', newline='') as s:
            save = csv.writer(s)
            reader = csv.reader(w)
            for line in reader:
                line=[i for i in line if i!='']
                wList=[]
                if str(line[1]).startswith("2010"):
                    continue
                for index,i in enumerate(line):

                    if index<2:
                        wList.append(i)
                        continue
                    if int(i)<0:
                        wList.append(0.2)
                    else:
                        wList.append(0.01)
                new_seq=[]
                for i in range(67):
                    if i<len(wList):
                        new_seq.append(wList[i])
                    else:
                        new_seq.append(0.01)
                save.writerow(new_seq)

def getMilData(file_path,tarFilePath):
    with open(file_path, 'r+') as w:
        with open(tarFilePath, 'w+', newline='') as s:
            save = csv.writer(s)
            reader = csv.reader(w)
            for line in reader:
                line = [i for i in line if i != '']
                wList = []
                if str(line[1]).startswith("2010"):
                    continue
                for index, i in enumerate(line):
                    if index<2:
                        continue
                    if index==2:
                        wList.append(1)
                        continue
                    wList.append(int(int(i)/60))
                save.writerow(wList)

if __name__=='__main__':
    '''
    logon.csv
    device.csv
    email.csv
    file.csv
    http.csv
    '''
    # srcFilePath=r'E:\cert42\oridata\file.csv'
    # tarFilePath=r'E:\cert42\dateTmp\file.csv'
    # ShowFirst5Col(srcFilePath,True)
    # saveDf2csv(srcFilePath,tarFilePath)
    # ShowFirst5Col(tarFilePath,True)

    # srcFilePath=r'E:\data\aecert\train_noral.csv'
    # tarFilePath=r'E:\data\aecert\cleanCert_no.csv'
    # MilFile_Path=r'E:\data\aecert\train_annoral.csv'
    # delMulseq(MilFile_Path, srcFilePath, tarFilePath)

    # srcFilePath=r'E:\code\zhuheCode\act-Classification\utils\train_annoral.csv'
    # tarFilePath=r'E:\code\zhuheCode\act-Classification\utils\Htrain_annoral.csv'
    # MilFile_Path=r'E:\data\aecert\train_annoral.csv'
    # httpreduce(srcFilePath, tarFilePath, 30, 0)

    # srcFilePath=r'E:\data\aecert\cleanCert_corpus.csv'
    # tarFilePath=r'E:\data\aecert\330Cert42_60corpus.csv'
    # get24corpus(srcFilePath, tarFilePath)

    # file_path=r'C:\Users\Rafael\Desktop\42answers\42all'
    # tarFilePath=r'C:\Users\Rafael\Desktop\42answers\milres.csv'
    # recordMil(file_path,tarFilePath)
    #
    # file_path = r'C:\Users\Rafael\Desktop\42answers\milres.csv'
    # tarFilePath = r'C:\Users\Rafael\Desktop\42answers\milnum.csv'
    # milAct2Num(file_path,tarFilePath)


    # file_path = r'C:\Users\Rafael\Desktop\42answers\milnum.csv'
    # tarFilePath = r'C:\Users\Rafael\Desktop\42answers\MergeMilnum.csv'
    # mergeUser(file_path,tarFilePath)

    # file_path = r'C:\Users\Rafael\Desktop\42answers\MergeMilnum.csv'
    # tarFilePath = r'C:\Users\Rafael\Desktop\42answers\train_annoral.csv'
    # testData(file_path,tarFilePath)


    # file_path = r'C:\Users\Rafael\Desktop\42answers\MergeMilnum.csv'
    # tarFilePath = r'C:\Users\Rafael\Desktop\42answers\train_annoral.csv'
    # save_path=r'C:\Users\Rafael\Desktop\42answers\42TA_tagtest.csv'
    # compareData(file_path,tarFilePath,save_path)

    #sameDetect(save_path)

    # file_path = r'C:\Users\Rafael\Desktop\42answers\MergeMilnum.csv'
    # tarFilePath = r'C:\Users\Rafael\Desktop\42answers\Htrain_annoral.csv'
    # save_path=r'C:\Users\Rafael\Desktop\42answers\42HAT.csv'
    # tag_compareData(file_path,tarFilePath,save_path)

    #
    # file_path = r'C:\Users\Rafael\Desktop\42answers\42HAT.csv'
    # tarFilePath = r'C:\Users\Rafael\Desktop\42answers\2011HAT.csv'
    # selectMilItem(file_path,tarFilePath)

    file_path = r'C:\Users\Rafael\Desktop\42answers\Htrain_annoral.csv'
    tarFilePath = r'C:\Users\Rafael\Desktop\42answers\test2011.csv'
    getMilData(file_path, tarFilePath)
