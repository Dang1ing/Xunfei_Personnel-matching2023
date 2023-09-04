# -*- encoding: utf-8 -*-
'''
@File    :   generator.py
@Time    :   2021/05/21 20:23:18
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
import pandas as pd
import numpy as np
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import  MultiLabelBinarizer,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class OfflineData(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        data = self.x.iloc[index].to_dict()
        for key,value in data.items():
            data[key] = torch.tensor(value,dtype=torch.float32)
            # data[key] = torch.tensor(value)
        return data, torch.tensor(self.y[index],dtype=torch.float32)
        # return data, torch.tensor(self.y[index])


class OnlineData(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


class DataGenerator():
    def __init__(self, batch_size, num_workers):
        self.bs = batch_size
        self.num_workers = num_workers

        self.filename = '../Data/encode.csv'
        self.Encoder_result = {}

        self._load_data()   # 加载数据集
        self._preprocess()  # 处理数据集


    def _load_data(self):
        '''加载数据集'''
        self.CSVreader = pd.read_csv(self.filename)

    
    def _split_data(self):
        '''线下测评的话, 分割数据集, 用分割出的测试集代替加载的擦拭及'''
        self.test = self.train.loc[self.train['date_']==14]
        self.train = self.train.loc[self.train['date_']<14]

    
    def _preprocess(self):
        '''对数据进行预处理'''
        # 岗位ID onehot编码
        data = self.CSVreader['岗位ID'].to_list()
        self.encoder = LabelBinarizer()
        self.Encoder_result['岗位ID'] = self.encoder.fit_transform(data).tolist()

        field_list = set(self.CSVreader.columns.to_list()) - {'岗位ID'}
        # for field in field_list:
        #     self.Encoder_result[field] = self.CSVreader[field].to_list()
        for field in field_list:
            data = self.CSVreader[field].to_list()
            encoder = LabelBinarizer()
            self.Encoder_result[field] = encoder.fit_transform(data).tolist()

        # LabelEncoder编码
        # field1_list = {'工作经历数量','社会经历数量','项目数量','技能数量','荣誉数量'}

        # for field in field1_list:
        #     data = self.CSVreader[field].to_list()
        #     encoder = LabelBinarizer()
        #     self.Encoder_result[field] = encoder.fit_transform(data).tolist()
        
        # MulitHot编码
        # field2_list = set(self.CSVreader.columns.to_list()) - field1_list - {'岗位ID'}
        # pca = PCA(1024)
        # for field in field2_list:
        #     data = self.CSVreader[field]
        #     label_record = []
        #     for item in data:
        #         item = item.replace("'",'').replace("[",'').replace("]",'').split(',')
        #         label_record.append(item)
        #     encoder = MultiLabelBinarizer()
        #     self.Encoder_result[field] = encoder.fit_transform(label_record).tolist()
        #     if len(self.Encoder_result[field][0]) > 1024:
        #         self.Encoder_result[field] = pca.fit_transform(self.Encoder_result[field]).tolist()
        
        self.data = pd.DataFrame(self.Encoder_result)
        # print(self.data)
        # sys.exit()
        self.train, self.test = train_test_split(self.data, train_size=0.75)
        self.data.drop(columns='岗位ID',inplace=True)
    
    def label_inverse_transform(self,y):
        label_list = []
        length = len(y[0])
        for item in y:
            item = item.tolist()
            tmp = [0]*length
            max_index = item.index(max(item))
            # print(max_index)
            tmp[max_index] = 1
            label_list.append(tmp)
            # print(tmp)
        label_list = self.encoder.inverse_transform(np.array(label_list))
        # print("label list")
        # for item in label_list:
            # print(item)
        return label_list


    def get_feature_info(self):
        '''获取feature的信息,名称：数据长度'''
        vocabulary_size = {}
        for feat in self.data.columns:
            # if feat == '岗位ID':
            #     vocabulary_size[feat] = len(self.data[feat][0])
            # else:
            #     vocabulary_size[feat] = 1
            vocabulary_size[feat] = len(self.data[feat][0])
        return vocabulary_size

    
    def make_train_loader(self):
        # 将datafram格式的self.train构造为样本对 
        x_ = self.train.drop(columns='岗位ID')
        y_ = self.train['岗位ID']
        traindata = OfflineData(x_,y_)
        return DataLoader(traindata, 
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=self.num_workers)


    def make_test_loader(self):
        # 同样的dataframe格式self.test
        x_ = self.test.drop(columns='岗位ID')
        y_ = self.test['岗位ID']
        testdata = OfflineData(x_,y_)
        return DataLoader(testdata,batch_size=len(y_)) # 一次加载完整


    def make_pretrain_loader(self):
        '''将所有目标的正样本作为正样本进行预训练'''
        label = None
        for target in TARGET:
            label += self.train[target]
        pretraindata = OfflineData(torch.tensor(self.train[self.features].values), 
                                   torch.tensor(label.values))
        return DataLoader(pretraindata, 
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: collate_feat(x, self.features))
    

    def make_pretest_loader(self):
        '''构建预训练任务的测试集'''
        label = None
        for target in TARGET:
            label += self.train[target]
        pretraintest = OfflineData(torch.tensor(self.train[self.features].values), 
                                   torch.tensor(label.values))
        return DataLoader(dataset=pretraintest,
                          batch_size=pretraintest.__len__(),
                          collate_fn=lambda x: collate_feat(x, self.features, self.mode))



def collate_feat(data, features=['user_id', 'item_id'], mode='offline', 
                 multi_task=False):
    '''
    聚合样本, 把一个batch数据组合成字典形式
    线下数据: (bs, 2)-->(x, y)
    线上数据: (bs)-->(x)
    '''
    batch_data = {}

    if mode == 'offline':
        x = list(map(lambda x: x[0], data)) # 取出全部特征数据
    elif mode == 'online':
        x = data

    x = torch.stack(x)  # (bs, feat_num)
    for i, feat in enumerate(features):
        batch_data[feat] = x[:, i].long()

    if mode == 'offline':
        y = list(map(lambda x: x[1], data))
        y = torch.stack(y)  # (bs, 1)
        y = y.unsqueeze(1)
        return batch_data, y
    
    elif mode == 'online':
        return batch_data



