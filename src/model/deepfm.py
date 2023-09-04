'''
Author: zhuhui
Date: 2023-09-01 15:29:45
LastEditors: zhuhui 2123535613@qq.com
LastEditTime: 2023-09-03 17:18:39
Description: 
Copyright (c) 2023 by zhuhui, All Rights Reserved. 
'''
from collections import OrderedDict
from model.basemodel import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import sys


class DeepFM(BaseModel):
    '''
    config:
    feat_list:特征列表，每个元素为{"name":,size:}
    embedding_size:嵌入后的维度列表
    '''
    def __init__(self, config, feat_list, embedding_size,num_classes, pretrain=False):
        super().__init__(config, pretrain=pretrain)

        self.feat_list = feat_list
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        self.FMEmbedding = nn.ModuleDict({})
        self.FMLinears1 = nn.ModuleDict({})

        # print(self.feat_list)
        # sys.exit()
        
        input_size = len(feat_list)*embedding_size

        for feat_name,feat_size in feat_list.items():
            self.FMEmbedding[feat_name] = nn.Linear(feat_size,self.embedding_size) # 通用embedding层
            self.FMLinears1[feat_name] = nn.Linear(self.embedding_size,self.num_classes) # FM一阶特征获取
            # self.FMLinears2[feat_name] = nn.Linear(self.embedding_size,self.num_classes) # FM二阶特征获取
            nn.init.normal_(self.FMEmbedding[feat_name].weight,mean=0.0,std=0.0001)
            nn.init.normal_(self.FMLinears1[feat_name].weight,mean=0.0,std=0.0001)
        
        
        self.dnn = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            # ('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            # ('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, self.num_classes, bias=False))
        ]))
        
        self.out = nn.Softmax(dim=1)

    '''
    x表示输入数据 字典格式
    '''
    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for key in x.keys():
            x[key] = self.FMEmbedding[key](x[key])
            EMlist.append(x[key])
            
            
        for key in x.keys():
            fmlinear += self.FMLinears1[key](x[key])  # (bs, 20) 一阶特征  (bs,embedding) (bs,20)
            # EMlist.append(self.FMLinears2[key](x[key])) # 二阶特征
        
        # EMlist (bs,feat_num,em_dim)
        '''FM'''
        # in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim) bs:batch size feat_num:特征数量 em_dim特征嵌入后的维度
        # square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim) 先求和，后平方
        # sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim) 先平方，后求和
        # yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        # yFM += fmlinear
        yFM = fmlinear
        '''DNN'''
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = self.dnn(in_dnn) # (bs, 20)

        y = self.out(yFM + yDNN)

        return y.float()

