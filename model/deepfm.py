# -*- encoding: utf-8 -*-
'''
@File    :   deepfm.py
@Time    :   2021/05/24 23:05:22
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from collections import OrderedDict
from model.basemodel import BaseModel
from model.baseMT import BaseMT
from utils.util import load_pretrain, load_ocr
import torch
import torch.nn as nn


class DeepFM(BaseModel):
    def __init__(self, config, feat_list, pretrain=False):
        super(DeepFM, self).__init__(config, pretrain=pretrain)
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            if feat.feat_name == 'item_ocr':
                self.EMdict[feat.feat_name] = nn.Embedding.from_pretrained(load_ocr())
                self.EMdict[feat.feat_name].requires_grad = False
            else:
                self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
                self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
                nn.init.normal_(self.FMLinear[feat.feat_name].weight, mean=0.0, std=0.0001)
                nn.init.normal_(self.EMdict[feat.feat_name].weight, mean=0.0, std=0.0001)
            input_size += feat.embedding_dim
        
        self.dnn = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            #('BN1', nn.BatchNorm1d(200, momentum=0.5)),
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1, bias=False))
        ]))
        
        self.out = nn.Sigmoid()

    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for key in x.keys():
            if key != 'item_ocr':
                EMlist.append(self.EMdict[key](x[key]))
                fmlinear += self.FMLinear[key](x[key])  # (bs, 1)
        
        
        '''FM'''
        in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim)
        sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim)
        yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        yFM += fmlinear

        '''DNN'''
        if 'item_ocr' in x.keys():
            EMlist.append(self.EMdict['item_ocr'](x['item_ocr']))
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = self.dnn(in_dnn) # (bs, 1)

        y = self.out(yFM + yDNN)

        return y.float()


class DeepFM_MT(BaseMT):
    def __init__(self, config, feat_list, task_num=4):
        super(DeepFM_MT, self).__init__(config)
        self.task_num = task_num

        '''只共享embedding层'''
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
            input_size += feat.embedding_dim

        '''构建不同task私有的参数层'''
        self.dnnList = nn.ModuleList([])
        self.outList = nn.ModuleList([])

        for _ in range(task_num):
            self.dnnList.append(nn.Sequential(OrderedDict([
                        ('L1', nn.Linear(input_size, 400)),
                        ('act1', nn.ReLU()),
                        ('L2', nn.Linear(400, 400)), 
                        ('act2', nn.ReLU()),
                        ('L3', nn.Linear(400, 1))
            ])))

            self.outList.append(nn.Sigmoid())


    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for key in x.keys():
            EMlist.append(self.EMdict[key](x[key]))
            fmlinear += self.FMLinear[key](x[key])  # (bs, 1)

        '''FM'''
        in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim)
        sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim)
        yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        yFM += fmlinear

        '''DNN'''
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = []
        for task in range(self.task_num):
            yDNN.append(self.dnnList[task](in_dnn)) # (bs, 1)

        y = []
        for task in range(self.task_num):
            y.append(self.outList[task](yFM + yDNN[task]))
            y[task] = y[task].float()

        return y


