# -*- encoding: utf-8 -*-
'''
@File    :   mmoe.py
@Time    :   2021/06/08 22:25:27
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from model.baseMT import BaseMT
from utils.util import load_ocr
import torch
import torch.nn as nn
import collections


class MMOE(BaseMT):

    def __init__(self, config, feat_list, task_num=4, expert_num=8):
        
        super().__init__(config)
        self.task_num = task_num

        # 构建embedding字典
        input_size = 0
        self.EMdict = nn.ModuleDict({})
        for feat in feat_list:
            if feat.feat_name == 'item_ocr':
                self.EMdict[feat.feat_name] = nn.Embedding.from_pretrained(load_ocr())
                self.EMdict[feat.feat_name].requires_grad = False
            else:
                self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
                nn.init.normal_(self.EMdict[feat.feat_name].weight, mean=0.0, std=0.0001)
            input_size += feat.embedding_dim

        self.experts = nn.ModuleList([])
        for _ in range(expert_num):
            self.experts.append(Expert(input_size))

        self.gates = nn.ModuleList([])
        self.towers = nn.ModuleList([])
        for _ in range(task_num):
            self.gates.append(Gate(input_size, expert_num))
            self.towers.append(Tower(50))


    def forward(self, x):

        # 获取embedding特征
        EMlist = []
        for key in x.keys():
            EMlist.append(self.EMdict[key](x[key]))
        em_out = torch.cat(EMlist, dim=1)

        expert_out = []
        for expert in self.experts:
            expert_out.append(expert(em_out))

        gate_list = []
        for gate in self.gates:
            gate_list.append(gate(em_out))

        y = []
        for i, gateW in enumerate(gate_list):
            task_in = []
            # 对不同expert输出的特征进行加权平均
            for j, exo in enumerate(expert_out):
                task_in.append(gateW[:, j].unsqueeze(1) * exo) # gateW(bs, expert_num), exo(bs, v_dim)
            task_in = torch.stack(task_in, dim=2)
            task_in = torch.sum(task_in, dim=2)
            y.append(self.towers[i](task_in))
            y[i] = y[i].float()
        #y = torch.cat(y, dim=1) # shape(bs, task_num)

        return y


class Expert(nn.Module):

    def __init__(self, input_size, layer_list=[50, 50], output=False):
        
        super().__init__()
        
        seq_dict = collections.OrderedDict()
        last_unit = input_size  # 记录上一层的单元数, 初始化为输入单元数
        
        # 根据每层的单元数进行创建
        for i, layer_unit in enumerate(layer_list):
            seq_dict['LE' + str(i)] = nn.Linear(last_unit, layer_unit)
            seq_dict['actE' + str(i)] = nn.ReLU()
            last_unit = layer_unit
        
        # 创建最后一层
        if output:
            seq_dict['LEO'] = nn.Linear(last_unit, 1)

        seq_dict['LLast'] = nn.Linear(50, 50)

        self.expert = nn.Sequential(seq_dict)

    def forward(self, x):

        y = self.expert(x)

        return y


class Gate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class Tower(nn.Module):

    def __init__(self, input_size, layer_list=[50, 50]):

        super().__init__()
        last_unit = input_size
        seq_dict = collections.OrderedDict()
        # 根据每层的单元数进行创建
        for i, layer_unit in enumerate(layer_list):
            seq_dict['LT' + str(i)] = nn.Linear(last_unit, layer_unit)
            seq_dict['actT' + str(i)] = nn.ReLU()
            last_unit = layer_unit
        seq_dict['LTO'] = nn.Linear(last_unit, 1)
        
        self.tower = nn.Sequential(seq_dict)
        self.actTO = nn.Sigmoid()

    def forward(self, x):

        y = self.tower(x)
        y = self.actTO(y)

        return y


