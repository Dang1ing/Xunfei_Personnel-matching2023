# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2021/05/30 20:26:30
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn
import pickle
import random
import pandas as pd
import numpy as np


def load_ocr():
    with open(r'./data/ocr_embedding_32.pkl', 'rb') as f:
        ocr_em = pickle.load(f)

    ocr_em = torch.tensor(ocr_em).float()
    return ocr_em


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrain(model, pretrain_model, model_path):
    '''加载预训练的Embedding层参数'''
    pretrain_model.load_state_dict(torch.load(model_path))

    # 筛选出符合条件的embedding层构成更新字典
    pretrain_dict = {k: v for k, v in pretrain_model.state_dict().items()
                     if ('item_ocr' not in k) & ('EMdict' in k) & ('FMLinear' in k)}
    
    # 加载目标模型的参数, 并进行更新(使用python字典的update方法)
    model_dict = model.state_dict()
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model

