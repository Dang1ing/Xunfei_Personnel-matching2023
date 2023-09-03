'''
Author: zhuhui
Date: 2023-09-01 16:18:39
LastEditors: zhuhui
LastEditTime: 2023-09-02 23:24:20
Description: 
Copyright (c) 2023 by zhuhui, All Rights Reserved. 
'''
import configparser

from utils.selection import *
from utils.util import set_seed, load_pretrain
from layers.input import sparseFeat
from utils.generator import DataGenerator
# import setproctitle
import json
import sys


def main(config,mode='offline'):
    data_generator = DataGenerator(1024,8)
    feat_list = data_generator.get_feature_info()


    model = select_model()(config=config, feat_list=feat_list,embedding_size=256,num_classes=20)  # 默认加载deep_fm
    model.fit(data_generator,mode)   


def to_json(config, res_dict):
    params = {'lr': config.getfloat('Train', 'lr'),
              'bs': config.getint('Train', 'batch_size'),
              'embedding_dim': config.getint(config['Model']['model'], 'embedding_dim'),
              'l2': config.getfloat('Train', 'l2'),
              'optimizer': config.get('Train', 'optimizer')}
    res_dict['params'] = params
    file_name = './log/json/' + config['Model']['model'] + '/' + \
                'lr' + config.get('Train', 'lr') + \
                '_bs' + config.get('Train', 'batch_size') + \
                '_em' + config.get(config['Model']['model'], 'embedding_dim') + \
                '_l2' + config.get('Train', 'l2') + '.json'
    with open(file_name, 'w') as f:
        json.dump(res_dict, f)




if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./config.ini', encoding='utf-8')
    main(config)



