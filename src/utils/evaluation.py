# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2021/05/25 21:19:31
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from sklearn.metrics import roc_auc_score

class Evaluator():
    def __init__(self, y_true, y_pred,):
        self.y_true = self.y_true.to('cpu').detach().numpy()
