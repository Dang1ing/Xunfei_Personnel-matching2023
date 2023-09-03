'''
Author: zhuhui 2123535613@qq.com
Date: 2023-09-01 14:45:55
LastEditors: zhuhui 2123535613@qq.com
LastEditTime: 2023-09-01 14:45:58
FilePath: /Xunfei2023/layers/_loss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- encoding: utf-8 -*-
'''
@File    :   _loss.py
@Time    :   2021/04/26 21:09:36
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class bpr_loss(nn.Module):
    '''
    Bayesian Personalized Ranking Loss
    loss = - (y_pos - y_neg).sigmoid().log().sum()
    '''
    def __init__(self):
        super(bpr_loss, self).__init__()

    def forward(self, y_ui, y_uj):
        y_uij = y_ui - y_uj     # shape: (bs, 1)
        y_uij = F.sigmoid(y_uij)
        y_uij = torch.log(y_uij + 1e-5)
        loss = -y_uij.squeeze(1).sum()
        return loss


class MT_Loss(nn.Module):
    '''
    Loss function for multi-task.
    '''
    def __init__(self, loss_type):
        super(MT_Loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, y_pred, y_true, weights):
        loss = 0
        for i, w in enumerate(weights):
            if self.loss_type == 'bce':
                criterion = nn.BCELoss()
                y = y_true[:, i]
                y_ = y_pred[i].squeeze()
                loss += w * criterion(y_, y)
        return loss





