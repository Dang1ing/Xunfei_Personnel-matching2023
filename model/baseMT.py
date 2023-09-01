# -*- encoding: utf-8 -*-
'''
@File    :   basemodel.py
@Time    :   2021/05/24 22:42:18
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import os
import time
import logging
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import selection
from layers import _loss
from sklearn.metrics import roc_auc_score


WEIGHTS = {'read_commet': 0.4, 'like': 0.3, 'click_avatar': 0.2, 'forward': 0.1}
MultiWeights = [1, 1, 1, 1]


class BaseMT(nn.Module):
    '''
    The BaseModel of all of models.

    '''
    def __init__(self, config, loss='bce', best_iteration=0) -> None:
        super().__init__()
        
        self.config = config
        self.loss = loss
        self.best_iteration = best_iteration     # the iteration is used for test 
        self.metrics = {}
        #TODO: 日志模块进行修改
        self._init_log()

    
    def fit(self, data_generator, mode='offline'):
        model = self.train()
        self._initialize_parameters(model)  # initialize parameters in model

        self.logger.info('************** Training **************')
        
        '''Load validation and test data'''
        test_data = data_generator.make_test_loader()

        optim_ = selection.select_optimizer(self.config.get('Train', 'optimizer'))
        schedule_ = selection.select_schedule(self.config.get('Train', 'scheduler'))
        
        optimizer = optim_(params=filter(lambda x: x.requires_grad, model.parameters()), 
                           lr=self.config.getfloat('Train', 'lr'),
                           weight_decay=self.config.getfloat('Train', 'l2'))
        if schedule_ is not None:
            scheduler = schedule_(optimizer, 
                                  gamma=self.config.getfloat('Train', 'lr_decay'), 
                                  last_epoch=-1)
       
        main_metric = []
        for epoch in range(self.config.getint('Train', 'epoch')):

            '''load dataset for bpr or not'''
            train_loader = data_generator.make_train_loader()

            self.logger.info('====Train Epoch: %d/%d====' % (epoch + 1, self.config.getint('Train', 'epoch')))
            train_loss, train_auc = [], []
            i = 0
            t_loss, t_auc = 0, 0    # clear up for each epoch
            '''train part'''
            for batch in tqdm(train_loader):
                '''the loss for point-wise and bpr is different'''
                x, y = self._move_device(batch[0]), self._move_device(batch[1])
                y = y.squeeze()
                y_ = model(x)
                criterion = _loss.MT_Loss(self.loss)
                loss = criterion(y_, y.float(), MultiWeights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_loss += loss.item() / self.config.getint('Log', 'batch_record')
                #t_auc += self._compute_auc(y, y_) / self.config.getint('Log', 'batch_record')

                i += 1
                if not i % self.config.getint('Log', 'batch_record'):
                    train_loss.append(t_loss), train_auc.append(t_auc)
                    self.writer.add_scalar('Train/Loss', t_loss, 
                                            round(i/self.config.getint('Log', 'batch_record')))
                    #self.writer.add_scalar('train/AUC', t_auc, 
                    #                        round(i/self.config.getint('Log', 'batch_record')))
                    self.logger.info('Epoch %d:(%d) Train Loss: %.5f' 
                                     % (epoch+1, i, t_loss))
                    t_loss, t_auc = 0, 0
                

            '''validation part'''
            if mode == 'offline':
                self.logger.info('************** Evaluating **************')
                uauc, uauc_dict = self.evaluate(test_data)
                self.writer.add_scalar('Test/uAUC', uauc, epoch+1)
                self.logger.info('uAUC for each target: ' + str(uauc_dict))
                self.logger.info('Test uAUC: %.5f' % uauc)
                main_metric.append((uauc, epoch+1))
            self._save_checkpoint(epoch+1)  # save checkpoint
        
        '''get best iteration according to main metric on validation'''
        if mode == 'offline':
            main_metric = sorted(main_metric, key=lambda x: x[0], reverse=True)
            self.best_iteration = main_metric[0][1]
            self.best_metric = main_metric[0][0]
            self.logger.info('The best iteration is %d', self.best_iteration)
        
        self._end_log()

    
    def _initialize_parameters(self, model, init_std=0.0001):
        '''
        Initialize each layer of the model according to various type
        of layer.
        '''
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
            #if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, mean=0, std=0.01)

    
    def _print_grad(self, model):
        '''Print the grad of each layer'''
        for name, parms in model.named_parameters():
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		                    ' -->grad_value:',parms.grad)

    
    def _init_log(self):
        '''
        Initialize the logging module. Concretely, initialize the
        tensorboard and logging
        '''
        # judge whether the folder exits
        if not os.path.exists(r'./log/text/' + self.config['Model']['model'] + '/'):
            os.makedirs(r'./log/text/' + self.config['Model']['model'] + '/')

        # get the current time string
        now_str = time.strftime("%m%d%H%M%S", time.localtime())

        '''Initialize tensorboard. Set the save folder.'''
        if self.config.getboolean('Log', 'log'):
            folder_name = './log/tensorboard/' + self.config['Model']['model'] + '/' + now_str + '/'
        else:
            folder_name = folder_name = './log/tensorboard/' + self.config['Model']['model'] + '/default/'
        self.writer = SummaryWriter(folder_name)

        '''Initialize logging. Create console and file handler'''
        self.logger = logging.getLogger('train')
        self.logger.setLevel(logging.DEBUG)  # must set
        # create file handler
        if self.config.getboolean('Log', 'log'):
            log_path = './log/text/'+ self.config['Model']['model'] + '/' + now_str + '.txt'
            fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fm = logging.Formatter("%(asctime)s-%(message)s")
            fh.setFormatter(fm)
            self.logger.addHandler(fh)

            # record the hyper parameters in the text
            self.logger.info('learning rate: ' + str(self.config.get('Train', 'lr')))
            self.logger.info('learning rate decay: ' + str(self.config.get('Train', 'lr_decay')))
            self.logger.info('batch size: ' + str(self.config.get('Train', 'batch_size')))
            self.logger.info('optimizer: ' + str(self.config.get('Train', 'optimizer')))
            self.logger.info('scheduler: ' + str(self.config.get('Train', 'scheduler')))
            
        #create console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.logger.addHandler(self.ch)


    def _end_log(self):
        '''
        End the logging module
        '''
        self.writer.close()
        self.logger.removeHandler(self.ch)


    def _save_checkpoint(self, epoch):
        '''save checkpoint for each epoch'''
        folder_path = r'./save_model/MT/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.state_dict(), folder_path + 'epoch_' + str(epoch) + '.ckpt')

    
    def _load_model(self):
        '''load model at best iteration'''
        check_dir = r'./save_model/'
        best_path = check_dir + 'epoch_' + str(self.best_iteration) + '.ckpt'
        self.load_state_dict(torch.load(best_path))
        #TODO:删除所有checkpoint,并把最好的模型存储下来

        '''remove all checkpoint files'''
        #for f in os.listdir(check_dir):
        #    os.remove(check_dir + f)

        '''save the best model'''
        torch.save(self.state_dict(), './save_model/' + self.config['Model']['model'] + '.pt')

    
    def _move_device(self, data):
        '''
        move data to specified device.
        '''
        #TODO: 可以重写,换用tensor.to()会更方便
        if self.config.getboolean('Device', 'cuda'):
            if isinstance(data, dict):
                for key in data:
                    data[key] = data[key].cuda(self.config.getint('Device', 'device_tab'))
            else:
                data = data.cuda(self.config.getint('Device', 'device_tab'))
            
        return data


    def evaluate(self, testloader):
        model = self.eval()
        user, pred, true = [], [], []
        for batch in tqdm(testloader):
            with torch.no_grad():
                x, y = self._move_device(batch[0]), batch[1]
                y_ = model(x)
                # 拼接不同目标的预测向量
                y_ = torch.cat(y_, dim=1)

                user.append(x['user_id'])
                pred.append(y_)
                true.append(y)
        
        # 将不同batch的向量拼接起来
        user = torch.cat(user, dim=0)
        pred = torch.cat(pred, dim=0)
        true = torch.cat(true, dim=0)
        pred = pred.to('cpu').squeeze().detach().numpy()
        true = true.to('cpu').squeeze().detach().numpy()
        user = user.to('cpu').squeeze().detach().tolist()

        uauc = 0
        uauc_dict = {}
        for i, key in enumerate(WEIGHTS.keys()):
            eval_df = pd.DataFrame({
                'user_id': user,
                'pred': pred[:, i],
                'true': true[:, i]
            })

            def get_auc(x):
                if (1 in x['true'].values) & (0 in x['true'].values):
                    auc = roc_auc_score(x['true'], x['pred'])
                    return pd.DataFrame({'user_id': [x['user_id'].iloc[0]], 
                                        'auc': [auc], 
                                        'flag': [1]})
                else:
                    return pd.DataFrame({'user_id': [x['user_id'].iloc[0]], 
                                        'auc': [0], 
                                        'flag': [0]})
            eval_df = eval_df.groupby('user_id').apply(get_auc)
            eval_df = eval_df.loc[eval_df['flag']==1]
            uauc += WEIGHTS[key] * eval_df['auc'].mean()
            uauc_dict[key] = eval_df['auc'].mean()
        return uauc, uauc_dict

    
    def _compute_auc(self, true, pred):
        true = true.to('cpu').detach().numpy()
        pred = pred.to('cpu').detach().numpy()
        auc = roc_auc_score(true, pred)
        return auc


    def save_best_model(self):
        '''只存储最优的模型'''
        folder_path = r'./save_model/MT/'
        os.rename(folder_path + 'epoch_' + str(self.best_iteration) + '.ckpt', folder_path + self.config.get('Model', 'target') + '.ckpt')
        all_file = os.listdir(folder_path)

        '''删除其他的模型'''
        for filename in all_file:
            if 'epoch_' in filename:
                os.remove(folder_path + filename)
