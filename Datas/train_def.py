# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:29:08 2022

@author: weizh
"""
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import sys,os

# sliding_window处理deep_learning数据
'''
之后这里添加滑窗的步长
'''
def doc_path(doc_name):
    __file__ = sys.argv[0]
    __root__ = os.path.dirname(os.path.realpath(__file__))     #获得所在脚本路径
    __path__ = os.path.realpath(os.path.join(__root__,"{}".format(doc_name)))
    return __path__

def paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print('{} KB Parameters ROM:'.format(round(total_trainable_params*4/1024)))
    #return total_params, total_trainable_params

class sliding_window():
    def __init__(self,X,y,length_window):
        self.X = X
        self.y = y
        self.length_window = length_window
    def slid_win_deepl_X(self):
        emp_list_X = []  
    
        len_data = self.X.shape[0]
    
        # 去掉不满足一个sliding window长度的数据
        # 不去掉数据最后shape就会变成一维
        if len_data/self.length_window != 0:  
            len_times = len_data//self.length_window
            len_data_new = self.length_window*len_times
        else:
            len_data_new = len_data
    
        for i in range(0,len_data_new,self.length_window):
            X_slid_win = self.X.iloc[i:i+self.length_window,:].values
            emp_list_X.append(X_slid_win)
            np_X = np.array(emp_list_X)
        return np_X
    
    def slid_win_deepl_y(self):
        emp_list_y = []
        len_data = self.y.shape[0]
        
        if len_data/self.length_window != 0:   
            len_times = len_data//self.length_window
            len_data_new = self.length_window*len_times
        else:
            len_data_new = len_data
    
        for i in range(0,len_data_new,self.length_window):
            y_slid_win = self.y.iloc[i:i+self.length_window,:].mode().values.tolist()[0]
            emp_list_y = emp_list_y + y_slid_win  
            np_y = emp_list_y
        return np_y 
    
class create_mask():
    def __init__(self,L,lm,masking_ratio,feat):
        self.L = L #L: length of mask and sequence to be masked
        self.lm = lm #average length of masking subsequences (streaks of 0s) 
        self.masking_ratio = masking_ratio #proportion of L to be masked
        self.feat = feat # #生成mask特征数
    def geom_noise_mask_single(self):
        """
        Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
        proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
        
        """
        keep_mask = np.ones(self.L, dtype=bool)
        p_m = 1 / self.lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * self.masking_ratio / (1 - self.masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > self.masking_ratio)  # state 0 means masking, 1 means not masking
        for i in range(self.L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state
        return keep_mask
    def complt_mask(self):
        fmask =self.geom_noise_mask_single().reshape(-1, 1)
        for i in range(self.feat-1):
            _mask_ = self.geom_noise_mask_single()
            _mask_ = _mask_.reshape(-1, 1)
            fmask= np.hstack((fmask,_mask_))
        return fmask.astype(np.float32)  #  boolean numpy array intended to mask ('drop') with 0s a sequence of length L

# 这里面X,Y输入的是滑窗后的信息
# 用于一维数据,输出[200,3]
# 这里可以直接交给Dataloader，抛出给训练器
class Dataset_Smoke(Dataset):
    def __init__(self,X,Y):
        self.data = X
        self.label = Y
        self.data_len = len(Y)  # 计算length
        # self.transformations = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self,index):
        slid_win_data = self.data[index]
        features = slid_win_data
        features = torch.from_numpy(features).float()     #torch.from_numpy()用来将数组array转换为张量Tensor

        slid_win_label = self.label[index] 
        # labels = self.transformations(slid_win_label)
        labels = np.array(slid_win_label)
        return features, labels

    def __len__(self):
        return self.data_len
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered
        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        return self.mse_loss(masked_pred, masked_true)
