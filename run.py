# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:04:58 2022

@author: weizh
"""
from Model.transformer import Transformer_Smoke,Transformer_Classify_Pre,Transformer_Classify
from Datas.train_def import create_mask,MaskedMSELoss
from Datas.main import Data_Input
import torch
import torch.nn as nn
from training import train_model_reg,train_model_class

d_model = 6  # 输入特征尺寸3
class_num = 12 # 分类的类别是多少类
max_len = 128  #sliding_window长度
d_ff = 512  # FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），最后会再接一个projection层
d_k = d_v = 128  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 3   # number of Encoder Layer（Block的个数，这里7个效果最好）
n_heads = 8  # number of heads in Multi-Head Attention（有几个头）
Mask = create_mask(max_len, 3, 0.15, d_model)
input_ls = [d_model,class_num,max_len,d_ff,d_k,d_v,n_layers,n_heads,Mask]
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")


class Run(object):
    
    def __init__(self,ls,lr_reg=0.00015,lr_class=0.00008,
                 batch_size = 128,reg_epoch=60,class_epoch=100):
        
        self.d_model,self.class_num,self.max_len,self.d_ff,self.d_k,self.d_v,self.n_layers,self.n_heads,self.Mask = ls
        self.lr_reg = lr_reg
        self.lr_class = lr_class
        self.batch_size = batch_size
        self.reg_epoch = reg_epoch
        self.class_epoch = class_epoch
        self.uci_har = Data_Input(self.max_len).input_uci_har()
        with open('C:/Users\weizh/桌面/Transformer_spyder/record.txt',"w") as f:
            f.write(str(ls[:-1])+'\n')    
            
    def regression(self):  
        
        #加载训练数据
        X_train_reg,y_train_reg,X_test_reg,y_test_reg = self.uci_har['Regression']
        
        # 创建训练模型
        transformer_smoke = Transformer_Smoke(self.d_model,self.d_k,self.d_v,self.n_heads,self.d_ff,self.max_len,self.n_layers).to(device)
        loss_func_dcl = MaskedMSELoss().to(device) # 空格中加入torch可使用张量
        optimizer_dcl = torch.optim.Adam(transformer_smoke.parameters(), lr=self.lr_reg)
        
        #开始训练
        train_model_reg(net=transformer_smoke, num_epoch=self.reg_epoch, batch_size=self.batch_size, 
                    dataset_train_X=X_train_reg, dataset_train_Y=y_train_reg, 
                    dataset_val_X=X_test_reg, dataset_val_Y=y_test_reg,     
                    loss_func=loss_func_dcl,optimizer=optimizer_dcl, 
                    val_data_size=len(y_test_reg),mask=Mask)
    
    def classification(self, pre=True):
        
        #加载训练数据
        X_train_class,y_train_class,X_test_class,y_test_class,weights = self.uci_har['Classify']
        
        #创建\加载训练模型
        if pre:
            transformer_classify = Transformer_Classify_Pre(self.class_num,self.max_len,self.d_model).to(device)
        else:
            transformer_classify = Transformer_Classify(self.class_num,self.max_len,self.d_model,
                                                        self.d_k,self.d_v,self.n_heads,self.d_ff,self.n_layers).to(device)
        # 定义训练参数
        class_weights = torch.FloatTensor(weights).to(device)
        loss_func_class = nn.CrossEntropyLoss(weight=class_weights).to(device) # 空格中加入torch可使用张量
        optimizer_class = torch.optim.Adam(transformer_classify.parameters(), lr=self.lr_class)
        
        #开始训练
        train_model_class(net=transformer_classify, num_epoch=self.class_epoch, batch_size=self.batch_size, 
                    dataset_train_X=X_train_class, dataset_train_Y=y_train_class, 
                    dataset_val_X=X_test_class, dataset_val_Y=y_test_class,     
                    loss_func=loss_func_class,optimizer=optimizer_class, val_data_size=len(y_test_class))

if __name__ == '__main__':
    running = Run(input_ls)
    running.regression()
    for i in range(10):
        running.classification(pre=True)