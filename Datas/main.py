# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:45:03 2022

@author: weizh
"""

#import numpy as np 
import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt
import sys,re,os
from sklearn import preprocessing
from Datas.train_def import *

def doc_path(doc_name):
    __file__ = sys.argv[0]
    __root__ = os.path.dirname(os.path.realpath(__file__))     #获得所在脚本路径
    __path__ = os.path.realpath(os.path.join(__root__,"{}".format(doc_name)))
    return __path__

class Data_Input():
    def __init__(self,window_length):
        self.window_length = window_length
    def input_wisdm(self): 
        columns = ['user','label','timestamp', 'x-axis', 'y-axis', 'z-axis']
        #os.path.abspath(os.path.dirname(os.getcwd())) 获取上一级文件夹
        path = doc_path('')+'\\WISDM_ar_v1.1\WISDM_ar_v1.1_raw.txt'
        df = pd.read_csv(path,header=None,names=columns)
        df["z-axis"]=df["z-axis"].replace('\;','',regex=True).astype(float) #清洗掉z-axis中的符号
        df.dropna()  #防止数据缺失，直接扔掉
        print(df.info())
        
        #发现有单一数据缺失，直接填充
        df["z-axis"].fillna(df["z-axis"].median(),inplace=True)
        #正则化
        df[['x-axis', 'y-axis', 'z-axis']]=(df[['x-axis', 'y-axis', 'z-axis']]-df[['x-axis', 'y-axis', 'z-axis']].mean())/(df[['x-axis', 'y-axis', 'z-axis']].std())
        plt.rcParams['font.sans-serif'] = ['SimHei']   ###为了方便图表显示中文
        plt.rcParams['axes.unicode_minus'] = False
        df['label'].value_counts().plot(kind='pie', title='每个动作的数据量')
        plt.show()
        
        #划分训练测试集
        user = list(set(df['user']))
        df_new_train = pd.DataFrame()
        df_new_test = pd.DataFrame()
        split = 31 # 取前31个人为训练集
        for i in user[:split]:
            a =df[df['user']==i]
            df_new_train = pd.concat([df_new_train,a])
        for i in user[split:]:
            b =df[df['user']==i]
            df_new_test = pd.concat([df_new_test,b])
        df_new_train = df_new_train[['x-axis', 'y-axis', 'z-axis','label']]
        df_new_test = df_new_test[['x-axis', 'y-axis', 'z-axis','label']]
        
        enc = preprocessing.LabelEncoder()
        enc=enc.fit(["Walking","Jogging","Upstairs","Downstairs","Sitting","Standing"])
        df_new_train["label"] = enc.transform(df_new_train["label"])
        df_new_test["label"] = enc.transform(df_new_test["label"])
        
        df_new_train_X = df_new_train[['x-axis', 'y-axis', 'z-axis']]
        df_new_train_y =  df_new_train[['label']]
        df_new_test_X =  df_new_test[['x-axis', 'y-axis', 'z-axis']]
        df_new_test_y =  df_new_test[['label']]
        
        sliding_window_train = sliding_window(df_new_train_X,df_new_train_y,self.window_length) 
        sliding_window_test = sliding_window(df_new_test_X,df_new_test_y,self.window_length) 
        X_train=y_train= sliding_window_train.slid_win_deepl_X()
         #= sliding_window_train.slid_win_deepl_y()
        X_test=y_test = sliding_window_test.slid_win_deepl_X()
         #= sliding_window_test.slid_win_deepl_y()
        return X_train ,y_train,X_test,y_test
    
    def input_uci_har(self):
        path = doc_path('')+'\\UCI HAR Dataset\\'
        #加载加速度传训练集
        filePath = path+'acc_train\\'
        acc_train_list = os.listdir(filePath)
        acc_train = pd.DataFrame()
        colnames = ['acc_x','acc_y','acc_z']
        for i in acc_train_list:
            mid = pd.read_csv(filePath+i,sep='\\s+',header=None,names=colnames)
            acc_train = pd.concat([acc_train,mid])
        #加载陀螺仪训练集
        filePath = path+'gyro_train\\'
        gyro_train_list = os.listdir(filePath)
        gyro_train = pd.DataFrame()
        colnames = ['gyro_x','gyro_y','gyro_z','label']
        for i in gyro_train_list:
            mid = pd.read_csv(filePath+i,sep='\\s+',header=None,names=colnames)
            gyro_train = pd.concat([gyro_train,mid])
        #加载加速度测试集
        filePath = path+'acc_test\\'
        acc_test_list = os.listdir(filePath)
        acc_test = pd.DataFrame()
        colnames = ['acc_x','acc_y','acc_z']
        for i in acc_test_list:
            mid = pd.read_csv(filePath+i,sep='\\s+',header=None,names=colnames)
            acc_test = pd.concat([acc_test,mid])
        #加载陀螺仪测试集
        filePath = path+'gyro_test\\'
        gyro_test_list = os.listdir(filePath)
        gyro_test = pd.DataFrame()
        colnames = ['gyro_x','gyro_y','gyro_z','label']
        for i in gyro_test_list:
            mid = pd.read_csv(filePath+i,sep='\\s+',header=None,names=colnames)
            gyro_test = pd.concat([gyro_test,mid])
        '''
        得到4个集合
        acc_train:  加速度训练集
        gyro_train: 陀螺仪训练集
        acc_test:   加速度测试集
        gyro_test:  陀螺仪测试集
        
        '''
        
        Train_reg = pd.concat([acc_train,gyro_train],axis=1) #Regression训练集所有数据
        Train_class = Train_reg.dropna(axis=0)               #Classify训练集只包含有标注数据
        Test_reg = pd.concat([acc_test,gyro_test],axis=1)    #..
        Test_class = Test_reg.dropna(axis=0)                 #..
        
        #产生种类的weights
        count=list(Train_class['label'].value_counts())
        def class_weight(ls):
            weights = []
            for i in ls:
                weights.append(0.5*sum(ls)/i)
            return weights
        weights = class_weight(count)
        
        X_train_reg = Train_reg[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']]
        y_train_reg = Train_reg[['label']]
        X_test_reg = Test_reg[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']]
        y_test_reg = Test_reg[['label']]
        
        #生成Regression Sliding Window集合
        sliding_window_train_reg = sliding_window(X_train_reg,y_train_reg,self.window_length) 
        sliding_window_test_reg = sliding_window(X_test_reg,y_test_reg,self.window_length) 
        X_train_reg=y_train_reg = sliding_window_train_reg.slid_win_deepl_X()
        X_test_reg=y_test_reg = sliding_window_test_reg.slid_win_deepl_X()
        
        X_train_class = Train_class[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']]
        y_train_class = Train_class[['label']]-1
        X_test_class = Test_class[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']]
        y_test_class = Test_class[['label']]-1
        
        #生成Classify Sliding Window集合
        sliding_window_train_classify = sliding_window(X_train_class, y_train_class, self.window_length) 
        sliding_window_test_classify = sliding_window(X_test_class, y_test_class, self.window_length) 
        
        X_train_class = sliding_window_train_classify.slid_win_deepl_X()
        y_train_class = sliding_window_train_classify.slid_win_deepl_y()
        X_test_class = sliding_window_test_classify.slid_win_deepl_X()
        y_test_class = sliding_window_test_classify.slid_win_deepl_y()
        
        #最终返回两个完整训练集列表
        return {'Regression':[X_train_reg,y_train_reg,X_test_reg,y_test_reg],'Classify':[X_train_class,y_train_class,X_test_class,y_test_class,weights]}
        
        

if __name__ == '__main__':
    data=Data_Input(128) #填写的是数据length
    '''
    X_train,y_train,X_test,y_test = data.input_wisdm()
    print(X_test[0].shape)
    '''
    uci_har = data.input_uci_har()
    print(uci_har['Regression'][0].shape)
    print(uci_har['Classify'][0].shape)