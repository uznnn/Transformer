# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:14:35 2022

@author: weizh
"""

import torch
import plotly.graph_objs as go
from torch.utils.data import DataLoader
from Datas.train_def import Dataset_Smoke, doc_path, paras_count

def train_model_reg(net, num_epoch, 
                dataset_train_X, dataset_train_Y, dataset_val_X, dataset_val_Y,
                batch_size, loss_func, optimizer, val_data_size,mask): #create_mask(200, 3, 0.15,3)
  
    '''
    Dataset_Smoke—>>dataloader
    '''
    #np.random.seed(123)
    #torch.manual_seed(112)
    
    paras_count(net) #显示可训练参数数量
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    dataset_train = Dataset_Smoke(dataset_train_X, dataset_train_Y)
    dataset_val = Dataset_Smoke(dataset_val_X, dataset_val_Y)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, collate_fn=None) 
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=None)

    train_ls, val_ls = [], []
  
    for epoch in range(num_epoch):
        print("=========The {} Epoch Training==========".format(epoch+1))
        net.train()
        #total_train_loss = 0
        for data in train_loader:
            noise_mask = torch.tensor(mask.complt_mask()).to(device)
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            features = features * noise_mask
            outputs = net(features)
            outputs = outputs.to(torch.float64)
            labels = labels.to(torch.float64)
            loss_train = loss_func(outputs, labels,(1- noise_mask).bool())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            #print(loss_train.item())
        print("epoch & train_loss: {:.2f}".format(loss_train.item())) #loss_train.item()
        train_ls.append(loss_train.item())


        net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                noise_mask = torch.tensor(mask.complt_mask()).to(device)
                features, labels = data
                features = features.to(device)
                labels = labels.to(device)
                features = features * noise_mask
                outputs = net(features)
                outputs = outputs.to(torch.float64)
                labels = labels.to(torch.float64)
                loss_val = loss_func(outputs, labels, (1-noise_mask).bool())

                total_val_loss =  total_val_loss + loss_val.item()
                


            print("Epoch & Val_loss: {:.2f}".format(total_val_loss/len(val_loader)))    
            val_ls.append(total_val_loss/len(val_loader))
            #valid_accuracy.append(total_accuracy.item()/val_data_size)
            if val_ls[-1] == min(val_ls):
                torch.save(net, doc_path('trans_reg.pre'))
    # Visualization of Result
    x_axis = [i for i in range(1, num_epoch+1)]
    train_loss_line = go.Scatter(x=x_axis, y=train_ls, mode='lines+markers', name='train loss')
    val_loss_line = go.Scatter(x=x_axis, y=val_ls, mode='lines+markers', name='validation loss')
    data = [train_loss_line, val_loss_line]
    fig = go.Figure(data=data)
    fig.update_layout(
        title='Visualization of Loss', xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()

# 定义一个函数,可以用于分类训练模型
def train_model_class(net, num_epoch, dataset_train_X, dataset_train_Y, dataset_val_X, dataset_val_Y,
        batch_size, loss_func, optimizer, val_data_size):
  
    '''
    根据定义的Dataset_Smoke类,得到相应的dataloader
    该Function可以用于别的数据,但是要更改Dataset子类
    '''
    #np.random.seed(123)
    #torch.manual_seed(112)
    
    paras_count(net) #显示可训练参数数量
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    dataset_train = Dataset_Smoke(dataset_train_X, dataset_train_Y)
    dataset_val = Dataset_Smoke(dataset_val_X, dataset_val_Y)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, collate_fn=None) 
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, collate_fn=None)
    train_ls, val_ls ,acc= [], [], []
    for epoch in range(num_epoch):
    
        print("------第{}轮训练开始------".format(epoch+1))

        net.train()
        for data in train_loader:
            features, labels = data
            features = features.to(device)
            labels = labels.to(torch.long).to(device)
            outputs = net(features)
            loss_train = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        print("epoch & train_loss:", loss_train.item())
        train_ls.append(loss_train.item())


        net.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in val_loader:
                features, labels = data
                features = features.to(device)
                labels =  labels.to(torch.long).to(device)
                outputs = net(features)
                loss_val = loss_func(outputs, labels)

                total_val_loss = total_val_loss + loss_val.item()
                accuracy = (outputs.argmax(1) == labels).sum()
                total_accuracy = total_accuracy + accuracy


            print("整体测试集上的正确率: {}".format(total_accuracy/val_data_size))
            print("epoch & val_loss:", loss_val.item())    
            val_ls.append(loss_val.item())
            acc.append(total_accuracy.item()/val_data_size)
            
    # 模型结果可视化
    x_axis = [i for i in range(1, num_epoch+1)]
    train_loss_line = go.Scatter(x=x_axis, y=train_ls, mode='lines+markers', name='train loss')
    val_loss_line = go.Scatter(x=x_axis, y=val_ls, mode='lines+markers', name='validation loss')
    acc_line = go.Scatter(x=x_axis, y=acc, mode='lines+markers', name='Accuracy')
    data = [train_loss_line, val_loss_line,acc_line]
    fig = go.Figure(data=data)
    fig.update_layout(
        title='Visualization of Loss/Acc', xaxis_title='epoch', yaxis_title='loss')
    fig.show()
    with open('C:/Users\weizh/桌面/Transformer_spyder/record.txt','r') as g:
        read_doc = g.readlines()
        print(read_doc)
    with open('C:/Users\weizh/桌面/Transformer_spyder/record.txt',"w") as f:
        read_doc.append(str(max(acc))+'\n')
        for i in read_doc:
            f.write(i)