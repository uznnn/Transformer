# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 21:27:04 2022

@author: weizh
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Datas.train_def import doc_path
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V): 
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        """
        可以直接去掉masked， 这里是optimal
        """
       
        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax    
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, dropout=0.1 ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
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


class MultiHeadAttention(nn.Module):
    """
    这个Attention类可以实现: Encoder的Self-Attention
    """

    def __init__(self,d_model,d_k,d_v,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)  
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #       线性变换         拆成多头

        """
        算出Q,K,V的值
        """
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成8维的，只是为了扩充mask
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V)
        # 将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 再做一个projection
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(device)(output + residual), attn

# 残差加LayerNorm
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
    
class Encoder(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads,d_ff, max_len,n_layers):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model, max_len)  # Transformer中位置编码为可训练位置编码
        self.layers = nn.ModuleList([EncoderLayer(d_model,d_k,d_v,n_heads,d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        # enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns
    
class Transformer_Smoke(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads,d_ff,max_len,n_layers):
        super(Transformer_Smoke, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.encoder = Encoder(d_model,d_k,d_v,n_heads,d_ff,max_len,n_layers)
        #self.projection = nn.Linear(self.max_len*d_model, class_num)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_outputs = enc_outputs.view(enc_outputs.shape[0],-1)   #self.max_len*self.d_model
        shou = enc_outputs.shape[0]
        return enc_outputs.view(shou,self.max_len,self.d_model)

class Transformer_Classify_Pre(nn.Module):
    
    def __init__(self, num_classes,max_len,d_model):
        super(Transformer_Classify_Pre, self).__init__()
        self.device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        net = torch.load(doc_path('trans_reg.pre'))
        net.classifier = nn.Sequential()    # 将分类层（fc）置空
        self.features = net.to(self.device)
        self.classifier = nn.Sequential(    # 定义一个卷积网络结构
            nn.Linear(max_len*d_model, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1).to(self.device)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

class Transformer_Classify(nn.Module):
    def __init__(self,num_classes,max_len,d_model,d_k,d_v,n_heads,d_ff,n_layers):
        super(Transformer_Classify, self).__init__()
        self.max_len = max_len
        self.encoder = Encoder(d_model,d_k,d_v,n_heads,d_ff,max_len,n_layers)
        self.projection = nn.Linear(self.max_len*d_model, num_classes)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_outputs = enc_outputs.view(enc_outputs.shape[0],-1) 
        enc_outputs = self.projection(enc_outputs)   
        return F.softmax(enc_outputs, dim=1)

    