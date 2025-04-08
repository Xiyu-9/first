import torch.nn as nn
import torch.nn.functional as F
from layers import GraphSN
import torch
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import numpy as np
class GNN(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batchnorm_dim, dropout_1, dropout_2):
        super().__init__()
        
        self.dropout = dropout_1   
 
        self.convs = nn.ModuleList()    
        self.convs.append(GraphSN(input_dim, hidden_dim, batchnorm_dim, dropout_2))
        for _ in range(n_layers-1):
            self.convs.append(GraphSN(hidden_dim, hidden_dim, batchnorm_dim, dropout_2))
        
        project_dim = input_dim+hidden_dim*(n_layers)
      
        self.instance_projector = torch.nn.Sequential(
            torch.nn.Linear(project_dim, project_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(project_dim, input_dim),
        )
        
        
        self.cluster_projector = nn.Linear(project_dim, output_dim)
        
        self.out_proj = nn.Linear((input_dim+hidden_dim*(n_layers)), output_dim)

    def forward(self, data):
        X=data[0]
        A=data[1]

        hidden_states = [X]
        
        for layer in self.convs:
            #out = layer(A, X)
            #if torch.isnan(out).any():
                #print("Layer output has NaN")
            X = F.dropout(layer(A, X), self.dropout)
            hidden_states.append(X)
        X = torch.cat(hidden_states, dim=2).sum(dim=1)
        #X = self.out_proj(X)
        #print("X.shape:", X.shape)

        return X


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor


    def forward(self, data):
        X=data[0]
        A=data[1]

        #print("=== 特征矩阵 X 的信息 ===")
        #print("维度:", X.shape if hasattr(X, 'shape') else "未知")

        #print("\n=== 邻接结构 A 的信息 ===")
        #print("张量形状:", A.shape)
        #user_input = input("请输入数字2以继续运行: ")
        aug1, aug2 = self.augmentor
        x1, edge_index1 = aug1(X, A )
        x2, edge_index2= aug2(X, A )
        
        # 构造增强后的数据字典
        data1 = data.copy()
        data2 = data.copy()
        data1[0] = x1
        data1[1] = edge_index1
        data2[0] = x2
        data2[1] = edge_index2
        
        g = self.encoder(data)
        g1 = self.encoder(data1)
        g2 = self.encoder(data2)
        
        return g, g1, g2

    def forward_cluster(self, x):
        
        g = self.encoder(x)
        c = self.encoder.out_proj(g)
        #c = torch.argmax(c, dim=1)
        return c