import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch
from torch.nn import Linear, ReLU, Parameter, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F
import math

class GraphSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, batchnorm_dim, dropout):
        super().__init__()
        
        self.mlp = Sequential(Linear(input_dim, hidden_dim), Dropout(dropout),
                              ReLU(), BatchNorm1d(batchnorm_dim),
                              Linear(hidden_dim, hidden_dim), Dropout(dropout), 
                              ReLU(), BatchNorm1d(batchnorm_dim))

        self.linear = Linear(hidden_dim, hidden_dim)
        
        self.eps = Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_eps = 0.25 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward(self, A, X):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        #if torch.isnan(A).any():
            #print("Input A has NaN")
        #if torch.isnan(X).any():
            #print("Input X has NaN")
        #user_input = input("请输入数字1以继续运行: ")
        batch, N = A.shape[:2]
        mask = torch.eye(N).unsqueeze(0)
        mask = mask.to(A.device)
        batch_diagonal = torch.diagonal(A, 0, 1, 2)
        #print("batch_diagonal device:", batch_diagonal.device)
        #print("mask device:", mask.device)
        #print("batch_diagonal before eps:", batch_diagonal)
        batch_diagonal = self.eps * batch_diagonal
        #print("batch_diagonal after eps:", batch_diagonal)
        #if torch.isnan(batch_diagonal).any():
            #print("batch_diagonal contains NaN")
        #else:
            #print("batch_diagonal is OK")

        #user_input = input("请输入数字2以继续运行: ")
        
        A = mask*torch.diag_embed(batch_diagonal) + (1. - mask)*A
        #if torch.isnan(A).any():
            #print("Modified A has NaN")
        #user_input = input("请输入数字3以继续运行: ")
        
        X_temp = A @ X
        #if torch.isnan(X_temp).any():
            #print("Result of A @ X has NaN")
        #else:
            #print("A @ X stats: min {:.4f}, max {:.4f}".format(X_temp.min().item(), X_temp.max().item()))

        #user_input = input("请输入数字4以继续运行: ")
        X_mlp = self.mlp(X_temp)
        #if torch.isnan(X_mlp).any():
            #print("Output of mlp has NaN")
        #user_input = input("请输入数字5以继续运行: ")
        X_linear = self.linear(X_mlp)
        
        
        #if torch.isnan(X_linear).any():
            #print("Output of linear has NaN")
        #user_input = input("请输入数字6以继续运行: ")
        X_relu = F.relu(X_linear)
        #if torch.isnan(X_relu).any():
            #print("Output of ReLU has NaN")
        #user_input = input("请输入数字7以继续运行: ")
        return X_relu

        #X = self.mlp(A @ X)
        #X = self.linear(X)
        #X = F.relu(X)

        #return X


