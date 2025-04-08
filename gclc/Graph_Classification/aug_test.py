import numpy as np
import os
import argparse
from data_reader import DataReader
from dataprocessor import DataProcessor
from aug import DataAugmentation
import numpy as np
import time
import networkx as nx
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import augmentors as A
import heapq as hp

from graph_data import GraphData
from data_reader import DataReader
from models import GNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import chain
from models import Encoder
from losses import SupConLoss
from sklearn import preprocessing
#from IPython.core.debugger import Tracer
from torch_geometric.utils import precision, recall, f1_score,true_positive, true_negative, false_positive, false_negative

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    os.chdir('/home/xuke/zxy_code/gclc/Graph_Classification')
    # 添加命令行参数
    parser.add_argument('--device', default='cuda', help='Select CPU/CUDA for training.')
    parser.add_argument('--dataset', default='Applications', help='Dataset name.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0025, help='Initial learning rate.')
    parser.add_argument('--wdecay', type=float, default=2e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of MLP layers for GraphSN.')
    parser.add_argument('--batchnorm_dim', type=int, default=30, help='Batchnormalization dimension for GraphSN layer.')
    parser.add_argument('--dropout_1', type=float, default=0.25, help='Dropout rate for concatenation the outputs.')
    parser.add_argument('--dropout_2', type=float, default=0.25, help='Dropout rate for MLP layers in GraphSN.')
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds in cross validation.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--log_interval', type=int, default=10 , help='Log interval for visualizing outputs.')
    parser.add_argument('--seed', type=int, default=117, help='Random seed.')

    args = parser.parse_args("")

    print('Loading data')

    datareader = DataReader(data_dir='./data/%s/' % args.dataset.upper(),
                            fold_dir=None,
                            rnd_state=np.random.RandomState(args.seed),
                            folds=args.n_folds,
                            use_cont_node_attr=False)

    print('Loading data completed')

        # 图的数量
    dataset_length = len(datareader.data['adj_list'])
    for itr in np.arange(dataset_length):
        # 每个图的矩阵
        A_array = datareader.data['adj_list'][itr]
        G = nx.from_numpy_matrix(A_array)

        sub_graphs = []
        subgraph_nodes_list = []
        sub_graphs_adj = []
        sub_graph_edges = []
        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
        
        # 每个图的子图
        for i in np.arange(len(A_array)):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                #if(A_array[i][j]==1):
                if(A_array[i][j]!=0):
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))

    
        # 每个图的每个子图的节点
        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))

        # 每个图的每个子图矩阵
        for index in np.arange(len(sub_graphs)):
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
        #print("sub_graphs_adj:", sub_graphs_adj)


        # 每个图的每个子图的边的数量
        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())

        # 每个图(包含每个图的子图)的新的矩阵
        for node in np.arange(len(subgraph_nodes_list)):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if(index==node):
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        sub_graph_index = nodes_list.index(index)
                        c_neighbors_list = list(c_neighbors)
                        #print(len(c_neighbors))
                        for i, item1 in enumerate(nodes_list):
                            if(item1 in c_neighbors):
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]

                    new_adj[node][index] = count / 2
                    new_adj[node][index] = new_adj[node][index]/(len(c_neighbors)*(len(c_neighbors)-1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)

        weight = weight + torch.FloatTensor(A_array)

        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag((coeff.T)[0])

        weight = weight + coeff

        weight = weight.detach().numpy()
        #weight = np.nan_to_num(weight, nan=0)
        weight = np.nan_to_num(weight)

        datareader.data['adj_list'][itr] = weight
    
    acc_folds = []
    #accuracy_arr = np.zeros((10, args.epochs), dtype=float)
    accuracy_arr = np.zeros((1, args.epochs), dtype=float)
    for fold_id in range(args.n_folds):
        device = torch.device(args.device)
        print('\nFOLD', fold_id)
        loaders = []
        for split in ['train', 'test']:
            gdata = GraphData(fold_id=fold_id,
                                datareader=datareader,
                                split=split)

            loader = torch.utils.data.DataLoader(gdata, 
                                                batch_size=args.batch_size,
                                                shuffle=split.find('train') >= 0,
                                                num_workers=args.threads)
            loaders.append(loader)
        user_input = input("请输入数字3以继续运行: ")   
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                                A.NodeDropping(pn=0.1),
                                A.FeatureMasking(pf=0.1),
                                A.EdgeRemoving(pe=0.1)], 1)
        gconv = GNN(input_dim=loaders[0].dataset.features_dim,
                        hidden_dim=args.hidden_dim,
                        output_dim=loaders[0].dataset.n_classes,
                        n_layers=args.n_layers,
                        batchnorm_dim=args.batchnorm_dim, 
                        dropout_1=args.dropout_1, 
                        dropout_2=args.dropout_2).to(args.device)
        
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)

        criterion = SupConLoss(temperature=0.07)
        #contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
        print('\nInitialize model')
        print(gconv)
        c = 0
        for p in filter(lambda p: p.requires_grad, encoder_model.parameters()):
            c += p.numel()
        print('N trainable parameters:', c)

        optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, encoder_model.parameters()),
                    lr=args.lr,
                    weight_decay=args.wdecay,
                    betas=(0.5, 0.999))
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)

        def train(train_loader):
            #scheduler.step()
            encoder_model.train()
            start = time.time()
            train_loss, n_samples = 0, 0
            for batch_idx, data in enumerate(train_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(args.device)
                optimizer.zero_grad()
                gs1, gs2 = encoder_model(data.x, data.edge_index)
                output = torch.cat((gs1, gs2), dim=0)
                #output = model(data)
                g1 = gs1.unsqueeze(1)
                g2 = gs2.unsqueeze(1)

                # 拼接这两个张量，结果的形状将是 [4, 2, 21]
                g = torch.cat([g1, g2], dim=1)

                print("拼接后的张量形状:", g.shape)
                loss=criterion(g, data[4])
                #loss = loss_fn(output, data[4])
                loss.backward()
                optimizer.step()
                time_iter = time.time() - start
                train_loss += loss.item() * len(output)
                n_samples += len(output)
                scheduler.step()
                if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                        epoch, n_samples, len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
                #scheduler.step()
        

if __name__ == "__main__":
    main()
