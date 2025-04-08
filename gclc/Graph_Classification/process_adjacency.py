import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

def process_adjacency_matrices(datareader):
    """
    处理数据集中的邻接矩阵，生成增强的邻接矩阵
    Args:
        datareader: 包含图数据的读取器对象，需要包含adj_list属性
    """
    dataset_length = len(datareader.data['adj_list'])
    
    for itr in tqdm(range(dataset_length), desc="Processing graphs", unit="graph"):
        # 获取每个图的邻接矩阵
        A_array = datareader.data['adj_list'][itr]
        G = nx.from_numpy_array(A_array)

        sub_graphs = []
        subgraph_nodes_list = []
        sub_graphs_adj = []
        sub_graph_edges = []
        new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])
        
        # 对每个图生成子图：每个节点与其连接的邻居构成一个子图
        for i in np.arange(len(A_array)):
            s_indexes = []
            for j in np.arange(len(A_array)):
                s_indexes.append(i)
                if A_array[i][j] != 0:
                    s_indexes.append(j)
            sub_graphs.append(G.subgraph(s_indexes))
    
        # 获取每个子图的节点列表
        for i in np.arange(len(sub_graphs)):
            subgraph_nodes_list.append(list(sub_graphs[i].nodes))
    
        # 获取每个子图的邻接矩阵
        for index in np.arange(len(sub_graphs)):
            sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
    
        # 统计每个子图的边数量
        for index in np.arange(len(sub_graphs)):
            sub_graph_edges.append(sub_graphs[index].number_of_edges())
    
        # 利用子图信息构造新的邻接矩阵 new_adj
        for node in np.arange(len(subgraph_nodes_list)):
            sub_adj = sub_graphs_adj[node]
            for neighbors in np.arange(len(subgraph_nodes_list[node])):
                index = subgraph_nodes_list[node][neighbors]
                count = torch.tensor(0).float()
                if index == node:
                    continue
                else:
                    c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                    if index in c_neighbors:
                        nodes_list = subgraph_nodes_list[node]
                        c_neighbors_list = list(c_neighbors)
                        for i, item1 in enumerate(nodes_list):
                            if item1 in c_neighbors:
                                for item2 in c_neighbors_list:
                                    j = nodes_list.index(item2)
                                    count += sub_adj[i][j]
    
                    new_adj[node][index] = count / 2
                    new_adj[node][index] = new_adj[node][index] / (len(c_neighbors) * (len(c_neighbors) - 1))
                    new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)
    
        weight = torch.FloatTensor(new_adj)
        weight = weight / weight.sum(1, keepdim=True)
        weight = weight + torch.FloatTensor(A_array)
    
        coeff = weight.sum(1, keepdim=True)
        coeff = torch.diag((coeff.T)[0])
    
        weight = weight + coeff
    
        weight = weight.detach().numpy()
        weight = np.nan_to_num(weight)
    
        datareader.data['adj_list'][itr] = weight