#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import re
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
# import augmentors as A  # Temporarily disabled due to PyTorch Geometric dependencies
import heapq as hp
import pickle
import time

from evu import AverageMeter, ProgressMeter
from cluster_analysis import *
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from munkres import Munkres
from graph_data import GraphData
from data_reader import DataReader
from models import GNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import chain
from models import Encoder
from losses import SupConLoss
from losses import ClusterLoss
from losses import ImprovedClusterLoss
from novelty_detection_system import NoveltyDetectionSystem
from sklearn import preprocessing
#from IPython.core.debugger import Tracer
#from torch_geometric.utils import precision, recall, f1_score,true_positive, true_negative, false_positive, false_negative
from tqdm import tqdm
from graphaug import GraphAugmentor
import logging
import os
from datetime import datetime
# =============================================================================
# Experiment parameters
# =============================================================================
'''
----------------------------
Dataset  |   batchnorm_dim
----------------------------
MUTAG    |     28
PTC_MR   |     64
BZR      |     57
COX2     |     56
COX2_MD  |     36
BZR-MD   |     33
PROTEINS |    620
D&D      |   5748
'''

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='Select CPU/CUDA for training.')
parser.add_argument('--dataset', default='Applications', help='Dataset name.')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train.')
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
parser.add_argument('--used_loss', type=str, default="default", help='Type of loss function used.')

parser.add_argument('--entropy_weight', type=float, default=0.5, help='Weight for entropy term in improved cluster loss')
parser.add_argument('--repulsion_weight', type=float, default=1.0, help='Weight for repulsion term in improved cluster loss')
parser.add_argument('--cluster_temperature', type=float, default=0.1, help='Temperature for cluster similarity')
parser.add_argument('--cluster_loss_weight', type=float, default=2.0, help='Temperature for cluster loss weight')


# 新增：监控并记录不同类别簇之间的平均距离函数
def monitor_cluster_distances(epoch, gc, labels, save_dir="./cluster_distances/"):
    """监控并记录不同类别簇之间的平均距离"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    gc_normalized = F.normalize(gc, p=2, dim=1)
    
    # 计算特征之间的余弦相似度矩阵
    similarity_matrix = torch.mm(gc_normalized, gc_normalized.t())
    
    num_classes = labels.max().item() + 1
    class_distances = torch.zeros((num_classes, num_classes), device=gc.device)
    class_counts = torch.zeros((num_classes, num_classes), device=gc.device)
    
    # 聚合同类别间的相似度
    for i in range(len(labels)):
        for j in range(len(labels)):
            class_i = labels[i].item()
            class_j = labels[j].item()
            if i != j:  # 排除自身
                class_distances[class_i, class_j] += 1 - similarity_matrix[i, j]  # 使用距离 = 1-相似度
                class_counts[class_i, class_j] += 1
    
    # 避免除零
    class_counts[class_counts == 0] = 1
    avg_class_distances = class_distances / class_counts
    
    # 计算类间平均距离和最小距离
    mask = torch.ones_like(avg_class_distances, dtype=torch.bool)
    mask.fill_diagonal_(False)
    inter_class_distances = avg_class_distances[mask].view(num_classes, num_classes-1)
    
    avg_distance = inter_class_distances.mean().item()
    min_distance = inter_class_distances.min().item()
    
    # 记录到日志
    print(f"Epoch {epoch} - Avg Class Distance: {avg_distance:.4f}, Min Class Distance: {min_distance:.4f}")
    
    # 保存距离矩阵图
    if epoch % 5 == 0:  # 每5个epoch保存一次
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_class_distances.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Class Distance Matrix - Epoch {epoch}")
        plt.savefig(f'{save_dir}/class_distances_epoch_{epoch}.png')
        plt.close()
    
    return {
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'distance_matrix': avg_class_distances.cpu().numpy()
    }
def find_latest_encoder_checkpoint(fold_id):
    """查找特定fold_id的最新encoder模型检查点"""
    # 查找与当前fold_id相关的所有encoder模型文件
    pattern = f'encoder_model_fold_{fold_id}_epoch_*.pth'
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None, 0  # 没有找到检查点，从头开始训练
    
    # 从文件名中提取epoch编号并找出最大值
    epoch_numbers = []
    for file in model_files:
        match = re.search(r'epoch_(\d+)\.pth', file)
        if match:
            epoch_numbers.append((int(match.group(1)), file))
    
    if not epoch_numbers:
        return None, 0
    
    # 找出最大epoch编号的文件
    latest_epoch, latest_file = max(epoch_numbers, key=lambda x: x[0])
    return latest_file, latest_epoch


def find_improved_neighbors(g, gs1, gs2, labels):
    """
    从三个图集（原始图和两个增强版本）中选择正样本（同类）和负样本（不同类）邻居
    
    参数:
    g, gs1, gs2: 三个图集的特征表示，每个形状为 [batch_size, feature_dim]
    labels: 标签张量，形状为 [batch_size]
    
    返回:
    positive_neighbors: 每个样本的正邻居特征，形状为 [batch_size, feature_dim]
    negative_neighbors: 每个样本的负邻居特征，形状为 [batch_size, feature_dim]
    """
    device = g.device
    batch_size = g.size(0)
    
    # 合并所有特征
    all_features = torch.cat([g, gs1, gs2], dim=0)  # [3*batch_size, feature_dim]
    # 标签也要跟着复制三份
    all_labels = labels.repeat(3)  # [3*batch_size]
    
    pos_neighbors = []  # 同类正邻居
    neg_neighbors = []  # 不同类负邻居
    
    for i in range(batch_size):
        current_label = labels[i]
        current_embedding = g[i]
        
        # 创建掩码，排除自身
        not_self_mask = torch.ones(3*batch_size, dtype=torch.bool, device=device)
        not_self_mask[i] = False  # 排除原始图中的自身
        not_self_mask[i + batch_size] = False  # 排除gs1中对应的自身
        not_self_mask[i + 2*batch_size] = False  # 排除gs2中对应的自身
        
        # 找同类样本
        same_class_mask = (all_labels == current_label) & not_self_mask
        # 找不同类样本
        diff_class_mask = (all_labels != current_label)
        
        # 计算与所有样本的相似度 (使用余弦相似度)
        similarities = F.cosine_similarity(current_embedding.unsqueeze(0), all_features)
        
        # 选择同类最相似样本作为正邻居
        if same_class_mask.any():
            # 将不符合条件的位置设为很小的值
            valid_similarities = similarities.clone()
            valid_similarities[~same_class_mask] = -2.0  # 低于余弦相似度的最小值-1
            pos_idx = valid_similarities.argmax().item()
            pos_neighbors.append(all_features[pos_idx])
        else:
            # 极少数情况：没有同类样本，用自身作为正邻居
            pos_neighbors.append(current_embedding)
        
        # 选择不同类最相似样本作为"硬"负邻居（最具挑战性的负样本）
        if diff_class_mask.any():
            valid_similarities = similarities.clone()
            valid_similarities[~diff_class_mask] = -2.0
            neg_idx = valid_similarities.argmax().item()
            neg_neighbors.append(all_features[neg_idx])
        else:
            # 极少数情况：没有不同类样本，随机选择
            rand_idx = torch.randint(3*batch_size, (1,)).item()
            neg_neighbors.append(all_features[rand_idx])
    
    return torch.stack(pos_neighbors), torch.stack(neg_neighbors)
def find_nearest_neighbors(g, gs1, gs2, labels):
    """
    在三个图集中找到每个样本的最近邻居
    
    参数:
    g (torch.Tensor): 原始图特征，形状为 [batch_size, project_dim]
    gs1 (torch.Tensor): 增强1的图特征，形状为 [batch_size, project_dim]
    gs2 (torch.Tensor): 增强2的图特征，形状为 [batch_size, project_dim]
    labels (torch.Tensor): 标签，形状为 [batch_size]
    
    返回:
    neighbors (torch.Tensor): 邻居集，形状为 [batch_size, project_dim]
    """
    batch_size = g.size(0)
    device = g.device
    neighbors = torch.zeros_like(g)
    
    # 合并所有特征和标签
    all_features = torch.cat([g, gs1, gs2], dim=0)  # [3*batch_size, project_dim]
    all_labels = labels.repeat(3)  # 假设三个图集的标签相同
    
    # 计算特征之间的相似度矩阵 (使用余弦相似度)
    normalized_features = F.normalize(all_features, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_features[:batch_size], normalized_features.t())
    
    for i in range(batch_size):
        current_label = labels[i]
        
        # 创建掩码来筛选相同标签的样本
        same_label_mask = (all_labels == current_label)
        
        # 创建掩码来排除自身
        not_self_mask = torch.ones(3*batch_size, dtype=torch.bool, device=device)
        not_self_mask[i] = False  # 排除g中的自身
        
        # 结合掩码
        valid_mask = same_label_mask & not_self_mask
        
        # 检查是否有有效样本
        if not valid_mask.any():
            # 如果没有有效的邻居候选，使用自身作为邻居
            neighbors[i] = g[i]
            continue
        
        # 获取有效样本的相似度
        valid_similarities = similarity_matrix[i].clone()
        valid_similarities[~valid_mask] = -1  # 将无效样本的相似度设为-1
        
        # 找到最相似的样本索引
        most_similar_idx = valid_similarities.argmax().item()
        
        # 从对应的图集中获取邻居
        if most_similar_idx < batch_size:  # 来自g
            neighbors[i] = g[most_similar_idx]
        elif most_similar_idx < 2*batch_size:  # 来自gs1
            neighbors[i] = gs1[most_similar_idx - batch_size]
        else:  # 来自gs2
            neighbors[i] = gs2[most_similar_idx - 2*batch_size]
    
    return neighbors
# 设置日志配置
def setup_logger():
    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    log_file = 'train1.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
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

    # 保存处理后的数据集
    #with open(save_path, 'wb') as f:
    #    pickle.dump(datareader.data, f)
    #print(f'Processed data saved to {save_path}')

def evaluate(label, pred):
    print(f"真实类别数量: {len(set(label))}, 预测类别数量: {len(set(pred))}")
    print("真实标签 label:", np.unique(label))
    print("聚类结果 pred:", np.unique(pred))

    # 计算标准化互信息 (NMI)
    nmi = metrics.normalized_mutual_info_score(label, pred)
    # 计算调整兰德指数 (ARI)
    ari = metrics.adjusted_rand_score(label, pred)
    # 计算Fowlkes-Mallows指数 (F)
    f = metrics.fowlkes_mallows_score(label, pred)
    # 调整预测标签，使其与真实标签尽可能匹配
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    # 计算调整后的准确率
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc

def calculate_cost_matrix(C, n_clusters):
    """
    计算代价矩阵，用于解决最优匹配问题。
    C：混淆矩阵，表示聚类结果与真实标签的对应关系。
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    print("混淆矩阵:\n", C)
    print("代价矩阵:\n", cost_matrix)
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    """
    解析 Munkres 算法计算得到的最优匹配索引，返回最佳的簇标签对应关系。
    """
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    调整预测标签，使其与真实标签尽可能匹配。
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    print("最佳映射关系:", kmeans_to_true_cluster_labels)
    print("聚类分配的最大索引:", np.max(cluster_assignments))
    return y_pred

# 假设你已有的混淆矩阵绘制函数
def plot_confusion_matrix(cm, n_classes, epoch):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix at Epoch {epoch}')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()





def main():
    # 如果直接在命令行运行，则解析命令行参数；在 ipynb 中可以传入空列表
    args = parser.parse_args()
    print('Loading data')
    device = torch.device(args.device)
    print(f'Using device {device}')
    # 获取项目根目录（GCLC/）的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 关键修正！

    # 动态生成数据目录路径
    data_dir = os.path.join(project_root, 'Graph_Classification', 'data', 'APPLICATIONS')

    # 调试输出
    print(f"[DEBUG] 数据集绝对路径: {data_dir}")
    print(f"[DEBUG] 路径是否存在? {os.path.exists(data_dir)}")
    
    # 初始化 DataReader
    datareader = DataReader(
        data_dir=data_dir + '/',  # 确保以斜杠结尾
        fold_dir=None,
        rnd_state=np.random.RandomState(args.seed),
        folds=args.n_folds,
        use_cont_node_attr=False
    )
    # 构建数据读取器（DataReader）
    #datareader = DataReader(data_dir='./Graph_Classification/data/%s/' % args.dataset.upper(),
    #                        fold_dir=None,
    #                        rnd_state=np.random.RandomState(args.seed),
    #                        folds=args.n_folds,
    #                       use_cont_node_attr=False)
    print('Loading data completed')
    # =============================================================================
    # 对数据中每个图的邻接矩阵进行处理
    # =============================================================================

    #processed_data_path = './processed_data.pkl'
    
    # 检查是否存在已处理的数据文件
    #if os.path.exists(processed_data_path):
    #    print('Loading processed data')
    #    with open(processed_data_path, 'rb') as f:
    #        datareader.data = pickle.load(f)
    #    print('Loading processed data completed')
    #else:
    #    print('Processing graph data')
    #    process_adjacency_matrices(datareader, processed_data_path)
    #    print('Processing graph data completed')
    print('processing graph data')

    process_adjacency_matrices(datareader)

    print('processing graph data completed')

    # =============================================================================
    # 训练和测试部分
    # =============================================================================

    
    print('training and testing')
    acc_folds = []
    # accuracy_arr 的维度：折数 x epoch 数
    accuracy_arr = np.zeros((args.n_folds, args.epochs), dtype=float)
    
    for fold_id in range(args.n_folds):
        print('\nFOLD', fold_id)
        loaders = []
        for split in ['train', 'test']:
            gdata = GraphData(fold_id=fold_id,
                              datareader=datareader,
                              split=split)
    
            loader = torch.utils.data.DataLoader(gdata, 
                                                 batch_size=args.batch_size,
                                                 shuffle=('train' in split),
                                                 num_workers=args.threads)
            loaders.append(loader)
        
        # 构建 GNN 模型
        model = GNN(input_dim=loaders[0].dataset.features_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=loaders[0].dataset.n_classes,
                    n_layers=args.n_layers,
                    batchnorm_dim=args.batchnorm_dim, 
                    dropout_1=args.dropout_1, 
                    dropout_2=args.dropout_2).to(device)
    
        print('\nInitialize model')
        print(model)
        
        c = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('N trainable parameters:', c)
        #print("Model device:", next(model.parameters()).device)
        
        #user_input = input("请输入数字3以继续运行: ")   
        aug1 = GraphAugmentor(drop_prob=0.9, insert_ratio=0.1, edge_prob=0.9)
        aug2 = GraphAugmentor(drop_prob=0.9, insert_ratio=0.1, edge_prob=0.9)

        encoder_model = Encoder(encoder=model, augmentor=(aug1, aug2)).to(device)
        # 查找最新的encoder检查点
        latest_encoder_path, start_epoch = find_latest_encoder_checkpoint(fold_id)
        
        # 如果找到检查点，加载encoder模型状态
        if latest_encoder_path:
            print(f'找到最新encoder检查点 {latest_encoder_path}，从 epoch {start_epoch+1} 继续训练')
            encoder_model.load_state_dict(torch.load(latest_encoder_path))
        else:
            print('未找到encoder检查点，从头开始训练')
            start_epoch = 0
        #print("Encoder model device:", next(encoder_model.parameters()).device)
        optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, encoder_model.parameters()),
                    lr=args.lr,
                    weight_decay=args.wdecay,
                    betas=(0.5, 0.999))
        
        # 如果有检查点，尝试加载优化器状态
        optimizer_path = latest_encoder_path.replace('.pth', '_optimizer.pth') if latest_encoder_path else None
        if optimizer_path and os.path.exists(optimizer_path):
            print(f'加载优化器状态: {optimizer_path}')
            optimizer.load_state_dict(torch.load(optimizer_path))
        #scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.5)
        # 可以根据你的训练轮数调整学习率下降的时机
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[25, 35],  # 在1/3和2/3处降低学习率
            gamma=0.5
        )
        for _ in range(start_epoch):
            scheduler.step()
        logger = setup_logger()
        # 定义训练函数
        def train(train_loader, epoch):
             # 创建各种损失追踪器
            losses = AverageMeter('Loss', ':.4e')
            instance_losses = AverageMeter('Instance Loss', ':.4e')
            cluster_losses = AverageMeter('Cluster Loss', ':.4e')
            consistency_losses = AverageMeter('Consistency Loss', ':.4e')
            entropy_losses = AverageMeter('Entropy Loss', ':.4e')
            repulsion_losses = AverageMeter('Repulsion Loss', ':.4e')  # 新增
            # 创建进度显示器
            progress = ProgressMeter(len(train_loader),
                       [losses, instance_losses, cluster_losses, consistency_losses, entropy_losses, repulsion_losses], 
                       prefix="Epoch: [{}]".format(epoch),
                       output_file="progress_log.txt")
            
            encoder_model.train()
            
            #model.train()
            start = time.time()
            train_loss, n_samples = 0, 0
            for batch_idx, data in enumerate(train_loader):
    
                data = [d.to(device) for d in data]
                optimizer.zero_grad()
                
                g, gs1, gs2 = encoder_model(data)
                output = torch.cat((gs1, gs2), dim=0)
                
                gi1 = encoder_model.encoder.instance_projector(gs1)
                gi2 = encoder_model.encoder.instance_projector(gs2)
                
                gc= encoder_model.encoder.cluster_projector(g)
                
                total_loss, consistency_loss, entropy_loss,repulsion_loss = [], [], [], []
                # 找到正负邻居(从三个图集中选择)
                pos_neighbors, neg_neighbors = find_improved_neighbors(g, gs1, gs2, data[4])
                gc_pos = encoder_model.encoder.cluster_projector(pos_neighbors)
                gc_neg = encoder_model.encoder.cluster_projector(neg_neighbors)
                
                total_loss, consistency_loss, entropy_loss, repulsion_loss = improved_cluster_criterion(gc, gc_pos, gc_neg)
            
    
                g1 = gi1.unsqueeze(1)
                g2 = gi2.unsqueeze(1)
                gg = torch.cat([g1, g2], dim=1)
                
                in_loss=criterion(gg, data[4])
                
                #total_loss, consistency_loss, entropy_loss = [], [], []
                # user_input = input("请输入数字1以继续运行: ")
                
                #total_loss_, consistency_loss_, entropy_loss_ = criterion_cluster(gc1, gc2)
                #c_loss = criterion_cluster(gc1, gc2)
                
                
                cluster_loss = total_loss
                #cluster_loss = torch.sum(torch.stack(total_loss, dim=0))
                if args.used_loss == "instanceonly":
                    loss = in_loss
                elif args.used_loss == "clusteronly":
                    loss = cluster_loss
                else:
                    #loss = 3*in_loss + cluster_loss
                     loss = in_loss + args.cluster_loss_weight * total_loss
                #loss = loss_fn(output, data[4])
                #loss = F.cross_entropy(output, data[4])
                loss.backward()
                optimizer.step()
                
                losses.update(loss.item())
                instance_losses.update(in_loss.item())
                cluster_losses.update(cluster_loss.item())
                consistency_losses.update(consistency_loss.item())  # 修改：直接使用.item()
                entropy_losses.update(entropy_loss.item())  # 修改：直接使用.item()
                repulsion_losses.update(repulsion_loss.item())
                
                
                time_iter = time.time() - start
                train_loss += loss.item() * len(output)
                n_samples += len(output)
                scheduler.step()
                if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                    time_iter = time.time() - start
                    # 修改: 更新日志消息，增加排斥损失信息
                    log_message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tInstance Loss: {:.6f}\tCluster Loss: {:.6f}\tConsistency Loss: {:.6f}\tEntropy Loss: {:.6f}\tRepulsion Loss: {:.6f}\tsec/iter: {:.4f}'.format(
                        epoch, (batch_idx + 1) * len(output), len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader),
                        losses.avg, instance_losses.avg, cluster_losses.avg, 
                        consistency_losses.avg, entropy_losses.avg, repulsion_losses.avg,  # 新增: 排斥损失
                        time_iter / (batch_idx + 1)
                    )
                    logger.info(log_message)
                    progress.display(batch_idx)
                  # 保存训练后的模型
            # 保存encoder模型，包含fold信息
            encoder_save_path = f'encoder_model_fold_{fold_id}_epoch_{epoch}.pth'
            torch.save(encoder_model.state_dict(), encoder_save_path)
            
            # 也保存优化器状态，以便完整恢复训练
            optimizer_save_path = f'encoder_model_fold_{fold_id}_epoch_{epoch}_optimizer.pth'
            torch.save(optimizer.state_dict(), optimizer_save_path)
        def test(test_loader, epoch, log_file="metrics.log"):
            encoder_model.eval()
            start_time = time.time()
            
            # 初始化存储容器
            test_loss, correct, n_samples = 0, 0, 0
            preds_list, true_labels_list = [], []
            embeds_instance, embeds_cluster = [], []  # 分别存储两种嵌入
            
            # 第一步：收集所有嵌入和标签
            all_embeddings = []
            all_true_labels = []
            all_cluster_embeddings = []
            
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    data = [d.to(device) for d in data]
                    
                    # 获取模型输出
                    g, _, _ = encoder_model(data)
                    
                    # 获取嵌入
                    gi = encoder_model.encoder.instance_projector(g)  # 实例级嵌入
                    gc = encoder_model.encoder.cluster_projector(g)  # 簇级嵌入
                    
                    # 计算分类损失
                    loss = F.cross_entropy(gc, data[4], reduction='sum')
                    
                    test_loss += loss.item()
                    n_samples += len(gc)
                    
                    # 收集嵌入和真实标签
                    all_embeddings.append(gi.cpu())
                    all_true_labels.extend(data[4].cpu().numpy())
                    all_cluster_embeddings.append(gc.cpu()) 
                    
                    # 保存用于后续聚类分析
                    embeds_instance.append(gi.cpu())
                    embeds_cluster.append(gc.cpu())
            
            # 将嵌入转换为NumPy数组
            all_embeddings = torch.cat(all_embeddings).numpy()
            all_true_labels = np.array(all_true_labels)
            all_cluster_embeddings_tensor = torch.cat(all_cluster_embeddings)
            
            # 新增: 监控簇间距离
            cluster_distances = monitor_cluster_distances(
                epoch, 
                all_cluster_embeddings_tensor, 
                torch.tensor(all_true_labels, device=all_cluster_embeddings_tensor.device)
            )
            # 第二步：计算每个类别的簇中心
            num_classes = 21  # 假设有21个类别
            cluster_centers = {}
            
            for class_idx in range(num_classes):
                # 获取当前类别的所有样本
                class_mask = (all_true_labels == class_idx)
                if np.sum(class_mask) > 0:  # 确保该类别有样本
                    class_embeddings = all_embeddings[class_mask]
                    # 计算该类别的簇中心（平均值）
                    cluster_centers[class_idx] = np.mean(class_embeddings, axis=0)
            
            # 第三步：为每个样本分配伪标签（基于距离最近的簇中心）
            pseudo_labels = []
            
            for i, embedding in enumerate(all_embeddings):
                distances = {}
                for class_idx, center in cluster_centers.items():
                    # 计算欧氏距离
                    dist = np.linalg.norm(embedding - center)
                    distances[class_idx] = dist
                
                # 选择距离最小的类别作为伪标签
                pseudo_label = min(distances, key=distances.get)
                pseudo_labels.append(pseudo_label)
            
            pseudo_labels = np.array(pseudo_labels)
            
            # 第四步：计算评估指标
            correct = np.sum(pseudo_labels == all_true_labels)
            acc = 100.0 * correct / len(all_true_labels)
            
            # 转换为张量以使用现有的评估函数
            true_labels_tensor = torch.tensor(all_true_labels)
            preds_tensor = torch.tensor(pseudo_labels)
            
            # 计算分类指标
            classnums = 21
            r = recall(preds_tensor, true_labels_tensor, classnums)
            p = precision(preds_tensor, true_labels_tensor, classnums)
            f1 = f1_score(preds_tensor, true_labels_tensor, classnums)
            
            # 绘制混淆矩阵并保存
            conf_matrix = get_confusion_matrix(true_labels_tensor, preds_tensor)
            plt.figure(figsize=(18, 16), dpi=80)
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - Cluster Distance Based (Epoch {epoch})')
            plt.colorbar()
            plt.savefig(f'confusion_matrix_cluster_based_epoch_{epoch}.png')
            plt.close()
            
            # ================ 聚类评估 ================ #
            def evaluate_embeddings(embeds, prefix=""):
                """通用嵌入评估函数"""
                embeds_np = torch.cat(embeds).numpy()
                true_labels_np = np.array(all_true_labels)
                
                # 检查嵌入维度
                assert embeds_np.shape[1] >= 2, f"{prefix}嵌入维度需≥2，当前为{embeds_np.shape[1]}"
                
                # 使用真实类别数进行聚类
                n_clusters = len(np.unique(true_labels_np))
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(embeds_np)
                cluster_assignments = kmeans.labels_
                
                # 计算聚类指标
                nmi = metrics.normalized_mutual_info_score(true_labels_np, cluster_assignments)
                ari = metrics.adjusted_rand_score(true_labels_np, cluster_assignments)
                f = metrics.fowlkes_mallows_score(true_labels_np, cluster_assignments)
                silhouette = metrics.silhouette_score(embeds_np, cluster_assignments)
                
                # 计算调整后准确率
                pred_adjusted = get_y_preds(true_labels_np, cluster_assignments, n_clusters)
                acc = metrics.accuracy_score(pred_adjusted, true_labels_np)
                
                # 可视化t-SNE并保存（不显示）
                plt.figure(figsize=(10, 8))
                tsne = TSNE(n_components=2, perplexity=30, random_state=0)
                embeds_2d = tsne.fit_transform(embeds_np[:1000])  # 限制样本数量
                plt.scatter(embeds_2d[:,0], embeds_2d[:,1], 
                            c=true_labels_np[:1000], 
                            cmap='tab20', alpha=0.6)
                plt.title(f'{prefix}Embedding Space (t-SNE)')
                plt.colorbar()
                plt.savefig(f'{prefix}tsne_epoch_{epoch}.png')
                plt.close()
                
                return {
                    'nmi': nmi,
                    'ari': ari,
                    'f_score': f,
                    'silhouette': silhouette,
                    'acc': acc,
                }

            # 评估实例级嵌入
            metrics_instance = evaluate_embeddings(embeds_instance, "Instance-")
            
            # 评估簇级嵌入
            metrics_cluster = evaluate_embeddings(embeds_cluster, "Cluster-")

            # 运行详细的聚类分析
            instance_analysis = analyze_clustering(embeds_instance, all_true_labels, 
                                                n_classes=21, prefix="Instance-", 
                                                save_dir="./clustering_analysis/", 
                                                epoch=epoch)
            
            cluster_analysis = analyze_clustering(embeds_cluster, all_true_labels, 
                                                n_classes=21, prefix="Cluster-", 
                                                save_dir="./clustering_analysis/", 
                                                epoch=epoch)
            
            # ================ 日志记录 ================ #
            with open(log_file, 'a') as f:
                # 添加时间戳和分隔线
                f.write(f"\n{'='*80}\n")
                f.write(f"实验日期时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"轮次 (Epoch): {epoch}\n")
                f.write(f"{'-'*80}\n\n")
                
                # 基于簇中心距离的分类评估结果
                f.write("【基于簇中心距离的分类评估结果】\n")
                f.write(f"损失 (Loss): {test_loss/n_samples:.4f}\n")
                f.write(f"准确率 (Accuracy): {acc:.2f}%\n")
                
                
                # 新增: 簇间距离信息
                f.write(f"\n【簇间距离监控】\n")
                f.write(f"平均簇间距离: {cluster_distances['avg_distance']:.4f}\n")
                f.write(f"最小簇间距离: {cluster_distances['min_distance']:.4f}\n")
                
                f.write("\n📊 类别指标汇总:\n")
                # 使用更宽的列和明确的边框
                header = f"┌{'─'*8}┬{'─'*8}┬{'─'*8}┬{'─'*8}┐\n"
                header += f"│{'类别ID':^8}│{'召回率':^8}│{'精确度':^8}│{'F1值':^8}│\n"
                header += f"├{'─'*8}┼{'─'*8}┼{'─'*8}┼{'─'*8}┤\n"
                f.write(header)

                for class_idx in range(len(r)):
                    f.write(f"│{class_idx:^8}│{r[class_idx].item():.4f}│{p[class_idx].item():.4f}│{f1[class_idx].item():.4f}│\n")

                f.write(f"└{'─'*8}┴{'─'*8}┴{'─'*8}┴{'─'*8}┘\n\n")
                
                # 聚类评估 - 实例级嵌入
                f.write(f"{'-'*80}\n")
                f.write("【聚类评估 - 实例级嵌入】\n")
                f.write(f"标准化互信息 (NMI): {metrics_instance['nmi']:.4f}\n")
                f.write(f"调整兰德指数 (ARI): {metrics_instance['ari']:.4f}\n")
                f.write(f"Fowlkes-Mallows分数: {metrics_instance['f_score']:.4f}\n")
                f.write(f"轮廓系数 (Silhouette): {metrics_instance['silhouette']:.4f}\n")
                f.write(f"调整后准确率 (Adjusted Acc): {metrics_instance['acc']:.4f}\n")
                
                # 聚类评估 - 簇级嵌入
                f.write(f"{'-'*80}\n")
                f.write("【聚类评估 - 簇级嵌入】\n")
                f.write(f"标准化互信息 (NMI): {metrics_cluster['nmi']:.4f}\n")
                f.write(f"调整兰德指数 (ARI): {metrics_cluster['ari']:.4f}\n")
                f.write(f"Fowlkes-Mallows分数: {metrics_cluster['f_score']:.4f}\n")
                f.write(f"轮廓系数 (Silhouette): {metrics_cluster['silhouette']:.4f}\n")
                f.write(f"调整后准确率 (Adjusted Acc): {metrics_cluster['acc']:.4f}\n")
                
                # 新增：添加详细聚类分析结果路径
                f.write(f"{'-'*80}\n")
                f.write("【详细聚类分析】\n")
                f.write(f"实例级嵌入聚类分析报告: {instance_analysis['report_path']}\n")
                f.write(f"簇级嵌入聚类分析报告: {cluster_analysis['report_path']}\n")
                
                # 总结信息
                f.write(f"{'-'*80}\n")
                f.write(f"总耗时: {time.time()-start_time:.2f}秒\n")
                f.write(f"{'='*80}\n")
            
            # 只在日志中添加简要进度提示
            print(f"Epoch {epoch}: 评估完成，结果已保存到 {log_file}")
            print(f"详细聚类分析已保存到 clustering_analysis/ 目录")
            
            return {
                'classification': {
                    'loss': test_loss/n_samples,
                    'acc': acc,
                    'recall': r.tolist(),
                    'precision': p.tolist(),
                    'f1': f1.tolist()
                },
                'instance_embedding': metrics_instance,
                'cluster_embedding': metrics_cluster,
                'instance_clustering_analysis': instance_analysis,
                'cluster_clustering_analysis': cluster_analysis,
                'cluster_centers': {k: v.tolist() for k, v in cluster_centers.items()},  # 保存簇中心
                'cluster_distances': cluster_distances,
                'time': time.time()-start_time
            }   
            
        
        def plot_confusion_matrix(conf_matrix, num_classes, epoch):
            plt.imshow(conf_matrix, cmap=plt.cm.Blues)
            indices = range(len(conf_matrix))
            if num_classes == 21:
                classes = list(range(21))
            elif num_classes == 18:
                classes = list(range(18))
            elif num_classes == 15:
                classes = list(range(15))
            elif num_classes == 27:
                classes = list(range(27))
            elif num_classes == 33:
                classes = list(range(33))
            plt.xticks(indices, classes)
            plt.yticks(indices, classes)
            plt.colorbar()
            plt.xlabel('y_pred')
            plt.ylabel('y_true')
            for first_index in range(len(conf_matrix)):
                for second_index in range(len(conf_matrix[first_index])):
                    plt.text(first_index, second_index, conf_matrix[second_index, first_index])
            plt.savefig('./fig{}.png'.format(epoch), format='png')
            plt.show()
    
        # 计算混淆矩阵
        def get_confusion_matrix(label, pred):
            return confusion_matrix(label, pred)
    
        #loss_fn = F.cross_entropy  # 定义损失函数
        criterion = SupConLoss(temperature=0.07)
        criterion_cluster = ClusterLoss(args.entropy_weight).to(device)
        improved_cluster_criterion = ImprovedClusterLoss(
            entropy_weight=args.entropy_weight, 
            repulsion_weight=args.repulsion_weight, 
            temperature=args.cluster_temperature
        ).to(device)
        max_acc = 0.0
        t_start = time.time()
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train(loaders[0], epoch)
            #scheduler.step()
            result = test(loaders[1], epoch,log_file="experiment1.log")
            #acc = test(loaders[1], epoch)
            acc=result['classification']['acc']
            accuracy_arr[fold_id][epoch] = acc
            max_acc = max(max_acc, acc)
        print("time: {:.4f}s".format(time.time() - t_start))
        
        acc_folds.append(max_acc)
    
    # ================ 新增: 新类检测系统演示 ================ #
    print("\n" + "="*60)
    print("演示 NoveltyDetectionSystem 新类检测功能")
    print("="*60)
    
    try:
        # 这里添加新类检测的演示代码
        # 注意: 在实际使用中，您需要有真实的嵌入数据和标签
        print("NoveltyDetectionSystem 已成功导入并可以使用")
        print("使用方法:")
        print("1. from novelty_detection_system import NoveltyDetectionSystem")
        print("2. nds = NoveltyDetectionSystem()")
        print("3. nds.fit(known_embeddings, known_labels)")
        print("4. is_novel, distances = nds.predict_novelty(test_embeddings)")
        print("5. clustering_results = nds.cluster_novel_samples(novel_embeddings)")
        print("\n详细使用示例请参考 test_novelty_detection.py")
        
    except Exception as e:
        print(f"NoveltyDetectionSystem 演示出错: {e}")
    
    print("="*60)
    # ================ 新类检测系统演示结束 ================ #

    print("Accuracies for each fold:", acc_folds)
    # 以下代码可用于计算交叉验证的平均准确率和标准差
    # mean_validation = accuracy_arr.mean(axis=0)
    # maximum_epoch = np.argmax(mean_validation)
    # average = np.mean(accuracy_arr[:, maximum_epoch])
    # standard_dev = np.std(accuracy_arr[:, maximum_epoch])
    # print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, average, standard_dev))

if __name__ == '__main__':
    main()