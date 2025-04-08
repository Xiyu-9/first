import numpy as np
import os
import math
from os.path import join as pjoin
import torch
from sklearn.model_selection import StratifiedKFold
#from IPython.core.debugger import Tracer
from itertools import chain
import random


class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 data_dir,
                 fold_dir,
                 rnd_state=None,
                 use_cont_node_attr=False,
                 folds=10):

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist.")
        else:
            print(f"Directory {self.data_dir} exists. Files: {os.listdir(self.data_dir)}")
        self.fold_dir = fold_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        if fold_dir!=None:
            fold_files = os.listdir(self.fold_dir)
        # 字典data存储图节点关系
        data = {}

        # nodes包含节点到图id的映射，graphs包含每个图及其节点列表的字典
        print("Reading graph nodes and relations...")
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])

        print("Reading node features...")
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                 nodes, graphs, fn=lambda s: int(s.strip()))

        #data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)

        print("Reading adjacency list and edge features...")
        data['adj_list'] = self.read_edge_features_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0],
                                                       list(filter(lambda f: f.find('edge_attribute') >= 0, files))[0],
                                                       nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        ###################################################################################################################

        print("Reading targets...")
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
        # 打印 data['features'] 的长度
        #print("Length of data['features']:", len(data['features']))

        # 打印 data['adj_list'] 的长度
        #print("Length of data['adj_list']:", len(data['adj_list']))

        # 打印 data['targets'] 的长度
        #print("Length of data['targets']:", len(data['targets']))

        # 计算并打印每个类别的样本个数总和
        #unique_classes, counts = np.unique(data['targets'], return_counts=True)
        #for cls, count in zip(unique_classes, counts):
            #print(f"Class {cls}: {count} samples")

        print("Data loading complete.")
        from dataprocessor import DataProcessor

        # 初始化数据处理器并处理数据
        processor = DataProcessor(data)
        processor.process()

        # 从处理好的数据中每个类别随机抽取1000个样本作为新数据集
        augmented_data = self.sample_data(processor.data, sample_size=1000)

        #from aug import DataAugmentation

        #data_augmentation_instance = DataAugmentation(processor.data, random_seed=42)
        #augmented_data = data_augmentation_instance.augment()

                # 打印结果以验证增强效果
        #print("Augmented Features:", len(augmented_data['features']))
        #print("Augmented Targets:", len(augmented_data['targets']))

        #计算并打印每个类别的样本个数总和
        unique_classes, counts = np.unique(augmented_data['targets'], return_counts=True) 
        
        for cls, count in zip(unique_classes, counts):
            print(f"Class {cls}: {count} samples")

        # 将每个特征值的负号去掉
        augmented_data['features'] = [[abs(feature) for feature in sample] for sample in augmented_data['features']]

        print("Data samped complete.")
        #user_input = input("请输入数字4以继续运行: ")
        # 等待用户输入数字1再继续运行
        #user_input = input("请输入数字1以继续运行: ")

        #while user_input != '1':
            #print("输入无效，请输入数字1以继续运行.")
            #user_input = input("请输入数字1以继续运行: ")

        #print("用户输入了数字1，继续运行程序...")

        if self.use_cont_node_attr:
            augmented_data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
                                                   nodes, graphs, 
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
            
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(augmented_data['adj_list']):
            N = len(adj)  # number of nodes
            if augmented_data['features'] is not None:
                assert N == len(augmented_data['features'][sample_id]), (N, len(augmented_data['features'][sample_id]))
            #n = np.sum(adj)  # total sum of edges
            ################################################
            n = np.count_nonzero(adj)  # total sum of edges

            if n % 2 != 0:
                print(f"Warning: Graph {sample_id} has an odd number of edges: {n}. Adjusting to nearest even number.")
                n = math.ceil(n / 2) * 2  

            assert n % 2 == 0, n
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(augmented_data['features'][sample_id]))

        #user_input = input("请输入数字2以继续运行: ")                
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1) # number of possible values
        features_onehot = []
        for i, x in enumerate(features):
            #feature_onehot = np.zeros(((len(x), features_dim)), dtype=np.int)
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(augmented_data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)
        print("feature_onehot:", len(feature_onehot[0]))

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]
            
        shapes = [len(adj) for adj in augmented_data['adj_list']]
        labels = augmented_data['targets']        # graph class labels
        labels -= np.min(labels)        # to start from 0
        N_nodes_max = np.max(shapes)    

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), 
                                                              np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), 
                                                              np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), 
                                                                  np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        #user_input = input("请输入数字3以继续运行: ") 

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(augmented_data['adj_list']) == len(features_onehot), 'invalid data'

        
        #stratified splits 
        train_ids, test_ids = self.stratified_split_data(labels, self.rnd_state, folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        #Tracer()()

        augmented_data['features_onehot'] = features_onehot
        augmented_data['targets'] = labels
        augmented_data['splits'] = splits

        augmented_data['N_nodes_max'] = np.max(shapes)  
        augmented_data['features_dim'] = features_dim
        augmented_data['n_classes'] = n_classes
        
        self.data = augmented_data

    def sample_data(self, data, sample_size=1000):
            """
            从每个类别中随机抽取指定数量的样本
            """
            unique_classes, counts = np.unique(data['targets'], return_counts=True)
            sampled_data = {'features': [], 'adj_list': [], 'targets': []}
            
            for cls in unique_classes:
                class_indices = np.where(data['targets'] == cls)[0]
                if len(class_indices) > sample_size:
                    new_samples_by_class = random.sample(list(class_indices), sample_size)
                    # 增加到增强数据中
                    for idx in new_samples_by_class:
                        sampled_data['features'].append(data['features'][idx])
                        sampled_data['adj_list'].append(data['adj_list'][idx])
                        sampled_data['targets'].append(cls)
                else:
                    sampled_data['features'].extend([data['features'][i] for i in class_indices])
                    sampled_data['adj_list'].extend([data['adj_list'][i] for i in class_indices])
                    sampled_data['targets'].extend([data['targets'][i] for i in class_indices])
            
            sampled_data['features'] = list(sampled_data['features'])
            sampled_data['adj_list'] = list(sampled_data['adj_list'])
            sampled_data['targets'] = np.array(sampled_data['targets'])
            
            return sampled_data
    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids
    
    def split_ids_from_text(self, files, rnd_state=None, folds=10):
        
        train_ids = []
        test_ids = []
        
        test_file_list = sorted([s for s in files if "test" in s])
        train_file_list = sorted([s for s in files if "train" in s])

        for fold in range(folds):
            with open(pjoin(self.fold_dir, train_file_list[fold]), 'r') as f:
                train_samples = [int(line.strip()) for line in f]

            train_ids.append(np.array(train_samples))
            
            with open(pjoin(self.fold_dir, test_file_list[fold]), 'r') as f:
                test_samples = [int(line.strip()) for line in f]

            test_ids.append(np.array(test_samples))

        return train_ids, test_ids
    
    def stratified_split_data(self, labels, seed, folds):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        
        train_ids = []
        test_ids = []
        for fold_idx in range(folds):
            train_idx, test_idx = idx_list[fold_idx]
            train_ids.append(np.array(train_idx))
            test_ids.append(np.array(test_idx))

        return train_ids, test_ids
    
    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1

            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    ##############################################################################################
    def read_edge_features_adj(self, fpath, fpath1, nodes, graphs, fn):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        edge_features = self.parse_txt_file(fpath1, line_parse_fn=fn)
        for num in range(len(edge_features)):
            if edge_features[num] >= 0 and edge_features[num] <= 1:
                edge_features[num] = 1
            else:
                edge_features[num] = 0.01

        adj_dict = {}
        i = 0
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = edge_features[i]
            i += 1

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list

    def random_select_examplers(self, train_ids, num_classes_begin, num_classes_end, label, count=0, method='all'):
        graphs_part = DynamicList()
        graphs_all = []
        for labels in range(num_classes_begin, num_classes_end):
            for idx in range(0, len(train_ids)):
                if (label[train_ids[idx]] == labels):
                    graphs_part[labels].append(train_ids[idx])
        #print(len(graphs_part))

        for labels in range(num_classes_begin, num_classes_end):
            if len(graphs_part[labels]) > count:
                if method == "all":
                    graphs_part[labels] = graphs_part[labels]
                if method == 'before_numbers':
                    graphs_part[labels] = graphs_part[labels][0:count]
                if method == 'random':
                    graphs_part[labels] = random.sample(graphs_part[labels], count)
                graphs_all.append(graphs_part[labels])
            else:
                graphs_all.append(graphs_part[labels])

        graphs_all = list(chain(*graphs_all))
        #graphs_all_ = []
        #graphs_all_.append(np.array(graphs_all))

        return graphs_all
    ##############################################################################################
class DynamicList(list):

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))
    def __setslice__(self, i, j, seq):
        return self.__setitem__(slice(i, j), seq)
    def __delslice__(self, i, j):
        return self.__delitem__(slice(i, j))

    def _resize(self, index):
        n = len(self)
        if isinstance(index, slice):
            m = max(abs(index.start), abs(index.stop))
        else:
            m = index + 1
        if m > n:
            self.extend([self.__class__() for i in range(m - n)])

    def __getitem__(self, index):
        self._resize(index)
        return list.__getitem__(self, index)

    def __setitem__(self, index, item):
        self._resize(index)
        if isinstance(item, list):
            item = self.__class__(item)
        list.__setitem__(self, index, item)

def shared_params(model, begin_layers=0, end_layers=0, method='None'):
    if method == 'mlps_begin_end':
        for i in range(begin_layers, end_layers):
            for param in model.convs[i].parameters():
                param.requires_grad = False
    if method == 'mlps':
        for param in model.convs.parameters():
            param.requires_grad = False