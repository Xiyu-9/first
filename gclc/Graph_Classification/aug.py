import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
class DataAugmentation:
    def __init__(self, data, random_seed=None):
        self.data = data
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def augment(self):
        
        print("Starting data augmentation...")
        # 计算每个类别的样本数量
        unique_classes, counts = np.unique(self.data['targets'], return_counts=True)

        # 找到样本数最多的类别及其样本数
        max_class = unique_classes[np.argmax(counts)]
        max_count = counts.max()

        # 设定 w 和 p
        w = max_count
        p = w // 2

        # 存储增强后的样本
        augmented_data = {
            'features': [],
            'adj_list': [],
            'targets': []
        }

        # 从样本数最多的类别样本中随机抽取 p 个样本作为该类别的新样本数
        max_class_indices = np.where(self.data['targets'] == max_class)[0]
        new_samples_max_class = random.sample(list(max_class_indices), p)
        
        # 增加到增强数据中
        for idx in new_samples_max_class:
            augmented_data['features'].append(self.data['features'][idx])
            augmented_data['adj_list'].append(self.data['adj_list'][idx])
            augmented_data['targets'].append(max_class)

        # 遍历剩下的类别
        for cls in unique_classes:
            if cls != max_class:
                class_indices = np.where(self.data['targets'] == cls)[0]
                if len(class_indices) > p:
                    new_samples_by_class = random.sample(list(class_indices), p)
                    # 增加到增强数据中
                    for idx in new_samples_max_class:
                        augmented_data['features'].append(self.data['features'][idx])
                        augmented_data['adj_list'].append(self.data['adj_list'][idx])
                        augmented_data['targets'].append(cls)
                else:
                    print(f"Class {cls}: augmentation beigin")
                    current_samples = [(self.data['features'][i], self.data['adj_list'][i]) for i in class_indices]
                    # 调用数据增强方法，传入特征和邻接矩阵
                    augmented_samples = self.data_augmentation(current_samples, p)
                    print(f"Class {cls}: augmentation completed")
                    new_samples_by_class = augmented_samples
                    for sample in tqdm(augmented_samples, total=len(augmented_samples)):
                        features, adj_matrix = sample
                        augmented_data['features'].append(features)
                        augmented_data['adj_list'].append(adj_matrix)
                        augmented_data['targets'].append(cls)
                    

        return augmented_data


    def data_augmentation(self, samples, p):
        """
        数据增强方法，根据输入样本和目标数量 p，通过数据增强生成新的样本集。

        参数：
        - samples: 原始样本集（列表或数组）
        - p: 目标样本数量

        返回：
        - 增强后的样本集，大小为 p
        """
        n = len(samples)  # 当前类别的样本数
        if n == 0:
            raise ValueError("输入样本集不能为空！")

        t = p // n  # 数据增强的倍数
        origin = samples.copy()  # 原始样本集的副本
        augmented_samples = origin.copy()  # 初始化增强后的样本集
        print("Class augmentation into data_augmentation")
        # 增强 t 次，每次生成 n 个新样本
        for _ in range(t-1):
            #print(f"for {_}: augmentation beigin")
            new_samples = self.apply_random_augmentation(origin)  # 调用数据增强技术
            #print("Class augmentation out of apply_random_augmentation")
            augmented_samples.extend(new_samples)

        # 当前新数据集的样本数量
        current_sample_count = len(augmented_samples)
        
        print("Class augmentation into 补充_augmentation")
        # 如果 m > p，从 m 个样本中随机抽取 p 个
        if current_sample_count > p:
            augmented_samples = random.sample(augmented_samples, p)
        
        # 如果 m < p，补充额外的样本
        elif current_sample_count < p:
            
            additional_samples_needed = p - current_sample_count  # 需要补充的样本数
            supplement_samples = []

            # 随机选择原始样本进行增强，直到补充到目标数量
            while len(supplement_samples) < additional_samples_needed:
                random_sample = random.choice(origin)  # 随机选择一个原始样本
                
                augmented_sample = self.apply_random_augmentation([random_sample])[0]
                
                supplement_samples.append(augmented_sample)

            augmented_samples.extend(supplement_samples)

        return augmented_samples



    def apply_random_augmentation(self, samples):
        data_aug = deepcopy(samples)
        
        #random.shuffle(data_aug)  # 随机打乱样本
        #print("Class augmentation into apply_random_augmentation")
        for idx in range(len(data_aug)):
            #print(f"sample {idx}: augmentation beigin")
            aug = np.random.randint(4)  # 生成 0 到 3 的随机数
            #aug= 1
            features, adj_matrix = data_aug[idx]  # 解包特征和邻接矩阵
            
            if aug == 0:
                # 随机丢弃节点方法
                retention_prob=0.9
                new_features, new_adj_matrix = self.drop_node(features, adj_matrix, retention_prob)  
                #print("New Features after dropping nodes:", new_features)
                #print("New Adjacency Matrix after dropping nodes:\n", new_adj_matrix)
                # 检查边数是否为偶数
            
            elif aug == 1:
                # 随机增加节点方法
                insertion_ratio=0.1
                adj_matrix, features = self.add_node(features, adj_matrix, insertion_ratio)  
                #print("New Features after adding nodes:", features)
                #print("New Adjacency Matrix after adding nodes:\n", adj_matrix)
                # 检查边数是否为偶数
                n = np.count_nonzero(adj_matrix)  # 计算非零元素的数量（即边数）
                if n % 2 != 0:
                    print(f"Warning: Graph {idx} has an odd number of edges after add_node: {n}")
                    print(f"adj_matrix的形状是：{adj_matrix.shape}")
                    user_input = input("边不是偶数: ") 

            elif aug == 2:
                # 随机边属性扰动方法
                retention_prob=0.9
                adj_matrix = self.edge_perturb(adj_matrix, retention_prob)  
                #print("New Adjacency Matrix after perturbing edges:\n", adj_matrix)
                # 检查边数是否为偶数

            # 更新增强后的样本
            data_aug[idx] = (features, adj_matrix)  
            
        return data_aug

    def drop_node(self, features, adj_matrix, retention_prob):
        #print("sample drop_node augmentation beigin")
        num_nodes = len(features)

        # 确保 features 和 adj_matrix 是 NumPy 数组
        features = np.array(features)
        adj_matrix = np.array(adj_matrix)

        # 为每个节点生成保留概率
        retain_probs = [retention_prob] * num_nodes
        
        # 从Bernoulli分布中采样，得到布尔值的张量subset
        subset = np.random.binomial(1, retain_probs)  # 1表示保留，0表示丢弃
        
        # 创建新的特征矩阵和邻接矩阵
        new_features = features[subset == 1]  # 保留的特征
        retained_indices = np.where(subset == 1)[0]  # 被保留的节点索引
        
        # 更新邻接矩阵，删除未被保留的节点及其相关边
        new_adj_matrix = adj_matrix[np.ix_(retained_indices, retained_indices)]
        
        return new_features, new_adj_matrix



    def add_node(self, features, adj_matrix, insertion_ratio):
        # 获取当前图的节点和边数量
        num_nodes = adj_matrix.shape[0]
        num_edges = np.sum(adj_matrix) // 2

        # 按插入比例计算需要插入的节点数量
        num_insert_nodes = int(num_nodes * insertion_ratio)

        # 初始化新的邻接矩阵和特征矩阵
        new_adj_matrix = adj_matrix.copy()
        new_feature_matrix = features.copy()

        # 检查是否所有特征均为正或均为负
        all_positive = all(f > 0 for f in new_feature_matrix)
        all_negative = all(f < 0 for f in new_feature_matrix)

        while num_insert_nodes > 0 and (all_positive or all_negative):
                # 在倒数第二个位置插入新节点
                new_node_index = new_adj_matrix.shape[0]
                new_feature = new_feature_matrix[-2]  # 使用倒数第二个节点的特征值
                
                # 更新特征矩阵
                new_feature_matrix = np.append(new_feature_matrix, [new_feature])

                # 更新邻接矩阵
                new_row = np.zeros((1, new_adj_matrix.shape[1]))
                new_col = np.zeros((new_adj_matrix.shape[0] + 1, 1))
                new_adj_matrix = np.vstack((new_adj_matrix, new_row))
                new_adj_matrix = np.hstack((new_adj_matrix, new_col))

                # 连接新节点与倒数第二个节点
                new_adj_matrix[-2, new_node_index] = 1
                new_adj_matrix[new_node_index, -2] = 1

                # 减少插入节点的数量
                num_insert_nodes -= 1

        i = 0  # 初始化索引

        while num_insert_nodes > 0 and i < num_nodes - 2:
                if np.sign(new_feature_matrix[i]) != np.sign(new_feature_matrix[i + 1]):
                    
                    new_node_index = new_adj_matrix.shape[0]
                    new_feature = new_feature_matrix[i]  # 默认新节点特征值与当前节点相同

                    # 更新特征矩阵
                    new_feature_matrix = np.append(new_feature_matrix, [new_feature])

                    # 扩展邻接矩阵
                    new_row = np.zeros((1, new_adj_matrix.shape[1]))
                    new_col = np.zeros((new_adj_matrix.shape[0] + 1, 1))
                    new_adj_matrix = np.vstack((new_adj_matrix, new_row))
                    new_adj_matrix = np.hstack((new_adj_matrix, new_col))

                    # 与当前节点连接（层内边）
                    new_adj_matrix[i, new_node_index] = 1
                    new_adj_matrix[new_node_index, i] = 1

                    # 向前查找第一个特征符号不同的节点并调整边关系
                    found_connection = False
                    
                    for j in range(i - 1, -1, -1):
                        if np.sign(new_feature_matrix[j]) != np.sign(new_feature_matrix[i]):
                            # 新插入节点与C相连，删除B与C的边
                            if not (new_adj_matrix[i, j] == 0):  # 确保B与C之间存在边
                                new_adj_matrix[new_node_index, j] = 1
                                new_adj_matrix[j, new_node_index] = 1
                                new_adj_matrix[i, j] = 0
                                new_adj_matrix[j, i] = 0
                            found_connection = True
                            break
                    
                    if not found_connection:
                        # 如果向前找不到，向后查找第一个特征符号不同的节点并调整边关系
                        for j in range(i + 1, num_nodes):
                            if j < num_nodes and np.sign(new_feature_matrix[j]) != np.sign(new_feature_matrix[i]):
                                # 新插入节点与D相连，删除B与D的边
                                if not (new_adj_matrix[i, j] == 0):  # 确保B与D之间存在边
                                    new_adj_matrix[new_node_index, j] = 1
                                    new_adj_matrix[j, new_node_index] = 1
                                    new_adj_matrix[i, j] = 0
                                    new_adj_matrix[j, i] = 0
                                break

                    # 更新插入节点计数和总节点数
                    num_insert_nodes -= 1
                    num_nodes += 1

                i += 1  # 移动到下一个索引

        return new_adj_matrix, new_feature_matrix

    def edge_perturb(self, adj_matrix, retention_prob):
        #print("sample edge_perturb augmentation beigin")
        # 计算图中边数量
        num_edges = np.sum(adj_matrix > 0) // 2  # 计算有效边数量
        
        if num_edges == 0:
            return adj_matrix  # 如果没有边，直接返回原邻接矩阵

        # 为每条边生成保留概率
        retain_probs = np.full(num_edges, retention_prob, dtype=float)  # 使用 NumPy 数组，并确保是浮点数
        
        # 从Bernoulli分布中采样，得到布尔值的张量subset
        subset = np.random.binomial(1, retain_probs)  # 1表示不扰动，0表示扰动
        
        # 更新邻接矩阵
        new_adj_matrix = adj_matrix.copy()
        
        edge_count = 0
        for i in range(len(adj_matrix)):
            for j in range(i + 1, len(adj_matrix)):
                if adj_matrix[i][j] != 0:
                    if edge_count < len(subset) and subset[edge_count] == 0:  # 修改该边属性
                        new_adj_matrix[i][j] = 1 if adj_matrix[i][j] == 0.01 else 0.01
                        new_adj_matrix[j][i] = new_adj_matrix[i][j]
                    edge_count += 1
        
        return new_adj_matrix




