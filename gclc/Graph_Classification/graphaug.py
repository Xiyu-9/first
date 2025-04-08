import torch
import random
import numpy as np

class GraphAugmentor:
    def __init__(self, drop_prob=0.9, insert_ratio=0.1, edge_prob=0.9):
        self.drop_prob = drop_prob      # 节点保留概率
        self.insert_ratio = insert_ratio  # 节点插入比例
        self.edge_prob = edge_prob      # 边保留概率

    def __call__(self, features: torch.Tensor, adj_matrix: torch.Tensor):
        """
        对 batch 内的每张图分别随机选择一种增强方法，
        增强后保持 features 形状 (64, 30, 1461) 和 adj_matrix 形状 (64, 30, 30)
        """
        augmented_features_list = []
        augmented_adj_list = []
        batch_size = features.shape[0]

        # 对 batch 中每一张图单独处理
        for i in range(batch_size):
            # 单张图特征: (30, 1461)，邻接矩阵: (30, 30)
            f = features[i]
            a = adj_matrix[i]
            method = random.choice(['drop_node', 'add_node', 'edge_perturb'])

            if method == 'drop_node':
                f_aug, a_aug = self.drop_node(f, a, self.drop_prob)
            elif method == 'add_node':
                f_aug, a_aug = self.add_node(f, a, self.insert_ratio)
            elif method == 'edge_perturb':
                # 对于 edge_perturb，仅增强邻接矩阵，特征保持不变
                a_aug = self.edge_perturb(a, self.edge_prob)
                f_aug = f
            else:
                f_aug, a_aug = f, a

            # 保证增强后的 f_aug 和 a_aug 形状仍然与原始一致（30, 1461) 和 (30, 30)
            augmented_features_list.append(f_aug)
            augmented_adj_list.append(a_aug)

        # 合并回 batch 维度
        augmented_features = torch.stack(augmented_features_list, dim=0)
        augmented_adj = torch.stack(augmented_adj_list, dim=0)
        return augmented_features, augmented_adj

    def drop_node(self, features: torch.Tensor, adj_matrix: torch.Tensor, retention_prob: float):
        """
        对单张图进行 drop_node 增强：
          - features: (N, F)
          - adj_matrix: (N, N)
        通过对部分节点“丢弃”（将对应行/列置零）来实现增强，
        并保证输出形状与输入一致。
        """
        N, F = features.shape
        # 生成每个节点的保留 mask（保证至少保留一个节点）
        retain_mask = torch.bernoulli(torch.full((N,), retention_prob)).bool()
        if not retain_mask.any():
            retain_mask[0] = True

        # 对于 features：丢弃的节点置零（保持 shape 不变）
        new_features = features.clone()
        new_features[~retain_mask] = 0.0

        # 对于邻接矩阵：丢弃对应节点的所有边
        new_adj_matrix = adj_matrix.clone()
        new_adj_matrix[~retain_mask, :] = 0.0
        new_adj_matrix[:, ~retain_mask] = 0.0

        return new_features, new_adj_matrix

    def add_node(self, features: torch.Tensor, adj_matrix: torch.Tensor, insertion_ratio: float):
        """
        对单张图进行 add_node 增强：
          - features: (N, F)
          - adj_matrix: (N, N)
        原始代码基于 NumPy 实现插入新节点，但考虑到最终需要保持图大小不变，
        此处仅模拟“插入节点”的效果，通过调整边关系来实现增强。
        """
        N, F = features.shape
        num_insert_nodes = int(N * insertion_ratio)
        if num_insert_nodes == 0:
            return features, adj_matrix

        # 这里采用 NumPy 方式进行部分操作（尽量不改变原有逻辑），但最终恢复为原始 shape
        new_features = features.clone()
        new_adj_matrix = adj_matrix.clone()
        new_features_np = new_features.cpu().numpy()
        new_adj_matrix_np = new_adj_matrix.cpu().numpy()

        # 遍历节点对，寻找特征符号不一致的位置，并调整边关系
        # 注意：这里将每个节点的特征先按所有维度求和后取 sign 判断
        i = 0
        while num_insert_nodes > 0 and i < N - 1:
            if np.sign(new_features_np[i].sum()) != np.sign(new_features_np[i+1].sum()):
                # 模拟插入：调整 i 与 i+1 之间的边
                new_adj_matrix_np[i, i+1] = 1
                new_adj_matrix_np[i+1, i] = 1
                num_insert_nodes -= 1
            i += 1

        new_features = torch.tensor(new_features_np, dtype=features.dtype, device=features.device)
        new_adj_matrix = torch.tensor(new_adj_matrix_np, dtype=adj_matrix.dtype, device=adj_matrix.device)
        return new_features, new_adj_matrix

    def edge_perturb(self, adj_matrix: torch.Tensor, retention_prob: float):
        """
        对单张图进行边扰动：
          - adj_matrix: (N, N)
        遍历上三角部分，对每条边根据 retention_prob 随机保留或进行扰动，
        扰动方式为在 1 和 0.01 之间切换，形状保持不变。
        """
        N = adj_matrix.shape[0]
        new_adj_matrix = adj_matrix.clone()

        # 遍历上三角区域
        for i in range(N):
            for j in range(i+1, N):
                if new_adj_matrix[i, j] != 0:
                    if torch.bernoulli(torch.tensor(retention_prob)) == 0:
                        new_adj_matrix[i, j] = 1.0 if new_adj_matrix[i, j] == 0.01 else 0.01
                        new_adj_matrix[j, i] = new_adj_matrix[i, j]
        return new_adj_matrix
