import numpy as np
from sklearn import metrics
from munkres import Munkres  # Munkres算法用于解决最优匹配问题

def evaluate(label, pred):
    print(f"真实类别数量: {len(set(label))}, 预测类别数量: {len(set(pred))}")
    print("真实标签 label:", np.unique(label))
    print("聚类结果 pred:", np.unique(pred))

    # 计算标准化互信息 (NMI)，衡量聚类结果与真实标签的一致性
    nmi = metrics.normalized_mutual_info_score(label, pred)
    # 计算调整兰德指数 (ARI)，衡量聚类结果与真实标签的相似度
    ari = metrics.adjusted_rand_score(label, pred)
    # 计算Fowlkes-Mallows指数 (F)，衡量聚类结果的精确性和召回率
    f = metrics.fowlkes_mallows_score(label, pred)
    # 调整预测标签，使其与真实标签尽可能匹配
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    # 计算调整后的准确率
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc

def calculate_cost_matrix(C, n_clusters):
    """
    计算代价矩阵，用于解决最优匹配问题。
    C 是混淆矩阵 (confusion matrix)，表示聚类结果与真实标签的对应关系。
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))  # 初始化代价矩阵
    
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # 计算真实类别 j 的所有数据样本数量
        for i in range(n_clusters):
            t = C[i, j]  # 计算聚类簇 i 中被分类为 j 的样本数
            cost_matrix[j, i] = s - t  # 代价计算：希望最大化匹配，所以用 s - t
    print("混淆矩阵:\n", C)
    print("代价矩阵:\n", cost_matrix)

    
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    """
    解析Munkres算法计算得到的最优匹配索引，返回最佳的簇标签对应关系。
    """
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]  # 取出匹配的标签
    
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    计算最佳预测标签，使其与真实标签最优匹配。
    
    cluster_assignments:    kmeans 聚类输出的标签
    y_true:                 真实标签
    n_clusters:             数据集中类别的数量
    
    返回值:
        y_pred: 调整后的预测标签，使其尽可能匹配真实标签
    """
    # 计算混淆矩阵（confusion matrix），用于衡量聚类结果与真实标签的对应关系
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    
    # 计算代价矩阵，用于最优匹配
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    
    # 使用 Munkres 算法（匈牙利算法）计算最佳标签映射关系
    indices = Munkres().compute(cost_matrix)
    
    # 获取最佳的簇标签映射
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    
    # 确保聚类标签从 0 开始编号（KMeans 可能会从非零编号开始）
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    
    # 依据最佳匹配关系调整预测标签
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    print("最佳映射关系:", kmeans_to_true_cluster_labels)
    print("聚类分配的最大索引:", np.max(cluster_assignments)) 
    return y_pred
#它主要用于 无监督学习（比如 KMeans 聚类），
# 因为无监督学习的聚类结果的标签是 无序的，而真实标签是有意义的类别。
# 这段代码的核心思想是用 Munkres（匈牙利）算法 找出聚类标签与真实类别的 最佳一一对应关系，从而 
'''这段代码主要是 用于评估聚类算法的表现，并 调整聚类标签，使其尽量匹配真实标签。
通过 混淆矩阵 + Munkres（匈牙利）算法，找到 最佳的聚类标签映射，从而提高 准确率。
这对于 无监督学习的聚类任务 非常重要，因为 KMeans 聚类的标签是无序的，必须重新匹配才能和真实标签对比。'''