"""
Cluster analysis utilities for graph classification
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import os


def analyze_clustering(embeddings, true_labels, n_classes, prefix="", save_dir="./", epoch=0):
    """
    Perform comprehensive clustering analysis
    
    Args:
        embeddings: numpy array of embeddings
        true_labels: numpy array of true labels
        n_classes: number of classes for clustering
        prefix: prefix for output files
        save_dir: directory to save results
        epoch: current epoch number
    
    Returns:
        dict: analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate metrics
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    silhouette = silhouette_score(embeddings, cluster_labels)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: True labels
    plt.subplot(1, 3, 1)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings[:1000])  # Limit for visualization
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                c=true_labels[:1000], cmap='tab20', alpha=0.6)
    plt.title(f'{prefix}True Labels')
    plt.colorbar()
    
    # Plot 2: Clustering results
    plt.subplot(1, 3, 2)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                c=cluster_labels[:1000], cmap='tab20', alpha=0.6)
    plt.title(f'{prefix}Clustering Results')
    plt.colorbar()
    
    # Plot 3: Cluster centers
    plt.subplot(1, 3, 3)
    centers_2d = tsne.fit_transform(kmeans.cluster_centers_)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.title(f'{prefix}Cluster Centers')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}clustering_analysis_epoch_{epoch}.png'))
    plt.close()
    
    # Save detailed analysis
    analysis_file = os.path.join(save_dir, f'{prefix}analysis_epoch_{epoch}.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"Clustering Analysis - Epoch {epoch}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Normalized Mutual Information: {nmi:.4f}\n")
        f.write(f"Adjusted Rand Index: {ari:.4f}\n")
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
        f.write(f"Number of clusters: {n_classes}\n")
        f.write(f"Number of samples: {len(embeddings)}\n")
        
        # Cluster distribution
        f.write(f"\nCluster Distribution:\n")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            f.write(f"Cluster {cluster_id}: {count} samples\n")
    
    return {
        'nmi': nmi,
        'ari': ari,
        'silhouette': silhouette,
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'analysis_file': analysis_file
    }


def monitor_cluster_distances(epoch, embeddings, labels, save_dir="./cluster_distances/"):
    """
    Monitor and record average distances between different class clusters
    
    Args:
        epoch: current epoch
        embeddings: tensor of embeddings
        labels: tensor of labels
        save_dir: directory to save distance matrices
    
    Returns:
        dict: cluster distance information
    """
    import torch
    import torch.nn.functional as F
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Normalize embeddings
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
    
    num_classes = labels.max().item() + 1
    class_distances = torch.zeros((num_classes, num_classes), device=embeddings.device)
    class_counts = torch.zeros((num_classes, num_classes), device=embeddings.device)
    
    # Aggregate similarities by class
    for i in range(len(labels)):
        for j in range(len(labels)):
            class_i = labels[i].item()
            class_j = labels[j].item()
            class_distances[class_i, class_j] += similarity_matrix[i, j]
            class_counts[class_i, class_j] += 1
    
    # Average distances
    class_counts[class_counts == 0] = 1  # Avoid division by zero
    avg_distances = class_distances / class_counts
    
    # Convert to distance (1 - similarity for cosine similarity)
    avg_distances = 1 - avg_distances
    
    # Save distance matrix
    distance_file = os.path.join(save_dir, f'distances_epoch_{epoch}.txt')
    np.savetxt(distance_file, avg_distances.cpu().numpy(), fmt='%.4f')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_distances.cpu().numpy(), annot=True, cmap='viridis', 
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.title(f'Inter-class Distances (Epoch {epoch})')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.savefig(os.path.join(save_dir, f'distance_heatmap_epoch_{epoch}.png'))
    plt.close()
    
    return {
        'distance_matrix': avg_distances,
        'min_inter_class_distance': avg_distances[avg_distances > 0].min().item(),
        'max_intra_class_distance': torch.diag(avg_distances).max().item(),
        'distance_file': distance_file
    }