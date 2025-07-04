"""
NoveltyDetectionSystem for Graph Classification

This module implements a novelty detection system that can identify and cluster
novel classes that were not seen during training.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Tuple, Optional, Union


class NoveltyDetectionSystem:
    """
    A comprehensive novelty detection system for graph classification.
    
    This system can:
    1. Calculate known class centers from training data
    2. Determine distance thresholds for novelty detection
    3. Detect novel samples based on distance thresholds
    4. Cluster novel samples to discover new classes
    5. Provide evaluation and visualization capabilities
    """
    
    def __init__(self, 
                 distance_metric: str = 'cosine',
                 clustering_method: str = 'kmeans',
                 threshold_method: str = 'percentile',
                 threshold_percentile: float = 95.0,
                 min_cluster_size: int = 5,
                 max_clusters: int = 10,
                 random_state: int = 42):
        """
        Initialize the NoveltyDetectionSystem.
        
        Args:
            distance_metric: Distance metric to use ('cosine', 'euclidean')
            clustering_method: Clustering method ('kmeans', 'dbscan')
            threshold_method: Method to determine threshold ('percentile', 'std')
            threshold_percentile: Percentile for threshold calculation
            min_cluster_size: Minimum cluster size for DBSCAN
            max_clusters: Maximum number of clusters for K-means
            random_state: Random state for reproducibility
        """
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method
        self.threshold_method = threshold_method
        self.threshold_percentile = threshold_percentile
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.random_state = random_state
        
        # Internal state
        self.known_class_centers = None
        self.known_class_labels = None
        self.distance_threshold = None
        self.is_fitted = False
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> 'NoveltyDetectionSystem':
        """
        Fit the novelty detection system on known training data.
        
        Args:
            embeddings: Training embeddings of shape (n_samples, n_features)
            labels: Training labels of shape (n_samples,)
            
        Returns:
            self: Returns the fitted instance
        """
        self.logger.info("Fitting NoveltyDetectionSystem...")
        
        # Calculate class centers
        self.known_class_centers, self.known_class_labels = self._calculate_class_centers(
            embeddings, labels
        )
        
        # Determine distance threshold
        self.distance_threshold = self._calculate_distance_threshold(embeddings, labels)
        
        self.is_fitted = True
        self.logger.info(f"Fitted with {len(self.known_class_centers)} known classes")
        self.logger.info(f"Distance threshold set to: {self.distance_threshold:.4f}")
        
        return self
    
    def predict_novelty(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict which samples are novel (not belonging to known classes).
        
        Args:
            embeddings: Test embeddings of shape (n_samples, n_features)
            
        Returns:
            is_novel: Boolean array indicating novel samples
            distances: Distances to nearest known class center
        """
        if not self.is_fitted:
            raise ValueError("NoveltyDetectionSystem must be fitted before prediction")
        
        # Calculate distances to known class centers
        distances = self._calculate_distances_to_centers(embeddings)
        
        # Determine novelty based on threshold
        is_novel = distances > self.distance_threshold
        
        return is_novel, distances
    
    def cluster_novel_samples(self, 
                            novel_embeddings: np.ndarray, 
                            n_clusters: Optional[int] = None) -> Dict:
        """
        Cluster novel samples to discover new classes.
        
        Args:
            novel_embeddings: Embeddings of novel samples
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            dict: Clustering results including labels, centers, and metrics
        """
        if len(novel_embeddings) == 0:
            return {
                'cluster_labels': np.array([]),
                'cluster_centers': np.array([]),
                'n_clusters': 0,
                'silhouette_score': 0.0
            }
        
        if self.clustering_method == 'kmeans':
            return self._cluster_with_kmeans(novel_embeddings, n_clusters)
        elif self.clustering_method == 'dbscan':
            return self._cluster_with_dbscan(novel_embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def evaluate_novelty_detection(self, 
                                 embeddings: np.ndarray, 
                                 true_labels: np.ndarray,
                                 known_class_labels: List[int]) -> Dict:
        """
        Evaluate novelty detection performance.
        
        Args:
            embeddings: Test embeddings
            true_labels: True labels for test samples
            known_class_labels: List of known class labels
            
        Returns:
            dict: Evaluation metrics
        """
        # True novelty labels
        true_novel = ~np.isin(true_labels, known_class_labels)
        
        # Predicted novelty
        pred_novel, distances = self.predict_novelty(embeddings)
        
        # Calculate metrics
        tp = np.sum(true_novel & pred_novel)
        fp = np.sum(~true_novel & pred_novel)
        tn = np.sum(~true_novel & ~pred_novel)
        fn = np.sum(true_novel & ~pred_novel)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'threshold': self.distance_threshold
        }
    
    def visualize_novelty_detection(self, 
                                  embeddings: np.ndarray,
                                  labels: np.ndarray,
                                  is_novel: np.ndarray,
                                  save_path: str = None) -> None:
        """
        Visualize novelty detection results using t-SNE.
        
        Args:
            embeddings: All embeddings
            labels: True labels
            is_novel: Novelty predictions
            save_path: Path to save the visualization
        """
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, perplexity=30, random_state=self.random_state)
        embeddings_2d = tsne.fit_transform(embeddings[:1000])  # Limit for performance
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Original labels
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels[:1000], cmap='tab20', alpha=0.6)
        plt.title('True Labels')
        plt.colorbar(scatter)
        
        # Plot 2: Novelty detection
        plt.subplot(1, 3, 2)
        colors = ['blue' if not novel else 'red' for novel in is_novel[:1000]]
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=colors, alpha=0.6)
        plt.title('Novelty Detection (Blue: Known, Red: Novel)')
        
        # Plot 3: Known class centers
        if self.known_class_centers is not None:
            plt.subplot(1, 3, 3)
            # Transform centers to 2D space (approximate)
            if len(self.known_class_centers) > 1:
                perplexity = min(30, len(self.known_class_centers) - 1)
                tsne_centers = TSNE(n_components=2, perplexity=perplexity, random_state=self.random_state)
                centers_2d = tsne_centers.fit_transform(self.known_class_centers)
                plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                           c='red', marker='x', s=200, linewidths=3)
            else:
                # Single center case - just plot a marker
                plt.scatter([0], [0], c='red', marker='x', s=200, linewidths=3)
            plt.title('Known Class Centers')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _calculate_class_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the center (mean) for each known class."""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            class_mask = labels == label
            class_embeddings = embeddings[class_mask]
            center = np.mean(class_embeddings, axis=0)
            centers.append(center)
        
        return np.array(centers), unique_labels
    
    def _calculate_distance_threshold(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate the distance threshold for novelty detection."""
        distances = []
        
        # Calculate distances from each sample to its class center
        for label in self.known_class_labels:
            class_mask = labels == label
            class_embeddings = embeddings[class_mask]
            center = self.known_class_centers[self.known_class_labels == label][0]
            
            if self.distance_metric == 'cosine':
                # Normalize for cosine distance
                class_embeddings_norm = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
                center_norm = center / np.linalg.norm(center)
                class_distances = 1 - np.dot(class_embeddings_norm, center_norm)
            else:  # euclidean
                class_distances = np.linalg.norm(class_embeddings - center, axis=1)
            
            distances.extend(class_distances)
        
        distances = np.array(distances)
        
        if self.threshold_method == 'percentile':
            threshold = np.percentile(distances, self.threshold_percentile)
        elif self.threshold_method == 'std':
            threshold = np.mean(distances) + 2 * np.std(distances)
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        return threshold
    
    def _calculate_distances_to_centers(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate minimum distances from embeddings to known class centers."""
        distances = []
        
        for embedding in embeddings:
            center_distances = []
            
            for center in self.known_class_centers:
                if self.distance_metric == 'cosine':
                    # Cosine distance
                    embedding_norm = embedding / np.linalg.norm(embedding)
                    center_norm = center / np.linalg.norm(center)
                    distance = 1 - np.dot(embedding_norm, center_norm)
                else:  # euclidean
                    distance = np.linalg.norm(embedding - center)
                
                center_distances.append(distance)
            
            # Use minimum distance to any known class center
            distances.append(min(center_distances))
        
        return np.array(distances)
    
    def _cluster_with_kmeans(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Dict:
        """Cluster using K-means algorithm."""
        if n_clusters is None:
            # Determine optimal number of clusters using silhouette score
            best_score = -1
            best_k = 2
            
            for k in range(2, min(len(embeddings) // 2, self.max_clusters) + 1):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(embeddings, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            n_clusters = best_k
        
        # Final clustering with optimal k
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(embeddings, cluster_labels)
        else:
            silhouette = 0.0
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette
        }
    
    def _cluster_with_dbscan(self, embeddings: np.ndarray) -> Dict:
        """Cluster using DBSCAN algorithm."""
        # Determine epsilon using k-distance graph (simplified approach)
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=self.min_cluster_size).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        distances = np.sort(distances[:, -1])
        
        # Use knee point as epsilon (simplified heuristic)
        eps = np.percentile(distances, 90)
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        # Calculate cluster centers (for non-noise points)
        unique_labels = np.unique(cluster_labels)
        if -1 in unique_labels:  # Remove noise label
            unique_labels = unique_labels[unique_labels != -1]
        
        cluster_centers = []
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            center = np.mean(embeddings[cluster_mask], axis=0)
            cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])
        
        # Calculate silhouette score (excluding noise points)
        valid_labels = cluster_labels[cluster_labels != -1]
        valid_embeddings = embeddings[cluster_labels != -1]
        
        if len(np.unique(valid_labels)) > 1 and len(valid_labels) > 0:
            silhouette = silhouette_score(valid_embeddings, valid_labels)
        else:
            silhouette = 0.0
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'n_clusters': len(unique_labels),
            'silhouette_score': silhouette,
            'noise_points': np.sum(cluster_labels == -1)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to a file."""
        import pickle
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'known_class_centers': self.known_class_centers,
            'known_class_labels': self.known_class_labels,
            'distance_threshold': self.distance_threshold,
            'distance_metric': self.distance_metric,
            'threshold_method': self.threshold_method,
            'threshold_percentile': self.threshold_percentile
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> 'NoveltyDetectionSystem':
        """Load a fitted model from a file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.known_class_centers = model_data['known_class_centers']
        self.known_class_labels = model_data['known_class_labels']
        self.distance_threshold = model_data['distance_threshold']
        self.distance_metric = model_data['distance_metric']
        self.threshold_method = model_data['threshold_method']
        self.threshold_percentile = model_data['threshold_percentile']
        self.is_fitted = True
        
        self.logger.info(f"Model loaded from: {filepath}")
        return self


def create_novelty_detection_system(config: Dict = None) -> NoveltyDetectionSystem:
    """
    Factory function to create a NoveltyDetectionSystem with default or custom configuration.
    
    Args:
        config: Configuration dictionary with system parameters
        
    Returns:
        NoveltyDetectionSystem: Configured instance
    """
    default_config = {
        'distance_metric': 'cosine',
        'clustering_method': 'kmeans',
        'threshold_method': 'percentile',
        'threshold_percentile': 95.0,
        'min_cluster_size': 5,
        'max_clusters': 10,
        'random_state': 42
    }
    
    if config:
        default_config.update(config)
    
    return NoveltyDetectionSystem(**default_config)