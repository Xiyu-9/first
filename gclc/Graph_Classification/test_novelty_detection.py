#!/usr/bin/env python3
"""
Test script for NoveltyDetectionSystem
"""
import numpy as np
import matplotlib.pyplot as plt
from novelty_detection_system import NoveltyDetectionSystem

def test_novelty_detection_system():
    """Test the basic functionality of NoveltyDetectionSystem"""
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 128
    n_known_classes = 5
    n_novel_classes = 2
    
    # Generate known class data (5 classes)
    known_embeddings = []
    known_labels = []
    
    for class_id in range(n_known_classes):
        # Generate cluster centers
        center = np.random.randn(n_features) * 2
        class_samples = center + np.random.randn(n_samples // n_known_classes, n_features) * 0.5
        known_embeddings.append(class_samples)
        known_labels.extend([class_id] * (n_samples // n_known_classes))
    
    known_embeddings = np.vstack(known_embeddings)
    known_labels = np.array(known_labels)
    
    # Generate novel class data (2 new classes)
    novel_embeddings = []
    novel_labels = []
    
    for class_id in range(n_known_classes, n_known_classes + n_novel_classes):
        # Generate cluster centers far from known classes
        center = np.random.randn(n_features) * 5 + 10  # Shift far away
        class_samples = center + np.random.randn(100, n_features) * 0.5
        novel_embeddings.append(class_samples)
        novel_labels.extend([class_id] * 100)
    
    novel_embeddings = np.vstack(novel_embeddings)
    novel_labels = np.array(novel_labels)
    
    # Test NoveltyDetectionSystem
    print("Testing NoveltyDetectionSystem...")
    
    # Initialize and fit the system
    nds = NoveltyDetectionSystem(
        distance_metric='cosine',
        clustering_method='kmeans',
        threshold_percentile=95.0
    )
    
    print("Fitting on known classes...")
    nds.fit(known_embeddings, known_labels)
    
    # Test on mixed data (known + novel)
    all_test_embeddings = np.vstack([known_embeddings[:100], novel_embeddings])
    all_test_labels = np.concatenate([known_labels[:100], novel_labels])
    
    print("Predicting novelty...")
    is_novel, distances = nds.predict_novelty(all_test_embeddings)
    
    # Evaluate
    known_class_labels = list(range(n_known_classes))
    metrics = nds.evaluate_novelty_detection(
        all_test_embeddings, all_test_labels, known_class_labels
    )
    
    print("\nNovelty Detection Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    
    # Test clustering of novel samples
    novel_mask = is_novel
    if np.sum(novel_mask) > 0:
        print(f"\nClustering {np.sum(novel_mask)} novel samples...")
        novel_only_embeddings = all_test_embeddings[novel_mask]
        clustering_results = nds.cluster_novel_samples(novel_only_embeddings)
        
        print(f"Discovered {clustering_results['n_clusters']} novel clusters")
        print(f"Silhouette Score: {clustering_results['silhouette_score']:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    nds.visualize_novelty_detection(
        all_test_embeddings, all_test_labels, is_novel,
        save_path='test_novelty_detection.png'
    )
    
    print("Test completed successfully!")
    return nds, metrics

if __name__ == "__main__":
    nds, metrics = test_novelty_detection_system()