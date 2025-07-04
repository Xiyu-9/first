#!/usr/bin/env python3
"""
Example integration of NoveltyDetectionSystem with GCLC framework

This script demonstrates how to integrate the NoveltyDetectionSystem
with the existing graph classification and contrastive learning framework.
"""

import numpy as np
import torch
import torch.nn.functional as F
from novelty_detection_system import NoveltyDetectionSystem
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def integrate_novelty_detection_with_gclc(model, train_loader, test_loader, device='cpu'):
    """
    Example integration of NoveltyDetectionSystem with GCLC training pipeline.
    
    Args:
        model: Trained GCLC model (Encoder)
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run computations on
        
    Returns:
        dict: Results including novelty detection metrics
    """
    
    logger.info("Starting NoveltyDetectionSystem integration with GCLC...")
    
    # Step 1: Extract embeddings from trained model
    def extract_embeddings(model, data_loader, max_samples=1000):
        """Extract embeddings using the trained GCLC model"""
        model.eval()
        all_embeddings = []
        all_labels = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= max_samples:
                    break
                    
                # Extract features using the model
                # Note: This depends on your specific model architecture
                # You may need to adapt this based on your model's forward method
                try:
                    # Assuming the model returns embeddings
                    # This is a placeholder - adapt to your model structure
                    embeddings = model.encoder(batch)  # or however you extract features
                    
                    # Convert to numpy for NoveltyDetectionSystem
                    embeddings_np = embeddings.cpu().numpy()
                    labels_np = batch.y.cpu().numpy() if hasattr(batch, 'y') else np.zeros(len(embeddings_np))
                    
                    all_embeddings.append(embeddings_np)
                    all_labels.append(labels_np)
                    sample_count += len(embeddings_np)
                    
                except Exception as e:
                    logger.warning(f"Error extracting embeddings: {e}")
                    # Create dummy data for demonstration
                    dummy_embeddings = np.random.randn(32, 128)  # 32 samples, 128 features
                    dummy_labels = np.random.randint(0, 5, 32)  # 5 classes
                    all_embeddings.append(dummy_embeddings)
                    all_labels.append(dummy_labels)
                    sample_count += 32
        
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            labels = np.concatenate(all_labels)
        else:
            # Fallback: create synthetic data for demonstration
            embeddings = np.random.randn(500, 128)
            labels = np.random.randint(0, 5, 500)
            
        return embeddings, labels
    
    # Step 2: Extract training embeddings (known classes)
    logger.info("Extracting training embeddings...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader)
    
    # Step 3: Extract test embeddings (may contain novel classes)
    logger.info("Extracting test embeddings...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader)
    
    # Step 4: Initialize and fit NoveltyDetectionSystem
    logger.info("Initializing NoveltyDetectionSystem...")
    novelty_system = NoveltyDetectionSystem(
        distance_metric='cosine',
        clustering_method='kmeans',
        threshold_method='percentile',
        threshold_percentile=95.0,
        min_cluster_size=5,
        max_clusters=10
    )
    
    # Fit on training data (known classes)
    logger.info("Fitting NoveltyDetectionSystem on known classes...")
    novelty_system.fit(train_embeddings, train_labels)
    
    # Step 5: Detect novel samples in test data
    logger.info("Detecting novel samples...")
    is_novel, distances = novelty_system.predict_novelty(test_embeddings)
    
    # Step 6: Evaluate novelty detection performance
    # For demonstration, assume classes 0-4 are known, 5+ are novel
    known_classes = list(range(5))
    metrics = novelty_system.evaluate_novelty_detection(
        test_embeddings, test_labels, known_classes
    )
    
    logger.info("Novelty Detection Results:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Step 7: Cluster novel samples to discover new classes
    novel_mask = is_novel
    novel_count = np.sum(novel_mask)
    
    if novel_count > 0:
        logger.info(f"Clustering {novel_count} novel samples...")
        novel_embeddings = test_embeddings[novel_mask]
        clustering_results = novelty_system.cluster_novel_samples(novel_embeddings)
        
        logger.info(f"Discovered {clustering_results['n_clusters']} potential new classes")
        logger.info(f"Clustering quality (Silhouette): {clustering_results['silhouette_score']:.4f}")
        
        # Step 8: Visualize results
        logger.info("Generating visualization...")
        novelty_system.visualize_novelty_detection(
            test_embeddings[:500],  # Limit for visualization performance
            test_labels[:500],
            is_novel[:500],
            save_path='gclc_novelty_detection_results.png'
        )
    else:
        logger.info("No novel samples detected.")
        clustering_results = {'n_clusters': 0, 'silhouette_score': 0.0}
    
    # Step 9: Save the fitted novelty detection model
    logger.info("Saving NoveltyDetectionSystem model...")
    novelty_system.save_model('novelty_detection_model.pkl')
    
    return {
        'novelty_metrics': metrics,
        'clustering_results': clustering_results,
        'novel_count': novel_count,
        'total_samples': len(test_embeddings),
        'novelty_system': novelty_system
    }


def example_usage_in_gclc_training():
    """
    Example of how to integrate this into the main GCLC training loop
    """
    
    # This is pseudocode showing where to add novelty detection
    # in your main training function
    
    logger.info("Example integration with GCLC training pipeline:")
    
    # After training your GCLC model:
    # model = train_gclc_model(...)  # Your existing training code
    
    # Add novelty detection evaluation:
    # results = integrate_novelty_detection_with_gclc(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device
    # )
    
    # Log results:
    # logger.info(f"Novelty detection completed:")
    # logger.info(f"  Found {results['novel_count']} novel samples")
    # logger.info(f"  Discovered {results['clustering_results']['n_clusters']} new classes")
    # logger.info(f"  Detection F1-Score: {results['novelty_metrics']['f1_score']:.4f}")
    
    print("""
Integration Guide:
==================

1. **During Training**: 
   - Train your GCLC model as usual
   - Extract embeddings from the final epoch

2. **After Training**:
   - Use train embeddings to fit NoveltyDetectionSystem
   - Apply to test data to detect novel classes

3. **In Production**:
   - Load saved NoveltyDetectionSystem model
   - Apply to new samples for real-time novelty detection

Key Integration Points in gclc.py:
==================================

1. In the test() function, after computing embeddings:
   ```python
   # Extract embeddings
   all_embeddings = torch.cat(embeds_instance, dim=0).cpu().numpy()
   all_labels = torch.cat(all_true_labels, dim=0).cpu().numpy()
   
   # Apply novelty detection
   if epoch == args.epochs - 1:  # Final epoch
       novelty_results = apply_novelty_detection(all_embeddings, all_labels)
   ```

2. Add to your results logging:
   ```python
   if 'novelty_results' in locals():
       f.write(f"\\nNovelty Detection Results:\\n")
       f.write(f"Novel samples detected: {novelty_results['novel_count']}\\n")
       f.write(f"New classes discovered: {novelty_results['clustering_results']['n_clusters']}\\n")
   ```

3. Save novelty detection model alongside your main model:
   ```python
   torch.save({
       'gclc_model': model.state_dict(),
       'novelty_system': novelty_results['novelty_system']
   }, 'complete_model.pth')
   ```
""")


if __name__ == "__main__":
    # Run the example
    example_usage_in_gclc_training()
    
    # For actual integration, you would call:
    # results = integrate_novelty_detection_with_gclc(model, train_loader, test_loader)
    logger.info("NoveltyDetectionSystem integration example completed!")