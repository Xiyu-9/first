# NoveltyDetectionSystem for Graph Classification

This module provides a comprehensive novelty detection system for identifying and clustering novel classes in graph classification tasks that were not seen during training.

## Features

### Core Capabilities
- **Distance-based novelty detection** using cosine similarity or Euclidean distance
- **Adaptive threshold determination** using percentile or standard deviation methods
- **Novel class clustering** with K-means or DBSCAN algorithms
- **Comprehensive evaluation metrics** including precision, recall, F1-score, and accuracy
- **Visualization support** with t-SNE plots and cluster analysis
- **Model persistence** with save/load functionality

### Integration with GCLC
- Seamless integration with the Graph Classification with Contrastive Learning (GCLC) framework
- Works with any embedding space produced by graph neural networks
- Compatible with existing training pipelines and evaluation metrics

## Quick Start

```python
from novelty_detection_system import NoveltyDetectionSystem
import numpy as np

# Initialize the system
nds = NoveltyDetectionSystem(
    distance_metric='cosine',
    clustering_method='kmeans',
    threshold_percentile=95.0
)

# Fit on known training data
nds.fit(known_embeddings, known_labels)

# Detect novel samples
is_novel, distances = nds.predict_novelty(test_embeddings)

# Cluster novel samples to discover new classes
novel_embeddings = test_embeddings[is_novel]
clustering_results = nds.cluster_novel_samples(novel_embeddings)

# Evaluate performance
metrics = nds.evaluate_novelty_detection(
    test_embeddings, test_labels, known_class_labels
)

print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Discovered {clustering_results['n_clusters']} new classes")
```

## API Reference

### NoveltyDetectionSystem

#### Constructor Parameters
- `distance_metric` (str): Distance metric ('cosine', 'euclidean')
- `clustering_method` (str): Clustering algorithm ('kmeans', 'dbscan')
- `threshold_method` (str): Threshold calculation method ('percentile', 'std')
- `threshold_percentile` (float): Percentile for threshold (default: 95.0)
- `min_cluster_size` (int): Minimum cluster size for DBSCAN (default: 5)
- `max_clusters` (int): Maximum clusters for K-means (default: 10)
- `random_state` (int): Random seed for reproducibility (default: 42)

#### Core Methods

##### `fit(embeddings, labels)`
Fit the system on known training data.
- **embeddings**: Training embeddings (n_samples, n_features)
- **labels**: Training labels (n_samples,)

##### `predict_novelty(embeddings)`
Predict which samples are novel.
- **embeddings**: Test embeddings (n_samples, n_features)
- **Returns**: (is_novel, distances) - boolean array and distances to nearest centers

##### `cluster_novel_samples(novel_embeddings, n_clusters=None)`
Cluster novel samples to discover new classes.
- **novel_embeddings**: Embeddings of novel samples
- **n_clusters**: Number of clusters (auto-determined if None)
- **Returns**: Dictionary with clustering results

##### `evaluate_novelty_detection(embeddings, true_labels, known_classes)`
Evaluate novelty detection performance.
- **embeddings**: Test embeddings
- **true_labels**: True labels
- **known_classes**: List of known class labels
- **Returns**: Dictionary with evaluation metrics

##### `visualize_novelty_detection(embeddings, labels, is_novel, save_path=None)`
Create t-SNE visualization of novelty detection results.

##### `save_model(filepath)` / `load_model(filepath)`
Save/load fitted model for reuse.

## Integration with GCLC

### Method 1: Direct Integration

Add to your `gclc.py` training loop:

```python
from novelty_detection_system import NoveltyDetectionSystem

# After training, in the test function:
def test(test_loader, epoch, log_file="metrics.log"):
    # ... existing code ...
    
    # Extract final embeddings and labels
    all_embeddings = torch.cat(embeds_instance, dim=0).cpu().numpy()
    all_labels = torch.cat(all_true_labels, dim=0).cpu().numpy()
    
    # Apply novelty detection on final epoch
    if epoch == args.epochs - 1:
        nds = NoveltyDetectionSystem()
        nds.fit(train_embeddings, train_labels)  # You need train data here
        
        is_novel, distances = nds.predict_novelty(all_embeddings)
        metrics = nds.evaluate_novelty_detection(
            all_embeddings, all_labels, known_classes
        )
        
        # Log results
        with open(log_file, 'a') as f:
            f.write(f"\nNovelty Detection Results:\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
```

### Method 2: Post-Training Analysis

Use the complete integration example:

```python
from integration_example import integrate_novelty_detection_with_gclc

# After training your model
results = integrate_novelty_detection_with_gclc(
    model=trained_model,
    train_loader=train_loader,
    test_loader=test_loader
)
```

## Configuration Options

### Distance Metrics
- **cosine**: Cosine similarity (recommended for normalized embeddings)
- **euclidean**: Euclidean distance (good for general use)

### Clustering Methods
- **kmeans**: K-means clustering (faster, requires specifying k)
- **dbscan**: DBSCAN clustering (automatic cluster count, density-based)

### Threshold Methods
- **percentile**: Use percentile of training distances (recommended)
- **std**: Use mean + 2*std of training distances

## Examples

### Basic Usage
See `test_novelty_detection.py` for a complete working example with synthetic data.

### Integration Example
See `integration_example.py` for detailed integration guidance with GCLC.

### Advanced Configuration
```python
# High-precision configuration
nds = NoveltyDetectionSystem(
    distance_metric='cosine',
    clustering_method='dbscan',
    threshold_method='percentile',
    threshold_percentile=98.0,  # More conservative threshold
    min_cluster_size=10,        # Larger minimum clusters
    max_clusters=20            # Allow more clusters
)

# Fast configuration for large datasets
nds = NoveltyDetectionSystem(
    distance_metric='euclidean',
    clustering_method='kmeans',
    threshold_method='std',
    max_clusters=5             # Fewer clusters for speed
)
```

## Performance Considerations

- **Memory**: Embeddings are stored in memory. For large datasets, consider batch processing.
- **Computation**: t-SNE visualization is limited to 1000 samples for performance.
- **Clustering**: DBSCAN is slower but more flexible than K-means.

## Dependencies

- numpy
- scikit-learn
- matplotlib
- torch (for integration with GCLC)

## File Structure

```
Graph_Classification/
├── novelty_detection_system.py    # Main implementation
├── test_novelty_detection.py      # Test suite
├── integration_example.py         # Integration guide
├── evu.py                         # Training utilities
├── cluster_analysis.py            # Clustering analysis tools
└── gclc.py                        # Main GCLC framework (updated)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
2. **Memory Error**: Reduce dataset size or use batch processing
3. **Low Performance**: Adjust threshold_percentile or try different distance metrics
4. **No Novel Samples**: Lower threshold_percentile or check data distribution

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.INFO)
# NoveltyDetectionSystem will now output detailed logs
```

## Citation

If you use this NoveltyDetectionSystem in your research, please cite:

```bibtex
@misc{novelty_detection_system,
  title={NoveltyDetectionSystem for Graph Classification},
  year={2024},
  note={Integrated with GCLC framework}
}
```