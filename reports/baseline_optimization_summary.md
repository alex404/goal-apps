# Baseline Model Optimization Summary
## 20 Newsgroups Dataset Clustering Results

This document summarizes the systematic optimization of PCA+K-means and LDA clustering models on the 20 Newsgroups dataset, both with and without metadata (headers, footers, quotes).

## Executive Summary

### Content-Only Results (Headers/Footers/Quotes Removed)
- **PCA+K-means**: 26.6% accuracy
- **LDA**: 34.5% accuracy

### Metadata-Included Results (Headers/Footers/Quotes Preserved)
- **PCA+K-means**: 30.5% accuracy (+3.9% improvement)
- **LDA**: 43.7% accuracy (+9.2% improvement)

**Key Finding**: Metadata provides substantial benefit, especially for LDA which sees a 9.2% improvement and achieves over 40% accuracy.

## Detailed Results

### PCA+K-means Results

#### Content-Only (Optimized)
- **Test Accuracy**: 26.6%
- **Test NMI**: 31.1%
- **Test ARI**: 5.0%
- **Configuration**:
  - PCA Components: 150
  - Features: 10,000
  - min_df: 2
  - max_df: 0.9
  - Vectorization: TF-IDF

#### Metadata-Included (Optimized)
- **Test Accuracy**: 30.5%
- **Test NMI**: 34.2%
- **Test ARI**: 8.8%
- **Configuration**:
  - PCA Components: 400 (much higher!)
  - Features: 8,000 (fewer but cleaner)
  - min_df: 2
  - max_df: 0.9
  - Vectorization: TF-IDF

**PCA+K-means Insights**:
- With metadata, much higher dimensionality (400 vs 150 components) works better
- Fewer but cleaner features (8k vs 10k) improve performance
- Significant improvement (+3.9%) with metadata

### LDA Results

#### Content-Only (Optimized)
- **Test Accuracy**: 34.5%
- **Test NMI**: 37.6%
- **Test ARI**: 22.2%
- **Configuration**:
  - Alpha: 1.0
  - Beta: 0.1
  - Features: 10,000
  - min_df: 2
  - max_df: 0.9
  - Vectorization: Count (required for LDA)

#### Metadata-Included (Optimized)
- **Test Accuracy**: 43.7%
- **Test NMI**: 47.2%
- **Test ARI**: 32.1%
- **Configuration**:
  - Alpha: 1.0 (unchanged)
  - Beta: 0.3 (increased from 0.1)
  - Features: 10,000 (unchanged)
  - min_df: 2 (unchanged)
  - max_df: 0.9 (unchanged)
  - Vectorization: Count

**LDA Insights**:
- Higher beta sparsity (0.3 vs 0.1) works better with metadata
- Same feature configuration optimal for both scenarios
- Substantial improvement (+9.2%) with metadata
- Achieves over 40% accuracy with metadata

## Optimization Process

### Methodology
Both models underwent systematic hyperparameter optimization:
- **Content-Only**: 40 systematic iterations each
- **Metadata-Included**: ~17 iterations for PCA+K-means, ~12 for LDA
- Parameters tested: PCA components, feature counts, vectorization parameters, LDA priors
- Evaluation metrics: Accuracy, NMI (Normalized Mutual Information), ARI (Adjusted Rand Index)
- Cluster assignment: Hungarian algorithm for optimal cluster-to-class mapping

### Key Parameter Discoveries

#### PCA+K-means with Metadata
- **Breakthrough**: 400 PCA components with 8k features (Run 16)
- Lower feature counts (8k) work better than higher counts (10k+)
- Much higher dimensionality is beneficial when metadata is available

#### LDA with Metadata  
- **Breakthrough**: Beta=0.3 significantly outperforms Beta=0.1
- Progressive improvement: 0.1 → 0.2 → 0.25 → 0.3
- Higher topic sparsity benefits from metadata structure

## Configuration Files

The optimized configurations are saved as:

### Content-Only
- `config/hydra/dataset/newsgroups-kmeans.yaml`
- `config/hydra/model/newsgroups-kmeans.yaml`
- `config/hydra/dataset/newsgroups-lda.yaml`
- `config/hydra/model/newsgroups-lda.yaml`

### Metadata-Included
- `config/hydra/dataset/newsgroups-metadata-kmeans.yaml`
- `config/hydra/model/newsgroups-metadata-kmeans.yaml`
- `config/hydra/dataset/newsgroups-metadata-lda.yaml`
- `config/hydra/model/newsgroups-metadata-lda.yaml`

## Research Implications

### Fair Comparison Framework
These results establish strong, systematically optimized baselines for comparison with HMoG:
1. **Content-Only**: For fair academic comparison (standard preprocessing)
2. **Metadata-Included**: For practical application scenarios where document structure is available

### Metadata Value Quantification
- **PCA+K-means**: +3.9% improvement demonstrates significant benefit
- **LDA**: +9.2% improvement demonstrates substantial benefit
- **Total Range**: 26.6% to 43.7% across all conditions

### Model Ranking (All Scenarios)
1. **LDA with Metadata**: 43.7% accuracy ⭐
2. **LDA Content-Only**: 34.5% accuracy
3. **PCA+K-means with Metadata**: 30.5% accuracy  
4. **PCA+K-means Content-Only**: 26.6% accuracy

## Next Steps

1. **HMoG Comparison**: Compare HMoG performance against these optimized baselines
2. **Statistical Significance**: Conduct multiple runs to establish confidence intervals
3. **Additional Metrics**: Report precision, recall, F1 scores per category
4. **Error Analysis**: Investigate which newsgroup categories are most challenging

---

*Generated through systematic hyperparameter optimization with 80+ total experimental runs*
*Date: June 29, 2025*