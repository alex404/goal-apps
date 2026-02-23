# TASIC Dataset Clustering Problem Analysis

## Overview

The TASIC dataset represents a fundamental challenge in single-cell transcriptomics: identifying distinct cell types based on gene expression profiles. This is **not** a simple region classification problem, but rather a complex multi-class clustering task that aims to recover the biological cell type taxonomy discovered by Tasic et al. (2018).

## The Clustering Problem

**Dataset**: Tasic et al. (2018) single-cell RNA-seq data from mouse neocortex (GSE115746)
- **Cells**: ~23,822 single cells from two brain regions (VISp and ALM)  
- **Features**: ~45,000 genes (reduced to 3,000 highly variable genes)
- **Ground Truth**: 133 distinct transcriptomic cell types identified by the original study
- **Challenge**: Recover these 133 cell types from gene expression data alone

## Biological Context

The 133 cell types fall into major categories:
- **Glutamatergic neurons** (56 types): Excitatory projection neurons, mostly area-specific
- **GABAergic neurons** (61 types): Inhibitory interneurons, mostly shared across areas  
- **Non-neuronal cells** (16 types): Astrocytes, oligodendrocytes, microglia, etc.

Key insight from the paper: Most GABAergic interneurons are shared across brain regions, while most glutamatergic neurons are area-specific. This creates a complex clustering landscape where some cell types are region-invariant while others are region-specific.

## Methodological Implications

**For PCA+K-means**: 
- Must handle high-dimensional gene expression (3,000 features)
- Log transformation + standardization essential for scRNA-seq normalization
- PCA dimensionality reduction critical given high noise in scRNA-seq data
- K=133 clusters presents significant challenge for K-means algorithm

**For LDA Topic Modeling**:
- Can leverage count nature of RNA-seq data directly (no log transformation)
- 133 "topics" (cell types) must be learned from gene expression "documents" (cells)
- Each cell can have mixed membership across cell types (biologically relevant)
- Hyperparameters α and β control sparsity of cell-topic and topic-gene distributions

## Baseline Performance Expectations

This is a challenging clustering task. Random assignment would yield ~0.75% accuracy (1/133). Strong baselines should significantly exceed this, but perfect clustering is unlikely given:
- Technical noise in scRNA-seq measurements
- Continuous variation within cell types (as noted in the original paper)
- Potential batch effects and dropout events
- Intermediate cells that bridge multiple types

The goal is to establish optimized PCA+K-means and LDA baselines that can serve as fair comparisons for more sophisticated methods like HMoG, while respecting the true biological complexity of cortical cell type diversity.