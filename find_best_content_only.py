#!/usr/bin/env python3
"""
Script to find the best performing content-only runs from the systematic optimization sessions.
Content-only runs have remove: [headers, footers, quotes] in their configuration.
"""

import yaml
import joblib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_config(config_path: Path) -> Optional[Dict]:
    """Load and parse a YAML config file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        return None

def load_metrics(metrics_path: Path) -> Optional[Dict]:
    """Load metrics from joblib file."""
    try:
        return joblib.load(metrics_path)
    except Exception as e:
        print(f"Error loading metrics {metrics_path}: {e}")
        return None

def is_content_only(config: Dict) -> bool:
    """Check if this is a content-only run (removes headers, footers, quotes)."""
    dataset = config.get('dataset', {})
    remove_list = dataset.get('remove', [])
    
    # Check if remove list contains headers, footers, and quotes
    required_removals = {'headers', 'footers', 'quotes'}
    return required_removals.issubset(set(remove_list))

def get_model_type(config: Dict) -> str:
    """Extract model type from config."""
    model_target = config.get('model', {}).get('_target_', '')
    if 'lda' in model_target.lower():
        return 'LDA'
    elif 'kmeans' in model_target.lower():
        return 'PCA+K-means'
    else:
        return 'Unknown'

def extract_performance_metrics(metrics: Dict, model_type: str) -> Dict:
    """Extract test performance metrics for the model type."""
    result = {}
    
    if model_type == 'LDA':
        result['test_accuracy'] = metrics.get('LDA/Test Accuracy', [])
        result['test_nmi'] = metrics.get('LDA/Test NMI', [])
        result['test_ari'] = metrics.get('LDA/Test ARI', [])
    elif model_type == 'PCA+K-means':
        result['test_accuracy'] = metrics.get('PCA+K-means/Test Accuracy', [])
        result['test_nmi'] = metrics.get('PCA+K-means/Test NMI', [])
        result['test_ari'] = metrics.get('PCA+K-means/Test ARI', [])
    
    # Convert to final values (take last value if list, handle tuples)
    for key, value in result.items():
        if isinstance(value, list) and len(value) > 0:
            # Take the last value
            last_val = value[-1]
            # Handle tuples - extract the actual numeric value
            if isinstance(last_val, tuple) and len(last_val) > 0:
                result[key] = last_val[0]
            else:
                result[key] = last_val
        else:
            result[key] = None
    
    return result

def main():
    runs_dir = Path('/home/alex404/code/goal-apps/runs/single')
    
    # Find all run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    content_only_runs = []
    
    print("Scanning for content-only runs...")
    print(f"Found {len(run_dirs)} total run directories")
    
    for run_dir in run_dirs:
        config_path = run_dir / 'run-config.yaml'
        metrics_path = run_dir / 'metrics.joblib'
        
        # Skip if config or metrics don't exist
        if not config_path.exists() or not metrics_path.exists():
            continue
        
        # Load config
        config = load_config(config_path)
        if not config:
            continue
        
        # Check if content-only
        if not is_content_only(config):
            continue
        
        # Load metrics
        metrics = load_metrics(metrics_path)
        if not metrics:
            continue
        
        # Extract model type and performance
        model_type = get_model_type(config)
        performance = extract_performance_metrics(metrics, model_type)
        
        content_only_runs.append({
            'run_dir': str(run_dir),
            'model_type': model_type,
            'performance': performance,
            'config': config
        })
    
    print(f"\nFound {len(content_only_runs)} content-only runs")
    
    # Separate by model type and find best performances
    lda_runs = [r for r in content_only_runs if r['model_type'] == 'LDA']
    pca_kmeans_runs = [r for r in content_only_runs if r['model_type'] == 'PCA+K-means']
    
    print(f"\nLDA runs: {len(lda_runs)}")
    print(f"PCA+K-means runs: {len(pca_kmeans_runs)}")
    
    # Find best LDA run
    if lda_runs:
        valid_lda_runs = [r for r in lda_runs if r['performance']['test_accuracy'] is not None]
        if valid_lda_runs:
            best_lda = max(valid_lda_runs, key=lambda x: x['performance']['test_accuracy'])
            print(f"\n=== BEST LDA (Content-Only) ===")
            print(f"Run directory: {best_lda['run_dir']}")
            print(f"Test Accuracy: {best_lda['performance']['test_accuracy']:.3f} ({best_lda['performance']['test_accuracy']*100:.1f}%)")
            print(f"Test NMI: {best_lda['performance']['test_nmi']:.3f}")
            print(f"Test ARI: {best_lda['performance']['test_ari']:.3f}")
            print(f"Key parameters:")
            model_config = best_lda['config']['model']
            print(f"  - alpha: {model_config.get('alpha', 'unknown')}")
            print(f"  - beta: {model_config.get('beta', 'unknown')}")
            print(f"  - n_clusters: {model_config.get('n_clusters', 'unknown')}")
            print(f"  - max_iter: {model_config.get('max_iter', 'unknown')}")
            dataset_config = best_lda['config']['dataset']
            print(f"  - max_features: {dataset_config.get('max_features', 'unknown')}")
            print(f"  - min_df: {dataset_config.get('min_df', 'unknown')}")
            print(f"  - max_df: {dataset_config.get('max_df', 'unknown')}")
            print(f"  - use_count_vectorizer: {dataset_config.get('use_count_vectorizer', 'unknown')}")
    
    # Find best PCA+K-means run
    if pca_kmeans_runs:
        valid_pca_runs = [r for r in pca_kmeans_runs if r['performance']['test_accuracy'] is not None]
        if valid_pca_runs:
            best_pca = max(valid_pca_runs, key=lambda x: x['performance']['test_accuracy'])
            print(f"\n=== BEST PCA+K-means (Content-Only) ===")
            print(f"Run directory: {best_pca['run_dir']}")
            print(f"Test Accuracy: {best_pca['performance']['test_accuracy']:.3f} ({best_pca['performance']['test_accuracy']*100:.1f}%)")
            print(f"Test NMI: {best_pca['performance']['test_nmi']:.3f}")
            print(f"Test ARI: {best_pca['performance']['test_ari']:.3f}")
            print(f"Key parameters:")
            model_config = best_pca['config']['model']
            print(f"  - n_components: {model_config.get('n_components', 'unknown')}")
            print(f"  - n_clusters: {model_config.get('n_clusters', 'unknown')}")
            print(f"  - max_iter: {model_config.get('max_iter', 'unknown')}")
            print(f"  - algorithm: {model_config.get('algorithm', 'unknown')}")
            dataset_config = best_pca['config']['dataset']
            print(f"  - max_features: {dataset_config.get('max_features', 'unknown')}")
            print(f"  - min_df: {dataset_config.get('min_df', 'unknown')}")
            print(f"  - max_df: {dataset_config.get('max_df', 'unknown')}")
            print(f"  - use_count_vectorizer: {dataset_config.get('use_count_vectorizer', 'unknown')}")
    
    # Show top 5 performers for each model type
    print(f"\n=== TOP 5 LDA CONTENT-ONLY RUNS ===")
    if lda_runs:
        valid_lda_runs = [r for r in lda_runs if r['performance']['test_accuracy'] is not None]
        valid_lda_runs.sort(key=lambda x: x['performance']['test_accuracy'], reverse=True)
        for i, run in enumerate(valid_lda_runs[:5]):
            print(f"{i+1}. {Path(run['run_dir']).name}: {run['performance']['test_accuracy']*100:.1f}% accuracy")
    
    print(f"\n=== TOP 5 PCA+K-means CONTENT-ONLY RUNS ===")
    if pca_kmeans_runs:
        valid_pca_runs = [r for r in pca_kmeans_runs if r['performance']['test_accuracy'] is not None]
        valid_pca_runs.sort(key=lambda x: x['performance']['test_accuracy'], reverse=True)
        for i, run in enumerate(valid_pca_runs[:5]):
            print(f"{i+1}. {Path(run['run_dir']).name}: {run['performance']['test_accuracy']*100:.1f}% accuracy")

if __name__ == '__main__':
    main()