# Baseline Reproduction Results (5-Seed Runs)

**Date:** 2026-02-26

Reproduces the optimized baseline configs documented in `baseline_optimization_summary.md`, plus a new MNIST PCA+KMeans baseline. Each model was run 5× with seeds 0–4 (`model.random_state=i`, `model.pca_random_state=i`).

## Commands Used

```bash
# Newsgroups PCA+KMeans (content-only)
for i in 0 1 2 3 4; do
  goal train dataset=newsgroups-kmeans model=newsgroups-kmeans \
    model.random_state=$i model.pca_random_state=$i use_wandb=false
done

# Newsgroups PCA+KMeans (metadata, 8k features)
for i in 0 1 2 3 4; do
  goal train dataset=newsgroups-metadata-tfidf model=newsgroups-metadata-kmeans \
    model.random_state=$i model.pca_random_state=$i use_wandb=false
done

# Newsgroups LDA (content-only)
for i in 0 1 2 3 4; do
  goal train dataset=newsgroups-lda model=newsgroups-lda \
    model.random_state=$i use_wandb=false
done

# Newsgroups LDA (metadata)
for i in 0 1 2 3 4; do
  goal train dataset=newsgroups-metadata-count model=newsgroups-metadata-lda \
    model.random_state=$i use_wandb=false
done

# MNIST PCA+KMeans
for i in 0 1 2 3 4; do
  goal train dataset=mnist model=mnist-kmeans \
    model.random_state=$i model.pca_random_state=$i use_wandb=false
done
```

**Note:** Model params are nested under `model.` in the composed Hydra config, so overrides require the `model.` prefix.

---

## Results

### Newsgroups PCA+KMeans (content-only)
Config: `dataset=newsgroups-kmeans model=newsgroups-kmeans` (n_components=150, n_clusters=20)

| Seed | Test NMI | Test Accuracy | Test ARI |
|------|----------|---------------|----------|
| 0    | 0.2908   | 0.2608        | 0.0896   |
| 1    | 0.3296   | 0.2758        | 0.1099   |
| 2    | 0.3420   | 0.2930        | 0.0989   |
| 3    | 0.2836   | 0.2391        | 0.0689   |
| 4    | 0.3506   | 0.2933        | 0.1128   |
| **Mean ± σ** | **0.319 ± 0.030** | **0.272 ± 0.023** | **0.096 ± 0.018** |

---

### Newsgroups PCA+KMeans (metadata)
Config: `dataset=newsgroups-metadata-tfidf model=newsgroups-metadata-kmeans` (n_components=200, n_clusters=20, max_features=8000)

| Seed | Test NMI | Test Accuracy | Test ARI |
|------|----------|---------------|----------|
| 0    | 0.3243   | 0.2771        | 0.0889   |
| 1    | 0.3320   | 0.3060        | 0.0822   |
| 2    | 0.3046   | 0.2695        | 0.0809   |
| 3    | 0.2999   | 0.2604        | 0.0740   |
| 4    | 0.3265   | 0.2781        | 0.0773   |
| **Mean ± σ** | **0.317 ± 0.014** | **0.278 ± 0.017** | **0.081 ± 0.006** |

---

### Newsgroups LDA (content-only)
Config: `dataset=newsgroups-lda model=newsgroups-lda` (n_topics=20, max_iter=50, alpha=0.1, beta=0.3)

| Seed | Test NMI | Test Accuracy | Test ARI |
|------|----------|---------------|----------|
| 0    | 0.4246   | 0.4076        | 0.2643   |
| 1    | 0.4589   | 0.4048        | 0.3191   |
| 2    | 0.4341   | 0.4238        | 0.2754   |
| 3    | 0.4618   | 0.4392        | 0.3165   |
| 4    | 0.4461   | 0.3986        | 0.2865   |
| **Mean ± σ** | **0.445 ± 0.016** | **0.415 ± 0.017** | **0.292 ± 0.025** |

---

### Newsgroups LDA (metadata)
Config: `dataset=newsgroups-metadata-count model=newsgroups-metadata-lda` (n_topics=20, max_iter=50, alpha=1.0, beta=0.3)

| Seed | Test NMI | Test Accuracy | Test ARI |
|------|----------|---------------|----------|
| 0    | 0.4453   | 0.4229        | 0.2836   |
| 1    | 0.4869   | 0.4169        | 0.3461   |
| 2    | 0.4709   | 0.4465        | 0.3116   |
| 3    | 0.4822   | 0.4495        | 0.3332   |
| 4    | 0.4736   | 0.4114        | 0.3122   |
| **Mean ± σ** | **0.472 ± 0.016** | **0.429 ± 0.018** | **0.317 ± 0.024** |

---

### MNIST PCA+KMeans (new baseline)
Config: `dataset=mnist model=mnist-kmeans` (n_components=50, n_clusters=10, n_init=20)

| Seed | Test NMI | Test Accuracy | Test ARI |
|------|----------|---------------|----------|
| 0    | 0.4983   | 0.5171        | 0.3666   |
| 1    | 0.4984   | 0.5171        | 0.3666   |
| 2    | 0.4982   | 0.5170        | 0.3664   |
| 3    | 0.4985   | 0.5168        | 0.3665   |
| 4    | 0.4983   | 0.5170        | 0.3664   |
| **Mean ± σ** | **0.498 ± 0.000** | **0.517 ± 0.000** | **0.367 ± 0.000** |

MNIST results are essentially deterministic across seeds (variance < 0.0002), reflecting that PCA is deterministic and KMeans with n_init=20 converges reliably to the same solution.

---

## Summary Table

| Model | Dataset | NMI | Accuracy | ARI |
|-------|---------|-----|----------|-----|
| PCA+KMeans | Newsgroups (content) | 0.319 ± 0.030 | 27.2% ± 2.3% | 0.096 ± 0.018 |
| PCA+KMeans | Newsgroups (metadata) | 0.317 ± 0.014 | 27.8% ± 1.7% | 0.081 ± 0.006 |
| LDA | Newsgroups (content) | 0.445 ± 0.016 | 41.5% ± 1.7% | 0.292 ± 0.025 |
| LDA | Newsgroups (metadata) | 0.472 ± 0.016 | 42.9% ± 1.8% | 0.317 ± 0.024 |
| PCA+KMeans | MNIST | 0.498 ± 0.000 | 51.7% ± 0.0% | 0.367 ± 0.000 |

---

## Notes

- **Config fix:** `newsgroups-metadata-tfidf.yaml` was updated from `max_features: 10000` to `max_features: 8000` per the optimization summary.
- **New config:** `config/hydra/model/mnist-kmeans.yaml` created for the MNIST baseline.
- **Metadata advantage (LDA):** Metadata adds ~2.7% NMI and ~1.5% accuracy over content-only for LDA.
- **Metadata advantage (PCA+KMeans):** Negligible after 5-seed averaging — high variance obscures any gain.
- **LDA vs PCA+KMeans gap:** LDA outperforms PCA+KMeans by ~13% NMI and ~15% accuracy on Newsgroups, consistent with the optimization report.
