# MFA Benchmark: gans-n-gmms on MNIST and NG20

**Date:** 2026-02-26

Implementation: `scratch/mfa_benchmark.py` — a faithful JAX replication of the gans-n-gmms MFA:
- Classic parameterization (log_π, μ, A, √D)
- Woodbury / matrix-determinant lemma for log-likelihood
- km + per-cluster sklearn FactorAnalysis initialization
- Adam, LR=2e-5, batch_size=512

All results are 3 seeds (2 for the larger/longer configs), reported as mean ± std.

---

## MNIST (D=784)

Paper config: K=150, L=5, 6000 iters.

| Config | Seeds | Test NMI | Test LL/sample |
|--------|-------|----------|----------------|
| K=10, L=5, 6k iters | 3 | 0.301 ± 0.004 | 2248 |
| K=50, L=5, 6k iters | 3 | 0.473 ± 0.002 | 2489 |
| **K=150, L=5, 6k iters (paper)** | **3** | **0.488 ± 0.002** | **2509** |
| K=150, L=5, 12k iters | 2 | 0.478 ± 0.005 | 2472 |

**Observations:**
- NMI grows with K (more components → finer-grained clustering) but with diminishing returns past K=50.
- **Longer training does not help.** 12k iters gives slightly lower NMI and LL than 6k — 6000 iters is already at convergence. The paper's 6000-iter budget is appropriate.
- Each K=150 run takes ~260–290s (≈4.5 min/seed).

---

## NG20 (D=5000 TF-IDF, content-only)

Applied same architecture to 20 Newsgroups. TF-IDF features with max_features=5000, min_df=2, max_df=0.9 (matching the content-only PCA+KMeans baseline). 20 ground-truth classes.

| Config | Seeds | Test NMI | Test LL/sample |
|--------|-------|----------|----------------|
| K=20, L=5, 6k iters | 3 | 0.164 ± 0.003 | 17415 |
| K=20, L=5, 12k iters | 3 | 0.164 ± 0.003 | 17400 |
| K=20, L=10, 6k iters | 3 | 0.158 ± 0.006 | 17323 |
| K=150, L=5, 6k iters | 2 | 0.168 ± 0.008 | 17789 |

**Observations:**
- All configs plateau at ~0.16–0.17 NMI regardless of K or L. The model is not learning the newsgroup structure.
- Longer training (12k iters) is again flat — convergence happens quickly.
- L=10 vs L=5: marginally *lower* NMI (0.158 vs 0.164) with higher variance — the richer model gets worse with only ~565 samples per cluster.
- K=150 (paper config) on NG20 achieves the best LL (more components = better generative fit) but essentially the same NMI as K=20.

---

## Comparison against other baselines (NG20)

| Model | Test NMI | Notes |
|-------|----------|-------|
| PCA+KMeans (content) | 0.319 ± 0.030 | D=10000 TF-IDF → 150 PCA components |
| LDA (content) | 0.445 ± 0.016 | count features, 20 topics |
| **MFA K=150, L=5** | **0.168 ± 0.008** | D=5000 TF-IDF, Gaussian model |
| **MFA K=20, L=5** | **0.164 ± 0.003** | D=5000 TF-IDF, Gaussian model |

MFA falls well short of both baselines on text data. This is unsurprising: TF-IDF vectors are sparse and non-negative, while MFA assumes dense Gaussian observations. LDA's Dirichlet-multinomial model is a much better fit for bag-of-words structure.

---

## Comparison against other baselines (MNIST)

| Model | Test NMI | Notes |
|-------|----------|-------|
| PCA+KMeans (K=10) | 0.498 ± 0.000 | D=784 → 50 PCA, K=10 |
| **MFA K=150, L=5 (paper)** | **0.488 ± 0.002** | D=784, K=150 |
| **MFA K=10, L=5** | **0.301 ± 0.004** | D=784, K=10 |

On MNIST, MFA K=150 is competitive with PCA+KMeans (0.488 vs 0.498). However MFA uses K=150 components to achieve this — when matched to K=10 (one per class), MFA at 0.301 is much weaker than PCA+KMeans at 0.498. The high-K MFA result partly reflects the model using finer-grained sub-clusters.

---

## Summary

- **6000 iters is the right budget** — confirmed on both datasets.
- **MFA is competitive on image data** (MNIST NMI ~0.49 at K=150) but **fails on text** (NG20 NMI ~0.16–0.17), where the Gaussian generative assumption is a poor fit.
- For NG20, varying K (20→150) and L (5→10) makes no meaningful difference — the bottleneck is the model family, not capacity.
- The paper's primary use case (image generation) is well-suited to MFA; applying it to bag-of-words text is a mismatch.
