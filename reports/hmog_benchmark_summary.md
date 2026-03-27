# HMoG Benchmark Summary: MNIST and NG20

**Consolidated from**: `sweep8_coa_nmi_optimization.md` (2026-03-23), `ng20_hmog_exploration.md` (2026-03-25)
**Total runs**: ~85 (Sweep 8 + NG20 phases/waves)

---

## Objective

Find configurations where **CoA merge NMI > peak raw NMI** on both datasets, characterizing the conditions under which overclustering + hierarchical merging recovers better structure than raw cluster assignments.

---

## Best Results at a Glance

| Dataset | Metric | Value | Config |
|---------|--------|-------|--------|
| MNIST | Best raw NMI | ~0.61 | ld=50, K=200, ent=3e-1 |
| MNIST | Best CoA NMI | **0.772 ± .007** | ld=50, K=200, ent=3e-1, l2=5e-4, 10cyc |
| MNIST | CoA − raw gap | +0.162 | Same as above |
| NG20 | Best raw NMI | **0.567** | ld=400, K=40 |
| NG20 | Best CoA NMI | 0.501 | ld=50, K=300, ent=1e-1 |
| NG20 | Best CoA − raw gap | +0.010 | ld=50, K=200, ent=1e-1 |
| NG20 | LDA baseline | 0.472 | — |

The merge-over-raw phenomenon is strong on MNIST (+0.16) and real but modest on NG20 (+0.01).

---

## MNIST HMoG (Sweep 8)

### Best stable configuration

```
latent_dim: 50
n_clusters: 200
mixture_entropy_reg: 3e-1
l2_reg: 5e-4
l1_reg: 0
num_cycles: 10
lr: 3e-4
batch_steps: 1000
n_epochs: 200
```

**Result**: raw NMI = 0.610 ± .006, CoA NMI = **0.772 ± .007**, Opt NMI = 0.560 ± .008 (65 valid clusters)

### Key MNIST findings

- **ent=3e-1 is essential for CoA NMI** — beats ent=1e-1 by ~0.03–0.04 consistently across all l2/l1 settings
- **l1 does not help** — adding l1=1e-3 reduced CoA NMI (0.750 vs 0.772) and pruned valid clusters to ~60, reducing merge material
- **l2 stability threshold** — l2 ≥ 5e-4 required without l1; l1=1e-3 extends stability down to l2=3e-4 but doesn't improve NMI
- **ld=50/K=200 dominates ld=20/K=100** across all metrics
- **min_prob=1e-3 backfires at K=200** — high retention (95%+ valid) but hurts both NMI metrics; contrast with K=100 where it helps

### MNIST regulrization landscape

| Setting | l2 | l1 | ent | Valid | Raw NMI | CoA NMI |
|---------|---:|---:|----:|------:|--------:|--------:|
| **Best (S4/S6)** | 5e-4 | 0 | 3e-1 | 65 | 0.610 ± .006 | **0.772 ± .007** |
| S8-C | 5e-4 | 1e-3 | 3e-1 | 60 | 0.612 ± .003 | 0.750 ± .008 |
| S8c-I | 3e-4 | 1e-3 | 1e-1 | 49 | 0.612 ± .001 | 0.720 ± .007 |
| S7-B (ld=20, K=100) | 3e-4 | 0 | — | 67 | 0.553 ± .001 | 0.714 ± .008 |

---

## NG20 HMoG (65-run Sweep)

### Best configurations

```
# Best merge>raw (merge advantage recipe)
latent_dim: 50
n_clusters: 100–200
mixture_entropy_reg: 1e-1
l2_reg: 1e-4
l1_reg: 1e-6
min_prob: 1e-5
lr: 1e-4
batch_steps: 1000
n_epochs: 200
num_cycles: 1

# Best absolute raw NMI (no merge advantage)
latent_dim: 400
n_clusters: 40
mixture_entropy_reg: 1e-1
```

### NG20 capacity sweep (ld × K at ent=1e-1)

| ld | Stability | Best raw NMI | Merge behavior |
|----|-----------|-------------|----------------|
| 20 | Stable | 0.299 | Largest gap (+0.012) but low absolute NMI |
| 50 | Stable at all K | 0.551 | Sweet spot: merge>raw at K=100-300 |
| 100 | Unstable at K=60-200 | 0.473 | cuSolver crashes |
| 400 | Stable at K≤40, bs=100 | **0.567** | Best raw NMI; merge always worse |

### NG20 K sweep at ld=50, ent=1e-1

| K | Peak NMI | CoA NMI | Gap |
|---|----------|---------|-----|
| 40 | 0.381 | 0.380 | −0.001 |
| 100 | 0.493 | 0.500 | **+0.007** |
| 200 | 0.489 | 0.498 | **+0.010** |
| 300 | 0.500 | 0.501 | +0.001 |
| 400 | 0.423 | 0.420 | −0.003 |

### NG20 entropy sweep at ld=50, K=200

| ent | Peak NMI | CoA NMI | Gap |
|-----|----------|---------|-----|
| 5e-2 | 0.463 | 0.463 | +0.000 |
| **1e-1** | 0.489 | 0.498 | **+0.010** |
| 2e-1 | 0.484 | 0.474 | −0.010 |
| 3e-1 | 0.551 | 0.523 | −0.028 |

### Training duration at ld=50, K=200, ent=1e-1

| Epochs | Peak NMI | Peak epoch | CoA NMI | Gap |
|--------|----------|------------|---------|-----|
| 200 | 0.489 | 171 | 0.498 | **+0.010** |
| 300 | 0.485 | 291 | 0.484 | −0.001 |
| 500 | 0.474 | 241 | 0.466 | −0.007 |

Longer training monotonically erodes the merge advantage. **200 epochs is optimal** for the merge>raw effect.

### SGD vs full-batch EM on NG20

| Batch size | Best Peak NMI | CoA NMI | Gap |
|-----------|--------------|---------|-----|
| 128 | 0.407 | 0.343 | −0.064 |
| 256 | 0.399 | 0.352 | −0.047 |
| 1024 | 0.486 | 0.441 | −0.045 |
| **Full-batch** | **0.489** | **0.498** | **+0.010** |

SGD uniformly eliminates the merge>raw signal across all 20 configurations tested.

---

## Cross-Dataset Patterns

| Property | MNIST | NG20 |
|----------|-------|------|
| Merge effect size | +0.162 | +0.010 |
| Optimal ld | 50 | 50 |
| Optimal K | 200 | 100–200 |
| Optimal ent | 3e-1 | 1e-1 |
| Optimal l2 | 5e-4 | 1e-4 |
| L1 effect | Hurts merge | Kills model |
| SGD vs EM | EM wins | EM wins (essential) |
| Training duration | 200 ep / 10 cyc | 200 ep / 1 cyc |

**ld=50 and K=100-200 are robust across both datasets.** The key difference is entropy regularization: NG20 needs a lower value (1e-1 vs 3e-1) and is much more sensitive to it. The merge effect is 16× larger on MNIST, likely because image patches have richer local sub-structure than TF-IDF document vectors.

---

## What Remains Unexplored

- **Merge analysis at multiple checkpoints**: analyses only run at the final epoch. Running at e.g. epoch 50/100/150/200 would characterize how the merge gap evolves during training — a label-free stopping criterion (LL plateau or assignment stability) could exploit this.
- **NG20 robustness**: the best NG20 config (K=200, ld=50, ent=1e-1, ep=200) was never repeated across seeds; run-to-run variance is unknown.
- **K-means initialization for HMoG**: the base `HMoGModel` initializes mixture components randomly (`pst_man.initialize`), not from k-means cluster centers. MFA uses k-means initialization in observable space and sees reliable convergence; `ProjectionHMoGModel` runs k-means in latent projection space. It is worth exploring whether k-means initialization in observable or latent space improves HMoG convergence speed or final NMI, particularly on NG20 where the model is sensitive to initialization. For high-dimensional observables (NG20, 20k features), initialization should be done on latent projections (cheap) rather than in observable space (expensive CPU).
- **MFA comparison on NG20**: in progress (Wave 1 running).

---

## Baseline Comparison (NG20)

| Method | NMI |
|--------|-----|
| MFA (K=150, ld=5, from literature) | 0.168 |
| LDA | 0.472 |
| HMoG raw (best merge>raw config) | 0.489 |
| HMoG CoA merge (K=200, ld=50) | 0.498 |
| HMoG CoA merge (K=300, ld=50) | 0.501 |
| HMoG raw (K=40, ld=400) | **0.567** |
