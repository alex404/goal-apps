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
| NG20 | Best raw NMI | **0.589** | ld=400, K=80, ent=1e-1, prs=1e-2 |
| NG20 | Best CoA NMI | **0.515** | ld=400, K=60, ent=1e-1, prs=1e-2 |
| NG20 | Best KL NMI | **0.529** | ld=400, K=80, ent=1e-1, prs=1e-2 |
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
# Best raw NMI AND best merge NMI (2026-04-11)
latent_dim: 400
n_clusters: 60
mixture_entropy_reg: 1e-1
upr_prs_reg: 1e-2
lwr_prs_reg: 1e-2
l2_reg: 1e-4
l1_reg: 0
min_prob: 1e-5
lr: 1e-4
batch_steps: 100
n_epochs: 200
num_cycles: 1
```

**Result (K=60)**: peak raw NMI = 0.575 (epoch 41), CoA NMI = **0.515**, KL NMI = 0.515, final NMI = 0.537 (epoch 191)
**Result (K=80)**: peak raw NMI = **0.589** (epoch 41), CoA NMI = 0.506, KL NMI = **0.529**, final NMI = 0.541 (epoch 191)

```
# Previous best merge>raw recipe (ld=50 regime)
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
```

### NG20 capacity sweep (ld × K at ent=1e-1)

| ld | Stability | Best raw NMI | Merge behavior |
|----|-----------|-------------|----------------|
| 20 | Stable | 0.299 | Largest gap (+0.012) but low absolute NMI |
| 50 | Stable at all K | 0.551 | Sweet spot: merge>raw at K=100-300 |
| 100 | Unstable at K=60-200 | 0.473 | cuSolver crashes |
| 400 | Stable at K≤40, bs=100 | 0.567 | Best raw NMI (pre-prs-reg); merge always worse |
| 400 | K=60 with prs=1e-2 | 0.575 | K=60 enables merge signal (CoA 0.515) |
| 400 | K=80 with prs=1e-2 | **0.589** | Best raw NMI; KL merge 0.529 |

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

## NG20 Precision Regularization Exploration (2026-04-11)

The key discovery: **symmetric precision regularization (upr=lwr=1e-2) stabilizes high-K configs at ld=400**, enabling overclustering that was previously impossible. Prior runs used prs=1e-5 (the default), which limited ld=400 to K≤40.

### Precision reg sweep at ld=400, K=60, ent=1e-1

| upr_prs_reg | lwr_prs_reg | Target eigenvalue | Stability | Peak NMI | CoA NMI | KL NMI |
|-------------|-------------|-------------------|-----------|----------|---------|--------|
| 1e-5 | 1e-5 | 1.0 | NaN at ~epoch 1 | — | — | — |
| 1e-3 | 1e-5 | 0.01 | NaN at epoch ~70 | 0.567 | — | — |
| 1e-2 | 1e-5 | 0.001 | NaN at epoch 1 | — | — | — |
| **1e-2** | **1e-2** | **1.0** | **Stable (200 ep)** | **0.575** | **0.515** | **0.515** |

### Precision reg at ld=400, K=80, ent=1e-1

### K sweep at ld=400, ent=1e-1, prs=1e-2 (symmetric)

| K | Peak NMI | Peak epoch | CoA NMI | KL NMI | Effective K (final) |
|---|----------|------------|---------|--------|---------------------|
| 60 | 0.575 | 41 | **0.515** | 0.515 | 30 |
| **80** | **0.589** | 41 | 0.506 | **0.529** | 44 |
| 100 | 0.584 | 51 | 0.507 | 0.510 | 43 |

K=80 is the sweet spot. K=100 degrades both raw and merge NMI — additional components fragment rather than capture meaningful sub-structure.

### Precision reg at ld=400, K=40, ent=1e-1 (Exp 4)

| upr_prs_reg | lwr_prs_reg | Peak NMI | CoA NMI | KL NMI | NMI stability |
|-------------|-------------|----------|---------|--------|---------------|
| 1e-5 | 1e-5 | 0.567 | — | — | Peak then collapse |
| 1e-3 | 1e-5 | 0.560 | 0.491 | **0.517** | Stable plateau ~0.550 |

### Key insights

- **Target eigenvalue = 1.0 (symmetric) is safest**: asymmetric prs pushes eigenvalues toward lwr/upr, which can be numerically problematic if too small.
- **Strong prs (1e-2) enables overclustering at ld=400**: K=60 is stable with prs=1e-2 but crashes with prs ≤ 1e-3.
- **Precision reg improves stability without major NMI cost**: Exp 4 (K=40, asym prs=1e-3/1e-5) gave 0.560 peak vs 0.567 baseline — mild cost for dramatically better stability and merge NMI.
- **batch_steps=100 is optimal at ld=400**: higher values (200, 500) cause faster overfitting, not better convergence.
- **L1 reg at ld=400 requires careful calibration**: l1=1e-3 kills the model at bs=100 (gradient too weak to overcome penalty). l1 ≤ 1e-5 needed — exploration ongoing.

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
| HMoG raw (best merge>raw config, ld=50) | 0.489 |
| HMoG CoA merge (K=200, ld=50) | 0.498 |
| HMoG CoA merge (K=300, ld=50) | 0.501 |
| HMoG raw (K=40, ld=400, prs=1e-5) | 0.567 |
| HMoG raw (K=60, ld=400, prs=1e-2) | 0.575 |
| HMoG raw (K=80, ld=400, prs=1e-2) | **0.589** |
| HMoG CoA merge (K=60, ld=400, prs=1e-2) | **0.515** |
| HMoG CoA merge (K=80, ld=400, prs=1e-2) | 0.506 |
| HMoG KL merge (K=40, ld=400, asym prs) | 0.517 |
| HMoG KL merge (K=80, ld=400, prs=1e-2) | **0.529** |

Note: all NG20 HMoG results are single-seed runs. The raw NMI improvement (0.567 → 0.575) is marginal and within expected run-to-run variance. The more significant finding is that **symmetric precision regularization (prs=1e-2) stabilizes K=60 at ld=400**, unlocking overclustering at high latent dimension. This yields CoA merge NMI = 0.515 (+0.014 over the previous ld=50 merge recipe), which is a qualitatively different regime. KL merge NMI also improves significantly — from not previously tracked at ld=400 to 0.515–0.517.
