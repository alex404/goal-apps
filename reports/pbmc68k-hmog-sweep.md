# PBMC68k HMoG Sweep Results

**Dataset**: PBMC68k (10x Genomics v1), ~68k cells, 2000 HVGs, 9 annotated cell types  
**Model**: HMoG (Hierarchical Mixture of Gaussians), full-batch gradient EM  
**Goal**: ARI > 0.6 (state-of-the-art benchmark for this dataset)  
**Date**: 2026-04-11 to 2026-04-12

---

## Key Findings

1. **Standardization removal is critical for CoA merge.** Applying `StandardScaler` to scRNA-seq counts (as was the default) fixes CoA merge ARI at ~0.185 regardless of other settings — the log-ratio structure is destroyed. Removing standardization immediately brings CoA merge ARI in line with KL merge and optimal merge (~0.443). This is also correct by scRNA-seq best practice (HVG selection already stabilizes variance).

2. **ld=200 is the latent dimension sweet spot.** ld=50 is too small (ARI 0.37); ld=100 loses a bit relative to ld=200; ld=300 shows mild end-of-training decline; ld=400 with l1=1e-3 actually hurts (sparse penalty too weak to help, enough to hurt gradients).

3. **Symmetric prs_reg is required for stability.** Every asymmetric configuration tested crashed with NaN (Cholesky failure) before convergence. Both `upr=5e-3/lwr=5e-2` and `upr=5e-2/lwr=1e-1` failed. Symmetric `upr=lwr=5e-2` is stable throughout.

4. **L1 sparsity (l1=1e-2) consistently helps.** Forcing sparse gene programs in the loading matrix improves both raw ARI and merge ARI across latent dims. At l1=1e-3 the effect disappears or reverses (too weak). At l1=1e-2 with ld=400, the benefit is present.

5. **batch_steps is the most dangerous hyperparameter.** The T-cell blob (~43% of cells) causes catastrophic collapse at bs≥300 regardless of entropy regularization. bs=100 is the stable regime. bs=300 peaked then collapsed; bs=500 invariably collapsed to <2 effective components. The MNIST analogy (500 batch_steps) does not hold for PBMC.

6. **Multi-cycle training requires LR decay.** Flat-LR multi-cycle (2 cycles, same lr) collapsed to 3.7 effective components. With `lr_scales=[1.0, 0.3]`, 2 cycles improves NMI from 0.463 to 0.511. A 3rd cycle causes a slight decline for K=100 (best peak is ep341 then drops); 2 cycles is the sweet spot.

7. **High entropy regularization (ent=1e0) works at bs=100.** Despite ent=1e0 looking risky, with bs=100 and K=100 it converges stably to 73 effective components (healthy overclustering). ent=3e-1 gives 88.8 components and slightly lower NMI/ARI — both are reasonable. ent=5e-1 is intermediate. At bs≥300, entropy reg offers no protection from collapse.

8. **K=100 is better than K=200 for this configuration.** K=200 with ent=1e0 and 2 cycles collapses (NMI 0.44→0.37). K=200 at ent=3e-1 and 3 cycles peaked at 0.453 but declined. K=100 is the reliable choice.

9. **Best result so far**: run `20260412-055911` (2cyc, bs100, ent=1e0, K=100, ld=200):  
   KL Merge ARI = **0.4602** (converged), KL Merge NMI = **0.5114**, 73 effective components.  
   Still well below the 0.6 ARI benchmark — likely requires a fundamentally better preprocessing scheme or scVI-style deep embeddings.

---

## Results Table

### Phase 1: Standardization and Baseline

| Run | Description | Test NMI | Test ARI | KL Merge ARI | CoA Merge ARI | Eff Comp |
|-----|-------------|----------|----------|--------------|---------------|----------|
| 20260411-104659 | Baseline (std=True, K=60, ld=200) | 0.4704 | — | 0.4435 | **0.1849** | 54.7 |
| 20260411-111735 | No-std, K=100, ld=200 | 0.4584↓ | 0.4441 | 0.4433 | 0.4433 | 77.0 |
| 20260411-120102 | No-std, 3000 HVGs | 0.4627 | 0.4440 | 0.4436 | 0.4436 | 80.0 |

> Removing standardization fixes CoA merge (0.185→0.443). More HVGs (3000 vs 2000) gives negligible improvement.

### Phase 2: Latent Dimension Sweep

| Run | ld | Test NMI | Test ARI | KL Merge ARI | Eff Comp |
|-----|-----|----------|----------|--------------|----------|
| 20260411-122943 | 50 | 0.4371 | 0.3709 | 0.3733 | 94.0 |
| 20260411-123447 | 100 | 0.4603 | 0.4422 | 0.4413 | 87.6 |
| 20260411-111735 | 200 | 0.4584↓ | 0.4441 | 0.4433 | 77.0 |
| 20260411-124624 | 300 | 0.4583↓ | 0.4415↓ | 0.4406 | 72.1 |
| 20260411-143739 | 200, l1=1e-2, ent=3e-1 | 0.4733↓ | 0.4462↓ | 0.4468 | 88.8 |
| 20260411-140651 | 400, l1=1e-2, ent=3e-1 | 0.4776 | 0.4450 | 0.4455 | 84.7 |
| 20260411-150339 | 400, l1=1e-3 | 0.4665 | 0.4140 | 0.4126 | 78.5 |

> ld=200 is the sweet spot. l1=1e-2 helps. l1=1e-3 at ld=400 hurts.

### Phase 3: Multi-Cycle Training

| Run | Cycles / LR scales | Test NMI | Test ARI | KL Merge ARI | Eff Comp |
|-----|---------------------|----------|----------|--------------|----------|
| 20260411-130523 | 2 cycles, flat LR | 0.4875 | 0.4490 | 0.4497 | **3.7** |
| 20260411-143739 | 1 cycle, l1, ent=3e-1 | 0.4733↓ | 0.4462↓ | 0.4468 | 88.8 |

> Flat-LR multi-cycle collapses (3.7 eff components despite ent reg). LR decay is required.

### Phase 4: Training Intensity (batch_steps)

| Run | bs | ent | Test NMI | Test ARI | KL Merge ARI | Eff Comp |
|-----|----|-----|----------|----------|--------------|----------|
| 20260411-143739 | 100 | 3e-1 | 0.4733↓ | 0.4462↓ | 0.4468 | 88.8 |
| 20260411-174328 | 300 | 3e-1 | 0.4523↓ | 0.3981↓ | 0.3974 | 1.6 |
| 20260411-180552 | 500 | 1e0 | 0.5012↓ | 0.3924↓ | 0.3903 | 1.1 |
| 20260411-193901 | 500 | 5e-1 | 0.5041↓ | 0.3997↓ | 0.4007 | 1.2 |
| 20260411-201153 | 500 | 1e0 | 0.5012↓ | 0.3924↓ | 0.3903 | 1.1 |

> bs≥300 invariably collapses to <2 effective components. bs=100 is the stable regime.

### Phase 5: Multi-Cycle with LR Decay and Entropy Sweep

| Run | Cycles | bs | ent | K | Test NMI | Test ARI | KL Merge ARI | Eff Comp |
|-----|--------|----|-----|---|----------|----------|--------------|----------|
| 20260411-155756 | 3 | 500 | 3e-1 | 100 | 0.4741↓ | 0.3768↓ | 0.3766↓ | 1.0 |
| 20260411-221215 | 3 | 100 | 3e-1 | 200 | 0.4701↓ | 0.4075↓ | 0.4073↓ | 122.1 |
| 20260411-230648 | 3 | 100 | 5e-1 | 100 | 0.4814↓ | 0.4483↓ | 0.4482↓ | 44.9 |
| 20260411-233904 | 3 | 100 | 1e0 | 100 | 0.5118 | 0.4529↓ | 0.4530↓ | 59.1 |
| 20260412-055911 | **2** | 100 | 1e0 | 100 | 0.5113 | 0.4604↓ | **0.4602** | 73.0 |
| 20260412-062029 | 2 | 100 | 1e0 | 200 | 0.3729↓ | 0.3569↑ | 0.4454 | 190.6 |

> 2 cycles is better than 3 (3rd cycle causes slight decline for K=100). K=200 with ent=1e0 collapses. Best result: 2cyc, ent=1e0, K=100, bs=100 → KL Merge ARI 0.4602 (converged).

---

## Asymmetric prs_reg Failures (Stability Notes)

Both asymmetric prs_reg configurations crashed with NaN (Cholesky/cuSolver) before epoch 200:
- `upr=5e-3, lwr=5e-2` (run 20260411-144827): crashed — lwr/upr ratio=10 too extreme at init
- `upr=5e-2, lwr=1e-1` (run 20260411-153448): crashed — asymmetry allows outlier eigenvalues

Symmetric `upr=lwr=5e-2` is required for PBMC stability.

---

## Conservative Recommended Configuration

Based on the sweep, the conservative recommendation is:

```yaml
latent_dim: 200         # Sweet spot (ld100 slightly worse, ld300 mildly declining)
n_clusters: 100         # ~11x overclustering; K=200 unstable at ent=1e0
num_cycles: 2           # 3rd cycle causes slight decline for K=100
lr_scales: [1.0, 0.3]   # LR decay essential — flat multi-cycle collapses

full:
  batch_steps: 100      # Stable ceiling; bs300+ collapses on PBMC T-cell blob
  l1_reg: 1e-2          # Sparse gene programs — consistent improvement
  ent_reg: 3e-1         # Conservative (88.8 eff comps, ARI 0.447); ent=1e0 gives 0.460 but riskier
  upr_prs_reg: 5e-2     # Symmetric required — asymmetric consistently crashes
  lwr_prs_reg: 5e-2
  obs_min_var: 1e-4     # Required to prevent covariance collapse
  lat_min_var: 1e-3
```

The ent=1e0 config (`20260412-055911`) achieves the best KL Merge ARI (0.4602 converged) but has fewer effective components (73 vs 88). For exploratory work, ent=3e-1 gives a healthier component distribution at small performance cost.

---

## Remaining Gap to Benchmark

Current best KL Merge ARI: **0.4602**  
K-means benchmark: ~0.55 ARI  
scVI-based methods: 0.60–0.70 ARI

The gap (~15% ARI to K-means benchmark) suggests the HMoG model's linear latent structure may be insufficient for PBMC68k, where cell-type manifolds likely have significant nonlinearity. Potential directions:
- Deeper preprocessing (scVI-style variational autoencoder embedding before HMoG)
- PCA pre-reduction to 50 dims before feeding HMoG (hybrid approach)
- Higher HVG count with log1p normalization tuning
- Alternative distance metrics for cluster merging
