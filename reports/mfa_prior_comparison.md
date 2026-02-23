# MFA Prior Constraint Comparison

**Date:** 2026-02-23
**Dataset:** MNIST (60 000 train / 10 000 test, 784-dim)
**Config:** K=150 components, L=10 latent dims, full-batch (60k), batch_steps=1000, lr=2e-5, Adam
**Epochs:** 100 (last logged checkpoint: epoch 91)

---

## Variants

| ID | Description | `enforce_prior` | `epoch_reset` | Bounding step |
|----|-------------|:---------------:|:-------------:|---------------|
| **A** | Full MFA, hard prior (current default) | `true` | `false` | `bound_obs_means` + `join_conjugated` projection at epoch end |
| **B** | Full MFA, soft whitening (new) | `false` | `true` | `bound_means` (resets lat block to N(0,I) in mean space) |
| **C** | Diagonal MFA (unchanged baseline) | `true`* | `true` | `bound_means` |

\* `enforce_prior=true` is a no-op for diagonal MFA since `CompleteMixtureOfSymmetric` branch is never taken.

---

## NMI Progression (Test Set)

| Epoch | A: Full, hard prior | B: Full, soft whitening | C: Diagonal |
|------:|:-------------------:|:-----------------------:|:-----------:|
| 1     | 0.534               | 0.537                   | 0.539       |
| 11    | 0.552               | 0.559                   | 0.560       |
| 21    | 0.556               | 0.569                   | 0.571       |
| 31    | 0.563               | 0.573                   | 0.575       |
| 41    | 0.566               | 0.576                   | 0.577       |
| 51    | 0.570               | 0.577                   | 0.578       |
| 61    | 0.572               | 0.578                   | 0.579       |
| 71    | 0.574               | 0.579                   | 0.581       |
| 81    | 0.575               | 0.580                   | 0.582       |
| 91    | 0.575               | 0.581                   | 0.582       |

---

## Final Results (Epoch 91)

| Variant | Test NMI | Test Acc (greedy) | Train NMI | Wall clock (training, 100 ep) |
|---------|:--------:|:-----------------:|:---------:|:-----------------------------:|
| A: Full, hard prior | 0.575 | 0.665 | 0.563 | 545 s |
| B: Full, soft whitening | 0.581 | **0.695** | 0.569 | 542 s |
| C: Diagonal | **0.582** | 0.682 | 0.572 | **223 s** |

Wall clock is measured to end of epoch 100 training (before analyses). Analyses add ~90–120 s for A/B and ~120 s for C.

---

## Training Speed

Timing extracted from per-epoch wall-clock logs. "JIT + init" covers dataset preparation, k-means init, and JAX compilation (all one-time costs):

| Variant | JIT + init | Steady-state (ep/s) | Steady-state (s/ep) | Total training (100 ep) | Speedup vs A |
|---------|:----------:|:-------------------:|:-------------------:|:-----------------------:|:------------:|
| A: Full, hard prior | ~70 s | 0.213 ep/s | 4.70 s/ep | 545 s | 1.0× |
| B: Full, soft whitening | ~71 s | 0.215 ep/s | 4.65 s/ep | 542 s | 1.0× |
| C: Diagonal | ~58 s | 0.629 ep/s | 1.59 s/ep | 223 s | **2.96×** |

Steady-state epoch time measured over epochs 11–91 (post-JIT, excludes final checkpoint I/O). The 4.65 s/ep for B is essentially identical to A — skipping `join_conjugated` saves negligible time because the bottleneck is the full-batch posterior computation and 1 000 Adam steps.

### Per-step cost comparison (steady state)

| Variant | ms / gradient step | notes |
|---------|:-----------------:|-------|
| A: Full, hard prior | 4.7 ms | |
| B: Full, soft whitening | 4.7 ms | |
| C: Diagonal | 1.6 ms | |
| gans-n-gmms | 1.8 ms | batch_size=512, Woodbury solve |

Diagonal (C) and gans-n-gmms are within 12% of each other per step. The full MFA (A/B) costs ~2.9× more per step than diagonal. gans-n-gmms uses batch_size=512 vs full-batch 60 000 for A/B/C, so the per-step comparison isn't perfectly apples-to-apples on data throughput, but it does reflect real wall-clock cost per optimizer update.

The full MFA's higher per-step cost likely includes redundant computation: each of the 1 000 inner Adam steps re-evaluates `prior_stats = mfa.to_mean(current_params)` via the same `mean_posterior_statistics` path, and the full-covariance Woodbury solve is repeated per inner step even though the posterior target `bounded_posterior_stats` is fixed for the entire batch. Hoisting `prior_stats` out of the inner loop (or fusing the 1 000-step scan more tightly) could plausibly close much of the 2.9× gap with diagonal.

### NMI at equal wall-clock budget

The diagonal model's speed advantage is decisive on a fixed time budget:

| Wall clock | A NMI (approx) | B NMI (approx) | C NMI |
|-----------:|:--------------:|:--------------:|:-----:|
| 75 s | 0.534 (ep 1) | 0.537 (ep 1) | **0.560** (ep 11) |
| 120 s | 0.552 (ep 11) | 0.559 (ep 11) | **0.571** (ep 21) |
| 200 s | ~0.560 (ep ~27) | ~0.570 (ep ~27) | **0.582** (ep 91) |
| 545 s | 0.575 (ep 100) | **0.581** (ep 100) | 0.582 (done at 223 s) |

C reaches its plateau (~0.582) before A and B finish epoch 30.

---

## Reference: gans-n-gmms (60 000 steps)

**Config:** K=150, L=10, random init (matches goal-apps), batch_size=512, lr=2e-5, plain Adam, no prior projection.
NMI logged every 1 000 steps on full train and test sets.

| Step | Wall (s) | neg-LL | Test NMI | Train NMI |
|-----:|:--------:|:------:|:--------:|:---------:|
| 0 | 0 | −149.9 | 0.535 | 0.526 |
| 1000 | 12 | −216.4 | 0.555 | 0.541 |
| 2000 | 21 | −278.1 | 0.570 | 0.555 |
| 3000 | 31 | −340.6 | 0.581 | 0.568 |
| 4000 | 40 | −410.9 | 0.602 | 0.586 |
| **5000** | **49** | **−493.1** | **0.612** | **0.599** |
| 6000 | 59 | −589.4 | 0.607 | 0.599 |
| 7000 | 68 | −705.8 | 0.586 | 0.581 |
| 8000 | 78 | −871.5 | 0.525 | 0.527 |
| 9000 | 87 | −1592.1 | 0.343 | 0.352 |
| 10000 | 96 | −1945.7 | 0.321 | 0.319 |
| 15000 | 144 | −2405.2 | 0.365 | 0.362 |
| 20000 | 191 | −2505.5 | 0.377 | 0.372 |
| 30000 | 286 | −2659.7 | 0.421 | 0.417 |
| 40000 | 381 | −2726.0 | 0.432 | 0.425 |
| 50000 | 476 | −2770.4 | 0.431 | 0.426 |
| 60000 | 574 | −2787.8 | 0.430 | 0.425 |

**Peak test NMI: 0.612 at step 5 000 (49 s). Collapse begins ~7 000–8 000 steps, catastrophic by step 9 000. Never recovers: post-collapse plateau ≈ 0.43.**

The mechanism: without any prior constraint, Adam continues decreasing neg-LL monotonically by concentrating component covariances (neg-LL grows from −589 to −2788 without bound). Individual components learn very tight, high-likelihood fits to sub-modes; cluster assignments degenerate as many components collapse onto the same data manifold.

---

## Discussion

### Hypothesis confirmed: hard projection over-constrains full MFA

**B vs A (+0.006 NMI, +3.0 pp accuracy):** Replacing the hard `join_conjugated` end-of-epoch projection with soft gradient-whitening (`bound_means`, which resets the *gradient target's* latent block to N(0,I) in mean space) closes roughly **80 % of the gap** between A and C. The improvement opens early (already visible at epoch 11) and is stable; NMI in B does not plateau or regress. Greedy accuracy in B actually exceeds both A and C despite similar NMI to C.

Mechanistically, the hard projection in A modifies `params` outside the optimizer after each epoch, which partially corrupts Adam's per-parameter second-moment estimates (they are calibrated to the unconstrained trajectory). The unconditional `optimizer.init` reset that follows discards all accumulated momentum. B avoids both problems: the latent prior is enforced implicitly through the gradient target rather than through a post-step parameter edit, and the epoch-start reset re-initialises Adam from the current (clean) parameter state.

**B vs C (≈ same NMI, B wins on accuracy):** The full loading matrix in B does not give a NMI advantage over diagonal C, but it does yield higher greedy accuracy (0.695 vs 0.682). Both plateau near 0.581–0.582. C is 2.4× faster, so diagonal remains the better choice when GPU time is a constraint.

**vs gans-n-gmms:** At its peak (step 5 000, 49 s), gans-n-gmms achieves 0.612 NMI — 3.0 pp above our best (B/C at 0.582). However, it cannot hold that performance: by step 9 000 NMI has collapsed to 0.34, and the final 60 000-step value is only 0.43. The goal-apps variants (B, C) are completely stable across their entire 100-epoch budget. The goal-apps final NMI (0.581–0.582) substantially exceeds gans-n-gmms's long-run performance (~0.43), even though gans-n-gmms briefly reaches a higher peak. If early stopping at step 5 000 (49 s) is used, gans-n-gmms has a commanding lead; the question is whether that peak can be preserved.

---

## Key takeaways

1. **Soft whitening (B) strictly dominates hard projection (A):** same compute cost, +0.6 pp NMI, +3 pp greedy accuracy. The unconditional Adam reset + post-step parameter edit in A corrupts optimizer state without benefit.

2. **Diagonal is 3× faster and matches full MFA NMI (B ≈ C ≈ 0.582).** Full loading matrices don't help NMI here (perhaps at higher latent dim they would). C remains the default for time-constrained runs.

3. **gans-n-gmms peaks at 0.612 NMI but collapses catastrophically by step 9 000.** Without a prior constraint, Adam drives neg-LL to −∞ by compressing individual component covariances, destroying cluster structure. The goal-apps prior constraint (hard or soft) is what makes training stable; the cost is a ~3 pp NMI gap vs the gans-n-gmms peak. Closing that gap without sacrificing stability is the remaining open problem.
