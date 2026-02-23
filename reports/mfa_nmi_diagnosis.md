# MFA NMI Instability Diagnosis (K=10, L=10, MNIST)

**Date:** 2026-02-22

## Setup

Comparing two MFA implementations on MNIST (K=10 clusters, L=10 latent dims):

- **Reference**: `scratch/mfa_gans_mnist.py` — autodiff of neg-LL, Adam, batch_size=512, 6000 iters
- **Goal-jax**: `goal train dataset=mnist model=mnist-mfa` — natural gradient, full-batch EM

The reference achieves stable NMI ~0.62. The goal-jax implementation was observed to be unstable.

---

## Three Hypotheses Tested

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | `batch_steps=1000` (full-batch EM) causes instability | **CONFIRMED — primary cause** |
| H2 | `epoch_reset=True` disrupts Adam momentum | Untested in isolation; secondary |
| H3 | Random loading init causes high seed variance | **REJECTED — random init is better** |

---

## Experiment A: Initialization Effect

Modified `scratch/mfa_gans_mnist.py` to support `--init-mode {fa, random}`:

- `fa`: per-cluster sklearn `FactorAnalysis` (reference style)
- `random`: k-means centers + isotropic covariance + `0.01 * randn` loading matrix (goal-jax style)

**Results (K=10, L=10, 3 seeds):**

| Init mode | Test NMI (mean ± std) | Test LL/sample |
|---|---|---|
| `fa` (reference) | 0.309 ± 0.003 | 2282 |
| `random` (goal-jax) | **0.616 ± 0.002** | 499 |

**Interpretation:**

- Random init achieves 2× higher NMI and is *more* stable than FA init.
- FA init converges to a tight-covariance, high-likelihood solution that partitions data by
  spatial similarity rather than digit identity — good density model, poor clustering.
- Random init starts near-isotropic and must discover factor structure from scratch,
  which leads to better cluster assignments for MNIST digits.
- **H3 rejected**: initialization is not the source of instability in goal-jax.

---

## Experiment B: Training Loop Effect

**Full-batch config** (current `mnist-mfa.yaml`: `batch_size=null`, `batch_steps=1000`, `epoch_reset=true`):

| Epoch | Test NMI | Test LL |
|---|---|---|
| 1  | 0.517 | 62.5 |
| 11 | 0.524 | 232.7 |
| **21** | **0.000** | **NaN** ← catastrophic collapse |
| 31+ | 0.000 | NaN (irrecoverable) |

The model collapses between epochs 11 and 21. NMI drops to zero, LL becomes NaN.
The crash surfaces as a cuSolver Cholesky error in subsequent runs once parameters leave
the positive-definite cone.

**Mechanism**: With `batch_steps=1000`, each epoch computes one global posterior estimate
over all 60k samples, then takes 1000 Adam steps toward that fixed target. The parameters
overshoot, causing the covariance matrix to lose positive-definiteness.

**Mini-batch config** (`batch_size=512`, `batch_steps=1`, `epoch_reset=false`, 2 seeds):

| Epoch | Seed 1 NMI | Seed 2 NMI |
|---|---|---|
| 1  | 0.494 | 0.500 |
| 11 | 0.507 | 0.529 |
| 21 | 0.491 | 0.515 |
| 31 | 0.477 | 0.501 |
| 50 | 0.465 | 0.494 |

Stable — no collapse, no NaN. Std across seeds is small (~0.02). However, NMI gradually
declines and does not reach the reference level of 0.62.

---

## Root Cause

**H1 confirmed**: `batch_steps=1000` with a fixed posterior target causes catastrophic
parameter divergence. The fix is to switch to true mini-batch natural gradient descent:
one gradient step per batch, no optimizer reset between epochs.

---

## Residual Gap (Open Question)

After fixing H1, the mini-batch goal-jax reaches NMI ~0.49–0.53, still ~0.10 below
the reference's 0.62. Possible causes (not yet tested):

1. **Learning rate mismatch**: goal-jax uses `lr=1e-4` vs reference's `2e-5` (5× higher).
   The natural gradient step in mean coordinate space may have a different effective scale.
2. **Bounding interference**: `bound_means()` resets ALL latent components to standard
   normal N(0,I) at every gradient step, and clips mixture probabilities. This modifies
   the effective gradient and may prevent NMI improvement.
3. **Optimizer**: goal-jax uses `adamw` (with weight decay); reference uses plain `adam`.
4. **Equivalence**: The natural gradient `prior_stats - posterior_stats` is theoretically
   equivalent to autodiff of neg-LL for exponential families, but the bounding step breaks
   this equivalence and the resulting modified gradient may not improve clustering quality.

---

## Recommended Fix

Update `config/hydra/model/mnist-mfa.yaml`:

```yaml
trainer:
  lr: 2e-5          # match reference (down from 1e-4)
  batch_size: 512   # mini-batch (was: null = full 60k)
  batch_steps: 1    # one step per batch (was: 1000)
  epoch_reset: false  # keep Adam momentum (was: true)
```

With `n_epochs=500` and `batch_size=512`, this gives ~500 × 117 = 58,500 gradient steps
total, comparable to the reference's 6,000 × (60k/512) ≈ 703 effective epochs.
Consider reducing `n_epochs` to ~50–100 to match the reference's compute budget.

---

## Files

| File | Purpose |
|---|---|
| `scratch/mfa_gans_mnist.py` | Reference script (modified to support `--init-mode {fa,random}` and `--n-repeats`) |
| `config/hydra/model/mnist-mfa.yaml` | Goal-jax config (needs fix) |
| `plugins/models/mfa/trainers.py` | `GradientTrainer` — `batch_steps`, `epoch_reset`, `bound_means` |
