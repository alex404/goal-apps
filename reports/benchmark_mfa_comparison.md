# MFA Benchmark: goal-apps vs gans-n-gmms Reference

**Date:** 2026-02-23
**Dataset:** MNIST (60 000 train / 10 000 test, 784-dim)
**Config:** K=150 components, L=10 latent dims, batch_size=512, lr=2e-5, Adam

---

## Models compared

| ID | Model | Implementation | Gradient |
|----|-------|---------------|----------|
| **full-mfa** | Full MFA (`diagonal=False`) | goal-apps `CompleteMixtureOfSymmetric` + `join_conjugated` prior | EF natural gradient (prior_stats − posterior_stats) |
| **diag-mfa** | Diagonal MFA (`diagonal=True`) | goal-apps `CompleteMixtureOfConjugated` + `bound_means` | EF natural gradient (prior_stats − posterior_stats) |
| **gans-ref** | Full MFA (reference) | `scratch/mfa_gans_mnist.py` (faithful replication of gans-n-gmms) | Autodiff on neg-LL |

All three runs used the **same initialisation strategy**: k-means cluster centres as
observable means, isotropic covariance at mean data variance, loading matrix =
`0.01 * randn`, uniform mixture weights. The goal-apps `initialize_model` and the
script's `random_init` produce numerically equivalent starting points.

**Training budget (gradient steps):**
- goal-apps: 51 epochs × ⌊60 000/512⌋ = 117 batches × 1 step/batch ≈ **5 967 steps**
- gans-ref: **6 000 steps** (default `N_ITERS`)

---

## Results

### Final-epoch metrics

| Model | Test NMI | Train NMI | Test Acc | Test LL/sample |
|-------|----------|-----------|----------|----------------|
| **full-mfa** | 0.546 | 0.533 | 0.672 | 156.0 |
| **diag-mfa** | 0.539 | 0.526 | 0.645 | 151.9 |
| **gans-ref** | **0.596** | 0.590 | — | **569.1** |

### NMI over training (goal-apps models, logged every 10 epochs)

| Epoch | full-mfa Test NMI | diag-mfa Test NMI |
|-------|-------------------|-------------------|
| 1     | 0.539             | 0.535             |
| 11    | 0.540             | 0.535             |
| 21    | 0.543             | 0.536             |
| 31    | 0.544             | 0.537             |
| 41    | 0.545             | 0.538             |
| 51    | 0.546             | 0.539             |

gans-ref NMI was measured only at the end of training (no per-epoch logging).

---

## Discussion

### full-mfa vs diag-mfa

Full MFA consistently outperforms the diagonal variant (+0.7 pp NMI, +2.7 pp
accuracy at epoch 51) despite having only marginally more parameters (1 421 099 vs
1 414 349). Both models improve monotonically with no instability. The full model's
advantage grows slightly over training, suggesting the full loading matrices have more
capacity to exploit additional gradient steps.

The diagonal variant is ~3× faster per epoch (~93 s vs ~259 s for 51 epochs) since it
avoids full covariance operations and the `join_conjugated` projection.

### full-mfa vs gans-ref

The gans-n-gmms reference achieves **+5 pp NMI** (0.596 vs 0.546) over the goal-apps
full MFA and a dramatically higher test log-likelihood (569 vs 156 nats/sample) despite
an identical starting LL (~134 at step 0 / epoch 1).

Both models use the same architecture (per-component factor analysis with diagonal noise
and full loading matrices), the same initialisation, the same batch size, learning rate,
and step budget. The only structural differences are:

1. **Gradient computation.** gans-ref differentiates through the Woodbury
   log-likelihood directly. goal-apps uses the EF natural gradient
   `prior_stats − posterior_stats`, then applies a `join_conjugated` hard projection
   after each step to enforce the per-component N(0,I) prior exactly.

2. **Prior constraint.** gans-ref places no explicit constraint on the latent prior;
   the marginal p(z|component k) is free to drift. goal-apps projects `lat_nat` after
   every step so that θ_Y + ρ = N(0,I) in natural parameters.

3. **Optimizer state.** gans-ref accumulates Adam momentum across all 6 000 steps
   without reset. goal-apps uses `epoch_reset=false`, which also persists state, but
   the `join_conjugated` projection after each step modifies `params` outside the
   optimizer, which can partially corrupt the Adam second-moment estimates.

The LL gap (~413 nats/sample) is large enough that model-capacity differences cannot
explain it; FactorAnalysis with a fixed N(0,I) prior is in principle equally expressive
since the loading matrices and observation noise are unconstrained. The most likely
sources are:

- **Adam second-moment corruption**: the `join_conjugated` projection modifies the
  parameter vector between the optimizer update and the next gradient evaluation.
  Adam's per-parameter adaptive learning rates are calibrated to the unconstrained
  parameter trajectory; projecting after each step introduces a bias that may
  effectively reduce the usable learning rate for the loading matrices and observation
  parameters.
- **EF gradient mismatch**: `prior_stats − posterior_stats` is the natural gradient
  of the ELBO under a fixed prior. If the prior is being simultaneously adjusted by
  `join_conjugated`, the gradient target shifts between the outer compute of
  `posterior_stats` and the inner optimizer step, potentially slowing convergence.

### Takeaways

- The goal-apps EF MFA implementation is stable and correctly enforces the N(0,I)
  prior via `join_conjugated`.
- The **clustering quality gap relative to gans-ref is real and non-trivial** (~5 pp
  NMI). It is not explained by model capacity and likely reflects an optimisation
  issue introduced by the post-step projection interacting with Adam's adaptive
  estimates.
- Possible next steps: (a) compare with the gans-ref autodiff gradient but keeping the
  goal-apps EF parameterisation (i.e., differentiate the EF log-likelihood rather than
  using the natural gradient); (b) test whether removing `epoch_reset` and using a
  lower effective LR improves the EF version; (c) run the gans-ref with the default
  `km_init` (per-cluster FactorAnalysis init) to check how much of the gap is due to
  initialisation quality.
