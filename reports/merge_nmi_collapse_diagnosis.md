# Merge NMI Collapse — Diagnosis Report

**Date**: 2026-02-26
**Symptom**: CoAssignment merge NMI at or below base overclustered NMI, despite base NMI being good.

---

## The Core Problem

The co-assignment merge algorithm should *improve* NMI by merging 200 overclustered
components into 10. Historically it gave reliable boosts of +0.14 to +0.25. Recently
it gives boosts of ~0 or even negative.

| Run | epoch | Base NMI | Merge NMI | Boost |
|-----|-------|----------|-----------|-------|
| 20260208-162045 (MFA) | 200 | 0.644 | 0.804 | **+0.160** |
| 20260222-153217 (HMoG) | 200 | 0.535 | 0.747 | **+0.213** |
| 20260222-190710 (HMoG) | 500 | 0.643 | 0.785 | **+0.142** |
| 20260226-004412 (HMoG) | 200 | 0.606 | 0.710 | +0.104 |
| 20260226-011110 (HMoG) | 1000 | 0.706 | 0.710 | **+0.004** |

The current run (`011110`) has *better* base NMI than almost all historical runs, yet
the merge produces essentially no improvement.

---

## Finding 1: The "Best Run" Config is Not the Reliably Good Config

Two distinct configs were used on Feb 22 (same code: ac11513 + e19493c goal-jax):

### Config A — Reliable (runs 150939, 153217, 160242)
```
lr: 1e-4   l2_reg: 1e-4   mixture_entropy_reg: 1e-4
n_epochs: 100–200   num_cycles: 3
```
Results: merge NMI **0.724–0.772**, consistently, across all seeds tested.

### Config B — Lucky (run 190710, current yaml)
```
lr: 3e-4   l2_reg: 1e-3   mixture_entropy_reg: 1e-3
n_epochs: 500   num_cycles: 10
```
Results: one lucky run at **0.785**, but typical results are **0.68–0.71** (boost ≈ 0).

Config B has 3× higher lr, 10× higher L2, 10× higher entropy reg, and 2.5–5× more
training per cycle. The effort to match Config B has been chasing a one-off lucky seed.

---

## Finding 2: Over-Training Destroys Merge Structure

Across all HMoG runs on the same code and architecture, merge boost degrades with
more total training:

| Total epochs | Merge boost (typical) |
|---|---|
| 200 | +0.10 to +0.21 |
| 500–600 | +0.06 to +0.19 |
| 1000+ | +0.00 to +0.10 |

The `011110` run is the clearest example: at epoch 500 (1 cycle) the base NMI was
0.691 with merge NMI 0.686 (−0.005 boost). By epoch 1000 (2 cycles) base NMI reached
0.706 with merge NMI 0.710 (+0.004 boost). More training kept improving individual
cluster purity but not hierarchical structure.

**Hypothesis**: Longer training with high L2/entropy regularization causes each GMM
component to become highly specialized to a small local region of data, increasing
individual cluster purity. But these components lose their spatial organization in
the 50-dim latent space — components for the same digit class no longer cluster
together — so co-assignment patterns can't reveal the 10-class hierarchy.

---

## Finding 3: All Historically Reliable Results Were MFA, Not HMoG

Scanning all runs with merge boost > +0.12:

- **Jan 12**: MFA or early HMoG variants (boost +0.14 to +0.22)
- **Feb 1–8**: All MFA (boost +0.09 to +0.25)
- **Feb 23**: MFA, best ever — coa=**0.8385**, clus=0.575, boost=**+0.263**
- **Feb 22**: HMoG with Config A (boost +0.14 to +0.22)

The user's intuition that "merging overclustered models is a reliable source of
performance" was built on MFA runs, not HMoG. HMoG merge boost has always been
more variable.

---

## Finding 4: Code and Library Are Correct

Verified on the current checkout (goal-apps: ac11513, goal-jax: e19493c):

- `python -c "import goal; print(goal.__file__)"` → `/home/alex404/code/goal-jax/src/goal/__init__.py` ✓
- `Normal.whiten` (called by ac11513 `bound_means`) exists at e19493c ✓
- `CompleteMixtureOfSymmetric`, `from_mixture_params`, `to_mixture_params` all exist ✓
- No uncommitted changes in goal-jax (`git status` clean) ✓
- Co-assignment analysis code unchanged between ac11513 and main ✓

The regression is **not** a code or library mismatch.

---

## Finding 5: The Merge Algorithm Needs Hierarchical Latent Structure

The co-assignment merge works when:
1. Data from the same digit class is consistently assigned to the *same subset* of the 200 components
2. Those subsets are spatially organized (components for digit "1" are near each other in the 50-dim latent space)

High base NMI is *necessary but not sufficient*. A run with base NMI 0.69 and good
latent organization (190710) merges to 0.785. A run with base NMI 0.69 and poor
latent organization (011110) merges to only 0.711.

The metrics that distinguish good vs. bad merge runs:

| Metric at epoch 500 | Good merge run (190710) | Bad merge run (011110) |
|---|---|---|
| Mixture entropy | **3.86** (non-uniform) | 4.23 (more uniform) |
| Merge boost | **+0.142** | −0.005 |

The good run developed strongly non-uniform cluster weights faster, concentrating mass
in a few per-class dominant components. This concentration is what the merge algorithm
exploits.

---

## Recommended Actions

1. **Use Config A for HMoG**: `lr=1e-4`, `l2_reg=1e-4`, `mixture_entropy_reg=1e-4`,
   `n_epochs=200`, `num_cycles=3–5`. Reliably gives merge NMI 0.72–0.77.

2. **Use MFA for publication**: The Feb 23 MFA runs (ac11513 code, e19493c goal-jax)
   consistently gave merge NMI 0.78–0.84. Config: `lr=2e-5`, `n_epochs=100`,
   `n_clusters=150`, `latent_dim=10`. These used the *old* MFA trainer
   (`bound_means` resets all latent components to standard normal) which may be
   key to its reliability.

3. **Do not chase the 0.785 HMoG result**: That was a single lucky seed with
   Config B. The same config reliably gives 0.68–0.71, with occasional lucky
   runs up to 0.78. Not reproducible on demand.

4. **Short training beats long training for merge**: If a run hits a checkpoint at
   200–300 epochs and has merge NMI ≥ 0.72, that may be as good as it gets.
   Continuing to train typically hurts the merge structure.
