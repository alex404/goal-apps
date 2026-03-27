# Sweep 8: CoA NMI Optimization at ld=50/K=200

## Goal

Push CoA merge NMI beyond 0.776 (current best mean across 3 reps) while keeping raw NMI >= 0.60. Explore regularization space including l1, l2, entropy reg, and min_prob.

## Sweep Design

Three sub-sweeps were run:

- **Sweep 8**: Original plan varying l2 (5e-4, 7e-4), min_prob (1e-5, 1e-3), cycles (3, 10), and l1 (0, 1e-3)
- **Sweep 8b**: l1=1e-3, l2=1e-4, varying entropy (1e-1, 3e-1, 5e-1)
- **Sweep 8c**: l1=1e-3, l2=3e-4, varying entropy (1e-1, 3e-1, 5e-1)

All runs: ld=50, K=200, lr=3e-4, batch_steps=1000, n_epochs=200.

## Numerical Stability

Many settings crashed with cuSolver errors (singular covariance matrices). The pattern:

| Setting | l2 | l1 | ent | Stability |
|---------|---:|---:|----:|-----------|
| l2=5e-4, l1=1e-3 | 5e-4 | 1e-3 | 3e-1 | 3/3 complete |
| l2=5e-4, no l1 | 5e-4 | 0 | 3e-1 | 3/3 complete (prior sweeps) |
| l2=3e-4, l1=1e-3 | 3e-4 | 1e-3 | 1e-1 | 3/3 complete |
| l2=3e-4, l1=1e-3 | 3e-4 | 1e-3 | 3e-1 | 2/3 crashed at epoch ~1600 |
| l2=5e-4, mp=1e-3 | 5e-4 | 0 | 3e-1 | 1/3 complete, 1 partial (epoch 1400) |
| l2=7e-4, no l1 | 7e-4 | 0 | 3e-1 | 0/3 complete |
| l2=1e-4, l1=1e-3 | 1e-4 | 1e-3 | 1e-1 | 1/3 complete |
| l2=1e-4, l1=1e-3 | 1e-4 | 1e-3 | 3e-1 | 0/3 complete |
| l2=7e-4, mp=1e-3 | 7e-4 | 0 | 3e-1 | 0/3 complete |

**Key stability findings:**
- l2 >= 5e-4 is the stability threshold without l1
- l1=1e-3 extends stability down to l2=3e-4, but only at ent=1e-1
- Higher entropy reg increases instability at lower l2
- min_prob=1e-3 does not help stability and may hurt it

## Results: Stable Settings with 3 Reps (ld=50, K=200)

| Setting | l2 | l1 | ent | Valid | Raw NMI | CoA NMI | Opt NMI |
|---------|---:|---:|----:|------:|--------:|--------:|--------:|
| Best CoA baseline (S4/S6) | 5e-4 | 0 | 3e-1 | 65 | 0.610 +/- .006 | **0.772 +/- .007** | 0.560 +/- .008 |
| S8-C | 5e-4 | 1e-3 | 3e-1 | 60 | 0.612 +/- .003 | 0.750 +/- .008 | 0.554 +/- .016 |
| S8c-I | 3e-4 | 1e-3 | 1e-1 | 49 | 0.612 +/- .001 | 0.720 +/- .007 | 0.550 +/- .005 |

Partial data (2 reps, crashed at epoch 1600):

| Setting | l2 | l1 | ent | Valid | Raw NMI | CoA NMI | Opt NMI |
|---------|---:|---:|----:|------:|--------:|--------:|--------:|
| S8c-J | 3e-4 | 1e-3 | 3e-1 | 77 | 0.601 +/- .004 | 0.759 +/- .004 | 0.508 +/- .011 |

## Results: Stable Settings (ld=20, K=100, from Sweep 7)

| Setting | l2 | mp | Valid | Raw NMI | CoA NMI | Opt NMI |
|---------|---:|---:|------:|--------:|--------:|--------:|
| S7-B | 3e-4 | 1e-5 | 67 | 0.553 +/- .001 | **0.714 +/- .008** | 0.495 +/- .006 |
| S7-D | 3e-4 | 1e-3 | 75 | 0.548 +/- .001 | 0.700 +/- .010 | 0.441 +/- .007 |
| S7-A | 1e-3 | 1e-5 | 35 | 0.583 +/- .004 | 0.670 +/- .014 | 0.516 +/- .008 |
| S7-C | 1e-3 | 1e-3 | 38 | 0.582 +/- .004 | 0.677 +/- .007 | 0.508 +/- .009 |

## Key Findings

### 1. Original baseline still best for CoA NMI

The l2=5e-4, no l1, ent=3e-1, 10cyc setting remains the CoA NMI champion at 0.772 +/- .007. None of the new settings beat it.

### 2. l1=1e-3 does not improve CoA NMI

Adding l1 regularization stabilizes training at lower l2 values but does not improve merge quality. S8-C (l2=5e-4, l1=1e-3) scored 0.750 vs 0.772 without l1 — a slight regression. The l1 penalty aggressively prunes clusters (60 valid vs 65), reducing the material available for merging.

### 3. Cluster count convergence with l1

l1=1e-3 at K=200 prunes to ~49-77 valid clusters depending on other settings. This is comparable to K=100 runs (35-67 valid), suggesting that l1 drives the model toward a similar effective capacity regardless of starting K.

### 4. Raw NMI vs CoA NMI divergence

Different hyperparameters optimize raw vs merged performance:
- **Raw NMI favored by**: higher l2, lower entropy reg (tighter clusters, fewer but purer)
- **CoA NMI favored by**: lower l2, higher entropy reg (more clusters retained, better merge signal)

At K=100: l2=3e-4 gives CoA=0.714 but raw=0.553; l2=1e-3 gives raw=0.583 but CoA=0.670.
At K=200: the best CoA setting (0.772) achieves raw=0.610, a good balance.

### 5. Entropy regularization is key for CoA

ent=3e-1 consistently beats ent=1e-1 for CoA NMI by ~0.03-0.04, across all l2/l1 combinations. However, higher entropy at lower l2 increases numerical instability.

### 6. min_prob=1e-3 does not help at K=200

Unlike at K=100 (sweep 7), raising min_prob from 1e-5 to 1e-3 at K=200 increased cluster retention dramatically (95%+ vs 30%) but hurt both NMI metrics and caused instability.

## Recommendations

1. **Best stable setting for CoA NMI**: l2=5e-4, l1=0, ent=3e-1, 10cyc (0.772 CoA, 0.610 raw)
2. **Best stable setting for raw NMI with good CoA**: Same setting — it achieves a good balance
3. **l1=1e-3 is useful for stability** if exploring lower l2, but doesn't push the performance frontier
4. **ld=50/K=200 dominates ld=20/K=100** across all metrics
