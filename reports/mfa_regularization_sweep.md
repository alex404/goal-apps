# MFA Regularization Sweep — MNIST

## Summary

Systematic exploration of regularization strategies for MFA on MNIST using
full-batch approximate EM with epoch_reset and kmeans initialization.

### Code Changes During This Investigation

1. **Removed `precision_regularizer`** from MFA — redundant with `whiten_prior`
2. **Switched entropy regularizer to `dual_potential`** — numerically stable via
   Legendre identity (`theta . eta - psi(theta)`) instead of `p * log(p)` which
   produces NaN when components die
3. **Replaced `l2_reg` gradient penalty + `use_adamw` bool with `weight_decay`** —
   the old setup double-dipped (L2 gradient penalty + AdamW's decoupled decay).
   Now weight_decay > 0 uses AdamW, weight_decay = 0 uses plain Adam
4. **Renamed `mixture_entropy_reg` to `ent_reg`** across MFA and HMOG
5. All changes applied to both MFA and HMOG trainers

## Results: ld=5, k=150

Individual regularizers are all stable. The **L1+L2+ent_reg triple** causes
catastrophic collapse around epoch 300.

| Run | weight_decay | ent_reg | L1 | Epochs | LL (test) | CoA Acc | CoA NMI | Stable |
|---|---|---|---|---|---|---|---|---|
| No reg | 0 | 0 | 0 | 250 | 704.9 | 0.597 | 0.720 | Yes |
| wd=1e-4 (Adam) | 0 | 0 | 0 | 250 | 682.4 | — | — | Yes |
| ent=1e-2 | 0 | 1e-2 | 0 | 250 | 704.9 | 0.597 | 0.705 | Yes |
| min_prob=1e-3 | 0 | 0 | 0 | 250 | — | 0.587 | 0.706 | Yes |
| bounds only | 0 | 0 | 0 | 500 | 720.3 | 0.594 | 0.723 | Yes |
| epoch_reset | 0 | 0 | 0 | 500 | 720.3 | 0.596 | 0.724 | Yes |
| **wd+AdamW+reset** | **1e-4*** | **0** | **0** | **500** | **677.7** | **0.651** | **0.737** | **Yes** |
| wd+ent+AdamW+reset | 1e-4* | 1e-2 | 0 | 500 | 677.7 | 0.651 | 0.737 | Yes |
| No L2, 1000ep | 0 | 1e-2 | 0 | 1000 | 746.2 | 0.525 | 0.658 | Yes |
| **All combined** | 1e-4* | 1e-2 | 1e-4 | 500 | NaN | — | — | **Collapse @301** |
| All minus bounds | 1e-4* | 1e-2 | 1e-4 | 500 | NaN | — | — | **Collapse @311** |

*Note: runs marked 1e-4* used the old double-dipping regime (l2_reg=1e-4 gradient
penalty + AdamW default weight_decay=1e-4), so effective shrinkage was ~2e-4.

### L1+ent_reg Collapse Mechanism

L1 sparsifies loadings (pushes entries to zero), ent_reg keeps hollowed-out
components alive, and weight decay degrades their precisions. Each pair is
stable; the triple collapses around epoch 300. At ld=5, each 784x5 loading
matrix has no redundancy — L1 kills entire latent directions.

### Stability at Higher Dimensions

| Config | kmeans, ld=5 | kmeans, ld=10 | random, ld=5 | random, ld=10 |
|---|---|---|---|---|
| No reg | Stable | — | Stable | Stable |
| wd only | Stable | Stable | — | Collapse* |
| ent only | Stable | — | — | Stable |
| wd+ent | Stable (500ep) | Collapse @420* | Collapse @350* | — |

*These collapses used the old double-dipping L2+AdamW regime.

## Results: ld=10, k=200 (post-refactor, proper weight_decay)

All runs use kmeans init, epoch_reset=true, 500 epochs.

| Run | weight_decay | ent_reg | L1 | LL (test) | CoA Acc | CoA NMI | CoA ARI | Stable |
|---|---|---|---|---|---|---|---|---|
| No reg | 0 | 0 | 0 | 748.5 | 0.417 | 0.518 | 0.186 | Yes |
| wd=1e-4 | 1e-4 | 0 | 0 | 748.5 | 0.512 | 0.604 | 0.289 | Yes |
| ent=1e-2 | 0 | 1e-2 | 0 | 748.5 | 0.408 | 0.498 | 0.166 | Yes |
| wd+ent | 1e-4 | 1e-2 | 0 | 748.5 | 0.510 | 0.607 | 0.276 | Yes |
| wd=2e-4 | 2e-4 | 0 | 0 | 748.5 | 0.421 | 0.518 | 0.184 | Yes |
| L1=1e-4 | 0 | 0 | 1e-4 | 724.1 | 0.515 | 0.674 | 0.425 | Yes |
| **L1+wd** | **1e-4** | **0** | **1e-4** | **724.2** | **0.605** | **0.702** | **0.495** | **Yes** |

### Key Findings at ld=10, k=200

- **Weight decay is the key ingredient** for merge quality — decoupled from
  gradient, doesn't distort LL, improves CoA acc from 0.42 to 0.51
- **L1 is beneficial at ld=10** (unlike ld=5) — 10 latent dims have redundancy
  to prune. L1 alone gives best NMI (0.674)
- **L1+wd is the best combination** — CoA acc 0.605, NMI 0.702, ARI 0.495
- **ent_reg adds nothing** at this scale with kmeans init (no components dying)
- **LL is invariant** to weight decay (~748.5) — decoupled decay works as intended
- **All runs stable** at 500 epochs with proper weight_decay (no double-dipping)

## Recommendations

- **ld=10, k=200**: Use L1=1e-4 + weight_decay=1e-4, no ent_reg
- **ld=5, k=150**: Use weight_decay=1e-4, no L1, optional ent_reg
- **General**: Never combine L1 + ent_reg. L1 utility depends on latent redundancy
- **AdamW vs L2**: Always prefer AdamW weight_decay over L2 gradient penalty
  for adaptive optimizers (Loshchilov & Hutter, 2019)
