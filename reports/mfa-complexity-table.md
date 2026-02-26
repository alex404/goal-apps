# Batch Log-Likelihood Complexity (Time and Space)

This table focuses on **batch evaluation of log-likelihood**, which is the dominant cost in learning. Caching of data‑independent terms is assumed (i.e., good implementations).

Notation:
- n: batch size
- d: observed dimension
- l: latent dimension
- k: mixture components

Legend for constraints shorthand:
- W_i: component-specific loadings
- A: shared loadings
- D_i: component-specific observation noise
- D: shared observation noise
- I: identity
- diag: diagonal
- sph: spherical
- Omega_i: component-specific factor covariance
- post diag: diagonal posterior precision in EF LGM

Assumptions:
- Time is **per batch** for log-likelihood evaluation with cached data‑independent terms.
- Space counts **model parameters only** (not data).
- If A is shared, we assume per‑point work can share A^T D^-1 x across components.

| Model | Key Constraints | Time: batch log-likelihood | Space (params) |
| --- | --- | --- | --- |
| MFA (standard) | W_i, D_i diag, latent cov I | O(n k d l + k d l^2 + k l^3) | O(k d l + k d + k l) |
| MFA (MPPCA) | W_i, D_i sph | O(n k d l + k d l^2 + k l^3) | O(k d l + k d) |
| MCFA (Baek) | A shared, D diag, Omega_i full | O(n (d l + k l^2) + d l^2 + k l^3) | O(d l + d + k l^2) |
| MCUFSA (Baek) | A orthonormal, D sph, Omega_i diag | O(n (d l + k l)) | O(d l + k l) |
| Diag HMoG (EF) | A shared, D diag, Omega_i diag, post diag | O(n (d l + k l) + d l^2 + k l^3) | O(d l + d + k l) |

Code‑verified EF cost centers (goal‑jax):
- `log_observable_density` uses:
  - `obs_man.sufficient_statistic(x)` (O(d) for diagonal D),
  - `posterior_at(params, x)` (matvec with A: O(d l)),
  - `pst_man.log_partition_function` for the upper mixture (O(k l) if latent is diagonal),
  - `prior(params)` (data‑independent, cacheable) which calls `NormalLGM.conjugation_parameters` and includes a change‑of‑basis term O(d l^2).
- With caching, the O(d l^2) and mixture prior log‑partition terms are paid once per batch, yielding the time line above.
