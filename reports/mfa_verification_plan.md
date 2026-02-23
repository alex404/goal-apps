# MFA Implementation Verification Plan

## Summary of Concerns

1. **Training degrades cluster quality** - NMI drops during training
2. **Model goes NaN** - Parameters become NaN before convergence
3. **Cluster collapse** - One component absorbs all data

## Architecture Overview

The MFA is implemented as:
```
MixtureOfConjugated[Normal, Normal]
  ├── obs_man: Normal (diagonal covariance) - observable X
  ├── lat_man: CompleteMixture[Normal] - latent (Y, Z)
  │     ├── obs_man: Normal (full covariance) - per-component latent Y
  │     └── lat_man: Categorical - mixture indicator Z
  └── int_man: BlockMap[3 blocks] - interactions
        ├── xy: base FA interaction
        ├── xyk: per-component interaction offsets
        └── xk: per-component observable offsets
```

## Verification Tests

### 1. Initialization Verification

**Goal**: Verify that k-means initialization produces valid MFA parameters that behave like k-means.

**Tests**:
- [ ] Verify component means match k-means centers exactly
- [ ] Verify covariances are isotropic (diagonal with uniform variance)
- [ ] Verify interactions are small/zero (with init_scale=0)
- [ ] Verify latent params are standard normal N(0, I)
- [ ] Verify cluster assignments match k-means 100% at init

**Potential Issue**: The `from_mixture_params` / `to_mixture_params` conversion might not be inverses.

### 2. Log-Likelihood Verification

**Goal**: Verify log-likelihood computation is mathematically correct.

**Formula** (from harmonium.py:406):
```
log p(x) = θ_X · s_X(x) + ψ_Z(posterior_at(x)) - ψ_Z(prior) - ψ_X(θ_X) + log μ_X(x)
```

**Tests**:
- [ ] Compare log-likelihood to sklearn's GaussianMixture on simple data
- [ ] Verify log-likelihood is finite for all training data
- [ ] Check that log-likelihood increases monotonically in early epochs
- [ ] Numerical gradient check: ∂LL/∂θ via finite differences vs autodiff

**Potential Issues**:
- `log_partition_function` might overflow/underflow
- `posterior_at` might return invalid parameters

### 3. Posterior Computation Verification

**Goal**: Verify p(z|x) is computed correctly.

**Flow**:
1. `posterior_at(params, x)` → natural params of p(z|x)
2. `lat_man.prior(posterior)` → categorical natural params
3. `to_mean(cat_natural)` → categorical mean params
4. `to_probs(cat_mean)` → probabilities

**Tests**:
- [ ] Verify soft assignments sum to 1
- [ ] Verify soft assignments are all in [0, 1]
- [ ] Compare to manual computation: p(z|x) ∝ p(x|z)p(z)
- [ ] Verify hard assignments correspond to argmax of soft

**Potential Issues**:
- We fixed a bug where `to_probs` was called with natural instead of mean coords
- There might be other coordinate system confusions

### 4. Gradient Computation Verification

**Goal**: Verify the HMOG-style gradient is correct.

**Current gradient**:
```python
posterior_stats = mfa.mean_posterior_statistics(params, batch)
bounded_posterior_stats = self.bound_means(mfa, posterior_stats)
prior_stats = mfa.to_mean(params)
grad = prior_stats - bounded_posterior_stats + reg_grad
```

**Tests**:
- [ ] Verify gradient matches autodiff of negative log-likelihood
- [ ] Check gradient on simple 2D data with known solution
- [ ] Verify gradient is zero at MLE (without regularization)
- [ ] Check gradient norm remains bounded during training

**Potential Issues**:
- `mean_posterior_statistics` might not be computing correct expectations
- `bound_means` might be incorrectly modifying statistics
- The gradient formula might be sign-flipped or missing terms

### 5. Bound Means Verification

**Goal**: Verify bound_means produces valid parameters.

**Current logic**:
1. Split into obs, int, lat components
2. Regularize observable covariance (add jitter, clip min variance)
3. Reset ALL latent components to standard normal N(0, I)
4. Clip categorical probabilities to [min_prob, 1] and renormalize

**Tests**:
- [ ] Verify output is in valid parameter space
- [ ] Verify latent means are exactly standard normal after bounding
- [ ] Verify observable covariances remain positive definite
- [ ] Verify categorical probabilities sum to 1

**Potential Issues**:
- `lat_man.obs_man.standard_normal()` might not be what we expect
- `join_mean_mixture` might not correctly combine components
- Coordinate conversion after bounding might fail

### 6. Precision Regularization Verification

**Goal**: Verify precision regularization doesn't cause instability.

**Current logic**:
```
R(Λ) = upr_prs_reg * tr(Λ) - lwr_prs_reg * log|Λ|
```

**Tests**:
- [ ] Verify regularizer is finite for all valid parameters
- [ ] Check that regularizer gradient doesn't explode
- [ ] Test with disabled regularization (upr_prs_reg=lwr_prs_reg=0)
- [ ] Verify eigenvalues stay reasonable during training

**Potential Issues**:
- `slogdet` might return -inf for near-singular matrices
- Regularization might push in wrong direction

### 7. Numerical Stability Verification

**Goal**: Identify sources of NaN/Inf.

**Tests**:
- [ ] Add NaN/Inf checks after each major computation
- [ ] Track which parameter first becomes NaN
- [ ] Test with jax.debug.print to trace values
- [ ] Check for division by small numbers, log of negative, etc.

## Simplified Test Case

Create a minimal test with:
- 2D data, 3 clusters, 1D latent
- Well-separated Gaussian clusters
- Compare against known analytical solution

```python
# Test data: 3 well-separated 2D Gaussians
centers = [(0, 0), (5, 0), (2.5, 4)]
data = generate_clusters(centers, cov=0.5*I, n_per=100)

# Expected: MFA should recover these centers
# With zero interactions, MFA = GMM with diagonal covariance
```

## Comparison with gans-n-gmms

Key differences to verify:
1. They use N(0, I) latent for ALL components (we do this in bound_means)
2. They use raw SGD, we use AdamW
3. They initialize via k-means + per-cluster FA
4. They might have different parameterization

## Ground-Truth Recovery Tests

The best verification is to show MFA can recover known parameters from samples.

### Test 1: Trivial MFA (= GMM)
- 2D data, 3 clusters, 1D latent
- Zero interactions (loading matrix = 0)
- This should behave exactly like a GMM with diagonal covariance
- **Pass criterion**: Recover cluster means within tolerance

### Test 2: Simple MFA
- 2D data, 2 clusters, 1D latent
- Small non-zero interactions
- **Pass criterion**: Recover means and rough loading structure

### Test 3: Standard MFA
- 10D data, 3 clusters, 2D latent
- Moderate interactions
- **Pass criterion**: Log-likelihood close to ground-truth, good cluster recovery

### Test 4: MNIST-scale MFA
- 784D data (or 50D after PCA), 10 clusters, 10D latent
- **Pass criterion**: NMI improves or stays stable during training

## Verification Results (so far)

### PASSED ✓
1. **Log-likelihood computation** - Matches manual GMM calculation exactly
2. **Posterior computation** - Matches manual Bayes rule calculation exactly
3. **Gradient direction** - HMOG gradient has correlation=1.0 with autodiff gradient
4. **Gradient step** - Single gradient step improves LL
5. **Parameter conversion** - `from_mixture_params` and `to_mixture_params` roundtrip correctly
6. **Initialization** - Component variances are correct (0.1), means match k-means

### FAILED / NEEDS INVESTIGATION ✗
1. **Ground-truth recovery** - Training causes:
   - LL to EXCEED ground-truth (impossible for MLE)
   - One component becomes NaN
   - All data assigned to NaN component
   - NMI drops to 0

### Suspected Issues
1. **No upper bound on covariance** - `regularize_covariance` only has min_var, no max_var
2. **Component collapse dynamics** - When one component dominates, others may degenerate
3. **`bound_means` interaction with gradients** - Resetting latent every step may interfere with optimization

## Next Steps

1. Add max_var to `regularize_covariance`
2. Add NaN checks during training to catch early failures
3. Test with simpler optimizer (pure SGD) to rule out Adam issues
4. Try disabling `bound_means` latent reset temporarily

## Priority Order

1. **Ground-truth recovery (trivial case)** - If this fails, something fundamental is broken
2. **Numerical stability** - Find where NaN originates
3. **Gradient verification** - Ensure we're descending correctly
4. **Initialization** - Ensure we start in valid region
5. **Bound means** - Ensure bounds don't break parameters
6. **Log-likelihood** - Ensure we're optimizing right objective
