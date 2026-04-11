# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

goal-apps is a Python CLI application framework for training, analyzing, and evaluating statistical clustering models on various datasets. Built on the GOAL (Geometric OptimizAtion Libraries) JAX library, it provides a plugin-based architecture for extensible model and dataset implementations.

## Environment

This project uses `uv` for environment and dependency management. Use `uv run` to
execute project commands — do not manually activate the venv.

If you need to access goal-jax code for reference, it is typically under `/home/alex404/code/goal-jax`

### Package management strategy

- `pyproject.toml` — source of truth for deps (bare names, no version pins)
- `uv.lock` — committed lockfile; regenerate with `uv sync --all-extras`
- Do not `uv pip install` into the project env — use `uv add` for permanent deps,
  `uvx`/`uv run --with` for one-off tools

### Local dependencies

goal-jax is resolved from `../goal-jax` via `[tool.uv.sources]` in pyproject.toml.
This is a uv-specific override — standard tools (pip, build) resolve from PyPI.
Contributors must clone goal-jax as a sibling directory (`../goal-jax`) before
running `uv sync`. This override can be removed once a usable version is published
to PyPI.

### Dev tools

basedpyright and ruff are not project dependencies — run them via `uvx`:
- `uvx basedpyright`
- `uvx ruff check .`
- `uvx ruff format .`

## Common Commands

### Installation and Setup
```bash
# Sync all dependencies (core + all extras)
uv sync --all-extras

# Sync core dependencies only
uv sync

# Type checking (basedpyright, not pyright)
uvx basedpyright

# Linting and formatting
uvx ruff check .
uvx ruff format .
```

### Running Experiments

The CLI entry point is `goal` (defined in pyproject.toml as `apps.cli.main:main`).

```bash
# Train a model on a dataset
uv run goal train dataset=mnist model=hmog

# Train with custom parameters
uv run goal train dataset=mnist model=hmog latent_dim=50 n_clusters=200

# Analyze a trained model
uv run goal analyze run_name=<run_name>

# View trained run results (stored in runs/)
ls runs/<run_name>/

# Launch hyperparameter sweep (comma-separated, no brackets)
uv run goal sweep dataset=mnist model=mnist-hmog latent_dim=4,8,12,16 n_clusters=50,100,200

# Dry run to view configuration
uv run goal train dataset=mnist model=hmog --dry-run

# List available plugins
uv run goal plugins list

# Inspect plugin configuration
uv run goal plugins inspect hmog
uv run goal plugins inspect mnist
```

### Resuming and Retraining

```bash
# Resume training from latest checkpoint
uv run goal train run_name=<existing_run>

# Resume from specific epoch
uv run goal train run_name=<existing_run> resume_epoch=50

# Force restart from epoch 0
uv run goal train run_name=<existing_run> resume_epoch=0
```

### Configuration Overrides

Configuration follows Hydra's composition pattern:
- Base config: `config/hydra/config.yaml`
- Dataset configs: `config/hydra/dataset/*.yaml`
- Model configs: `config/hydra/model/*.yaml`

Override any parameter via CLI:
```bash
uv run goal train dataset=mnist model=hmog device=cpu jit=false use_wandb=false
```

## Architecture

### Plugin System

New models and datasets are added as plugins without modifying core code:
- **Models**: `plugins/models/<model_name>/`
- **Datasets**: `plugins/datasets/<dataset_name>/`

Each plugin:
1. Implements the appropriate abstract interface (`Model`, `Dataset`, `ClusteringModel`, `ClusteringDataset`)
2. Defines a dataclass config (inherits from `ModelConfig` or `DatasetConfig`)
3. Registers itself with Hydra's ConfigStore in its `__init__.py`

The `plugins/register_plugins()` function discovers and imports all plugins at startup.

### Core Abstractions

**Model Interface** (`src/apps/interface/model.py`):
- `train(key, handler, logger, dataset)` - Training loop implementation
- `analyze(key, handler, logger, dataset)` - Post-training analysis
- `initialize_model(key, data)` - Parameter initialization
- `get_analyses(dataset)` - Returns list of Analysis instances to run
- `n_epochs` - Total training epochs

**Dataset Interface** (`src/apps/interface/dataset.py`):
- `train_data`, `test_data` - JAX arrays with shape (n_samples, data_dim)
- `observable_shape` - Abstract property for visualization grid sizing
- `paint_observable(observable, axes)` - Visualization of single observation
- `has_labels`, `train_labels`, `test_labels` - Ground truth labels (optional)

**Clustering Interfaces** (`src/apps/interface/clustering/`):
- `ClusteringDataset` extends `Dataset` with `paint_cluster()`, `cluster_shape`
- `ClusteringModel` extends `Model` with `n_parameters` (for BIC)
- Protocols: `HasLogLikelihood`, `IsGenerative`, `HasSoftAssignments`, `CanComputePrototypes`, `HasClusterHierarchy`
- Shared analyses: `ClusterStatisticsAnalysis`, `CoAssignmentHierarchyAnalysis`, `OptimalMergeAnalysis`, `CoAssignmentMergeAnalysis`
- `ClusteringRunConfig` and `ClusteringAnalysesConfig` for structured analysis configuration

**Analysis Interface** (`src/apps/interface/analysis.py`):
- `generate(key, handler, dataset, model, epoch, params)` - Generate artifact
- `plot(artifact, dataset)` - Create matplotlib Figure from artifact
- `metrics(artifact)` - Return MetricDict (optional, defaults to `{}`)
- `process()` - Orchestrates generate/load, then plot and log
- `artifact_type` - Abstract property returning the artifact class

**Shared Analyses** (`src/apps/interface/analyses/`):
- `GenerativeSamplesAnalysis` - Generic generative sampling (used by any `IsGenerative` model)

**RunHandler** (`src/apps/runtime/handler.py`):
- Manages directory structure: `runs/single/<run_name>/epoch_<N>/` (or `runs/sweep/<sweep_id>/...`)
- Saves/loads model parameters, metrics, artifacts
- Handles resumption logic

**Logger** (`src/apps/runtime/logger.py`):
- Buffers metrics for batch logging
- Supports local file logging and W&B integration
- Tracks wall clock time from initialization
- `log_metrics()`, `log_figure()`, `log_artifact()`

**Runtime Metrics** (`src/apps/runtime/metrics.py`):
- `add_ll_metrics()` - Log-likelihood and BIC metrics
- `log_with_frequency()` - JIT-compatible frequency-gated logging

**Clustering Metrics** (`src/apps/interface/clustering/metrics.py`):
- `add_clustering_metrics()` - NMI and accuracy metrics

### Directory Structure

```
goal-apps/
├── config/default.mplstyle  # Matplotlib style configuration
├── src/apps/               # Core application framework
│   ├── cli/                # CLI commands (train, analyze, sweep)
│   ├── interface/          # Abstract base classes
│   │   ├── clustering/     # Clustering-specific interfaces, protocols, and shared analyses
│   │   └── analyses/       # Generic reusable analyses (generative samples)
│   └── runtime/            # RunHandler, Logger, metrics utilities
│
├── plugins/                # Pluggable implementations
│   ├── datasets/           # MNIST, CIFAR-10, SVHN, Tasic, Neural Traces, 20 Newsgroups
│   └── models/             # HMOG, MFA, K-means, LDA
│       ├── hmog/           # Hierarchical Mixture of Gaussians
│       │   ├── model.py    # HMoGModel base class
│       │   ├── trainers.py # Pre-training and gradient trainers
│       │   └── analyses/   # HMoG-specific analyses (KL hierarchy, loadings)
│       └── mfa/            # Mixture of Factor Analyzers
│           ├── model.py    # MFAModel (full and diagonal variants)
│           ├── trainers.py # Single-phase gradient trainer
│           └── configs.py  # MFAConfig, MFADiagonalConfig
│
├── config/hydra/           # Hydra configuration files
│   ├── config.yaml         # Base run configuration
│   ├── dataset/            # Pre-configured dataset settings
│   └── model/              # Pre-configured model settings
│
├── scratch/                # Debug/experiment scripts (not part of the package)
│
└── runs/                   # Training outputs (created at runtime)
    ├── single/<run_name>/  # Regular training runs
    └── sweep/<sweep_id>/   # Sweep runs
        └── <run_name>/
            ├── run-config.yaml
            ├── metrics.joblib
            └── epoch_<N>/
                ├── params.joblib
                └── <analysis_name>.joblib
```

### Training Pipeline

1. **Initialization** (`src/apps/cli/initialize.py`):
   - Compose Hydra config from defaults + overrides
   - Instantiate dataset and model via `hydra.utils.instantiate()`
   - Create RunHandler and Logger
   - Setup matplotlib style and signal handlers (SIGTERM/SIGINT)
   - Save full config to `runs/<run_name>/run-config.yaml`

2. **Training** (`model.train()`):
   - Initialize or load parameters via `prepare_model()`
   - Run training loop (model-specific implementation)
   - Call `process_checkpoint()` periodically to:
     - Save parameters to `epoch_<N>/params.joblib`
     - Run analyses and save artifacts
     - Save metrics buffer

3. **Analysis** (`model.analyze()`):
   - Load trained parameters
   - Execute each Analysis instance from `get_analyses()`
   - Generate visualizations and metrics
   - Results saved to epoch directories

### Batch Size vs Batch Steps

Training supports two distinct strategies controlled by `batch_size` and `batch_steps`:

- **`batch_size=null` (default)**: Full-batch approximate EM. The E-step computes expectations over the entire dataset, then `batch_steps` gradient steps are taken for the M-step against those fixed expectations. This is EM via gradient descent.
- **`batch_size=N`**: Mini-batch SGD. Each step samples a mini-batch of N points. Typically used with `batch_steps=1`. Standard stochastic optimization — a fundamentally different approach from the approximate EM.

### HMOG Model Architecture

The Hierarchical Mixture of Gaussians (HMOG) is the primary model with multi-phase training:

1. **LGMPreTrainer**: Pre-train latent Gaussian mixture components
2. **MixtureGradientTrainer**: Optimize mixture weights and parameters
3. **FullGradientTrainer**: End-to-end gradient-based optimization

**Gradient EM and `bound_means` whitening**: `FullGradientTrainer.bound_means()` applies `model.whiten_prior()` to the bounded posterior statistics (the EM target). The gradient is `prior_stats - bounded_posterior_stats`, so whitening the target means the model's mean parameters converge toward a whitened state where the prior is N(0,I). This is intentional and correct — it keeps the latent prior from drifting. Do NOT treat this as a bug or remove the `whiten_prior` call. This same principle applies to MFA and any other model using gradient EM with whitening.

HMOG-specific analyses (`plugins/models/hmog/analyses/`):
- `KLHierarchyAnalysis`: KL-divergence-based hierarchical clustering of components
- `KLMergeAnalysis`: KL-divergence-based optimal merge analysis
- `LoadingMatrixAnalysis`: PCA-like interpretation of latent dimensions

Shared clustering analyses (from `src/apps/interface/clustering/analyses/`):
- `ClusterStatisticsAnalysis`: Cluster sizes, assignments, prototypes
- `CoAssignmentHierarchyAnalysis`: Co-assignment-based hierarchical clustering
- `OptimalMergeAnalysis` / `CoAssignmentMergeAnalysis`: Cluster merge analyses
- `GenerativeSamplesAnalysis`: Sample new observations (from `src/apps/interface/analyses/`)

### MFA Model Architecture

The Mixture of Factor Analyzers (MFA) is a simpler clustering model:
- Supports both full factor analysis and diagonal covariance variants
- Single-phase gradient training via `GradientTrainer`
- Initialized from k-means cluster centers
- Registered as `mfa` and `mfa-diagonal` in Hydra
- Implements `ClusteringModel`, `HasLogLikelihood`, `IsGenerative`, `HasSoftAssignments`, `CanComputePrototypes`

### Configuration Composition

Hydra composes configs in order: base defaults → dataset config → model config → CLI overrides → saved config (on resume). Model parameters are nested under the `model` key (e.g., `model.n_clusters=60`, `model.full.ent_reg=1e-1`). Top-level params like `run_name`, `device`, `jit`, `use_wandb`, `log_level` are overridden directly.

### Hyperparameter Sweeps

The sweep system (`src/apps/cli/sweep.py`) uses comma-separated overrides (**not** bracket syntax) and creates W&B grid sweeps. Supports `--validate` and `--dry-run` flags. See Running Experiments above for CLI examples.

## Development Patterns

### Adding a New Model

1. Create `plugins/models/<model_name>/` directory
2. Implement model class extending `Model` or `ClusteringModel`
3. Define config dataclass with `_target_` pointing to model class
4. Register with ConfigStore in `__init__.py`:
   ```python
   from hydra.core.config_store import ConfigStore
   cs = ConfigStore.instance()
   cs.store(group="model", name="mymodel", node=MyModelConfig)
   ```
5. Implement required methods: `train()`, `analyze()`, `initialize_model()`, `get_analyses()`
6. Create config file: `config/hydra/model/<dataset>-<model>.yaml`

### Adding a New Dataset

1. Create `plugins/datasets/<dataset_name>/` directory
2. Implement dataset class extending `Dataset` or `ClusteringDataset`
3. Define config dataclass with `_target_` pointing to dataset class
4. Register with ConfigStore
5. Implement: `train_data`, `test_data`, `paint_observable()`, `paint_cluster()` (if clustering)
6. Create config file: `config/hydra/dataset/<name>.yaml`

### Adding a New Analysis

1. Create analysis class (model-specific in `plugins/models/<model>/analyses/`, or shared in `src/apps/interface/clustering/analyses/`)
2. Extend `Analysis[DatasetType, ModelType, ArtifactType]`
3. Implement:
   - `generate()`: Generate artifact from model parameters
   - `plot()`: Create matplotlib Figure from artifact
   - `artifact_type`: Property returning the artifact class
   - `metrics()`: Optional, return MetricDict
4. Add analysis instance to model's `get_analyses()` method
5. Analysis results automatically saved to epoch directories via `process()`

### Working with JAX

- All numerical computations use JAX arrays
- Enable/disable JIT with `jit=true/false`
- Use `device=gpu` or `device=cpu` to control execution backend
- **Single GPU — never run two training jobs simultaneously.** Only one `goal train` process at a time. Do not start a second job while one is already running, even in the background.

### Debugging Numerical Instabilities

**cuSolver / Cholesky / NaN errors** during training are almost always caused by degenerate (non-positive-definite) covariance matrices — i.e. insufficient regularization for the given configuration. Do not assume GPU corruption or hardware issues.

**Diagnosis workflow:**
1. Run with `log_level=STATS` (or `DEBUG`) to get per-component diagnostics every epoch — precision condition numbers, covariance log-determinants, gradient norms per parameter group, and regularization penalties.
2. Load `metrics.joblib` from the run directory and inspect the trajectories of key metrics: `Components/Precision Cond Max`, `Components/Covariance LogDet Min`, `Grad Norms/*`, `Regularization/*`. These reveal which parameter group is diverging and when.
3. The fix is typically to increase `upr_prs_reg` (prevents precision eigenvalue blow-up → covariance collapse) or `min_var`/`lat_min_var` (floors on variance). Reducing model capacity (`n_clusters`, `latent_dim`) also helps.
4. `upr_prs_reg` and `lwr_prs_reg` push latent precision eigenvalues toward `lwr/upr`. Symmetric values (e.g., both 1e-2) push toward 1.0 (isotropic). Asymmetric (upr >> lwr) allows broader distributions but can be numerically unstable at initialization if the ratio is too extreme (e.g., upr=1e-2, lwr=1e-4 crashes; upr=1e-3, lwr=1e-5 is safe).

### Dev Tool Configuration

**basedpyright**: "recommended" mode with relaxations for dynamic Hydra instantiation and JAX idioms (`reportUnknown*Type`, `reportAny`, `reportExplicitAny`, `reportMissingSuperCall`, `reportUnusedCallResult` disabled).

**ruff**: Pyflakes, PEP8, McCabe, naming, imports, simplify, pyupgrade, ruff-specific, unused-arguments, tryceratops. Ignores: E501, D (docstrings), ARG002/003, TRY003, F722 (jaxtyping false positives).

## Key Dependencies

Core: **goal-jax** (local, brings JAX), **Hydra/OmegaConf**, **Typer**, **wandb**, **matplotlib/seaborn**, **scikit-learn**, **joblib**, **scipy**.
Optional extras: **torchvision** + **h5py** (`datasets`), **pytest** (`test`).

