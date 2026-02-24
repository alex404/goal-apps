# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

goal-apps is a Python CLI application framework for training, analyzing, and evaluating statistical clustering models on various datasets. Built on the GOAL (Geometric OptimizAtion Libraries) JAX library, it provides a plugin-based architecture for extensible model and dataset implementations.

## Environment

This project uses **uv** with a virtual environment at `.venv/`. All commands must be run inside this environment. Always activate it before running anything:

```bash
source .venv/bin/activate
```

All tools (basedpyright, ruff, the `goal` CLI, Python itself) are installed in this venv. Do **not** use system Python or install packages outside the venv.

If you need to access goal-jax code for reference, it is typically under `/home/alex404/code/goal-jax`

## Common Commands

### Installation and Setup
```bash
# Install the package in editable mode
uv pip install -e .

# Install with GPU support
uv pip install -e ".[gpu]"

# Install with dataset support (torchvision, h5py)
uv pip install -e ".[datasets]"

# Type checking (basedpyright, not pyright)
basedpyright

# Linting and formatting
ruff check .
ruff format .
```

### Running Experiments

The CLI entry point is `goal` (defined in pyproject.toml as `apps.cli.main:main`).

```bash
# Train a model on a dataset
goal train dataset=mnist model=hmog

# Train with custom parameters
goal train dataset=mnist model=hmog latent_dim=50 n_clusters=200

# Analyze a trained model
goal analyze run_name=<run_name>

# View trained run results (stored in runs/)
ls runs/<run_name>/

# Launch hyperparameter sweep (comma-separated, no brackets)
goal sweep dataset=mnist model=mnist-hmog latent_dim=4,8,12,16 n_clusters=50,100,200

# Dry run to view configuration
goal train dataset=mnist model=hmog --dry-run

# List available plugins
goal plugins list

# Inspect plugin configuration
goal plugins inspect hmog
goal plugins inspect mnist
```

### Resuming and Retraining

```bash
# Resume training from latest checkpoint
goal train run_name=<existing_run>

# Resume from specific epoch
goal train run_name=<existing_run> resume_epoch=50

# Force restart from epoch 0
goal train run_name=<existing_run> resume_epoch=0
```

### Configuration Overrides

Configuration follows Hydra's composition pattern:
- Base config: `config/hydra/config.yaml`
- Dataset configs: `config/hydra/dataset/*.yaml`
- Model configs: `config/hydra/model/*.yaml`

Override any parameter via CLI:
```bash
goal train dataset=mnist model=hmog device=cpu jit=false use_wandb=false
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
- `add_clustering_metrics()` - NMI and accuracy metrics
- `log_with_frequency()` - JIT-compatible frequency-gated logging

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

### HMOG Model Architecture

The Hierarchical Mixture of Gaussians (HMOG) is the primary model with multi-phase training:

1. **LGMPreTrainer**: Pre-train latent Gaussian mixture components
2. **MixtureGradientTrainer**: Optimize mixture weights and parameters
3. **FullGradientTrainer**: End-to-end gradient-based optimization

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

Hydra composes configurations in this order (later overrides earlier):
1. Base defaults from `config/hydra/config.yaml`
2. Dataset-specific config from `config/hydra/dataset/<name>.yaml`
3. Model-specific config from `config/hydra/model/<name>.yaml`
4. CLI overrides (e.g., `latent_dim=100`)
5. Saved config when resuming (loaded from `runs/<run_name>/run-config.yaml`)

Common config parameters:
- `run_name`: Experiment identifier (default: timestamp)
- `device`: "gpu" or "cpu"
- `jit`: Enable JAX JIT compilation (default: true)
- `use_wandb`: Enable Weights & Biases logging (default: true)
- `use_local`: Enable local file logging (default: true)
- `resume_epoch`: Epoch to resume from (None = latest, 0 = fresh start)
- `recompute_artifacts`: Force recomputation of analysis artifacts (default: true)
- `repeat`: Run repetition count (default: 1)
- `log_level`: Configurable log level (INFO, DEBUG, STATS, WARNING, etc.)
- `project`: W&B project name (default: "goal")
- `group`, `job_type`, `run_id`, `sweep_id`: W&B tracking fields

### Hyperparameter Sweeps

Sweeps use Weights & Biases:
```bash
# Create sweep (comma-separated values, no brackets)
goal sweep dataset=mnist model=mnist-hmog latent_dim=10,20,50 n_clusters=50,100,200

# Validate a sample config without launching
goal sweep dataset=mnist model=mnist-hmog latent_dim=10,20,50 --validate

# Dry run to view sweep config tree
goal sweep dataset=mnist model=mnist-hmog latent_dim=10,20,50 --dry-run

# Specify W&B project
goal sweep dataset=mnist model=mnist-hmog latent_dim=10,20,50 --project my-project

# After launching, run agents to execute the sweep:
wandb agent <sweep_id>
```

The sweep system (`src/apps/cli/sweep.py`):
- Parses comma-separated overrides (e.g., `param=1,2,3`) — **not** bracket syntax
- Creates W&B grid sweep configuration
- Supports validation via `--validate` flag (initializes a sample config to check validity)

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
- Random keys generated from `os.urandom()` at run start
- Use `device=gpu` or `device=cpu` to control execution backend

### Type Checking Notes

Pyright is configured with "recommended" mode. Some checks are disabled:
- `reportUnknownParameterType`, `reportUnknownMemberType`, `reportUnknownArgumentType`, `reportUnknownVariableType`, `reportUnknownLambdaType`
- `reportAny`, `reportExplicitAny`
- `reportMissingSuperCall`, `reportUnusedCallResult`
- These relaxations accommodate dynamic Hydra instantiation patterns and JAX idioms

### Ruff Configuration

Enabled rules: Pyflakes, PEP8, McCabe, naming, imports, simplify, pyupgrade, ruff-specific, unused-arguments, tryceratops

Ignored rules:
- E501 (line length)
- D (all docstring checks)
- ARG002, ARG003 (unused arguments in methods/functions)
- TRY003 (exception message in place)
- F722 (false positives from jaxtyping)

## Key Dependencies

- **JAX** (goal-jax): Numerical computing with autodiff
- **Hydra + OmegaConf**: Configuration management
- **Typer**: CLI framework
- **wandb**: Experiment tracking and sweeps
- **matplotlib + seaborn**: Visualization
- **scikit-learn**: Metrics and benchmark models
- **joblib**: Serialization of parameters/artifacts
- **pandas**: Data manipulation
- **scipy**: Scientific computing
- **torchvision**: Vision dataset loaders (optional, `datasets` extra)
- **h5py**: HDF5 file access for Tasic and Neural Traces datasets (optional, `datasets` extra)
- **pytest**: Testing (optional, `test` extra)

## File Locations

- CLI entry: `src/apps/cli/main.py`
- Abstract interfaces: `src/apps/interface/`
- Clustering interfaces: `src/apps/interface/clustering/`
- Shared analyses: `src/apps/interface/analyses/`, `src/apps/interface/clustering/analyses/`
- Runtime utilities: `src/apps/runtime/`
- Plugin registration: `plugins/__init__.py`
- Base config: `config/hydra/config.yaml`
- Matplotlib style: `config/default.mplstyle`
- Run outputs: `runs/single/<run_name>/` or `runs/sweep/<sweep_id>/<run_name>/`
- Dataset cache: `.cache/`
