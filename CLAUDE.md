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

# Launch W&B hyperparameter sweep (comma-separated, no brackets)
uv run goal tune wandb dataset=mnist model=mnist-hmog latent_dim=4,8,12,16 n_clusters=50,100,200

# Create an Optuna study
uv run goal tune optuna create --metric "Clustering/Test ARI" \
    experiment=my-study dataset=mnist model=hmog \
    model.n_clusters=suggest_categorical:10,20,50

# Run a trial against an existing study
uv run goal tune optuna run my-study

# Check study status
uv run goal tune optuna status my-study

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

Read the source for method lists; one-line pointers here:

- `src/apps/interface/model.py` — `Model` base: `train`, `analyze`, `initialize_model`, `get_analyses`, `n_epochs`.
- `src/apps/interface/dataset.py` — `Dataset` base: `train_data`, `test_data`, `observable_shape`, `paint_observable`, optional labels.
- `src/apps/interface/clustering/` — `ClusteringModel`/`ClusteringDataset`, protocols (`HasLogLikelihood`, `IsGenerative`, `HasSoftAssignments`, `CanComputePrototypes`, `HasClusterHierarchy`), shared clustering analyses, `ClusteringRunConfig`/`ClusteringAnalysesConfig`.
- `src/apps/interface/analysis.py` — `Analysis[Dataset, Model, Artifact]`: `generate`, `plot`, `metrics`, `process`, `artifact_type`.
- `src/apps/interface/analyses/` — shared analyses reusable across models (e.g. `GenerativeSamplesAnalysis` for any `IsGenerative` model).
- `src/apps/runtime/handler.py` — `RunHandler`: run directory layout, atomic param/metric/artifact I/O, resume logic.
- `src/apps/runtime/logger.py` — `Logger`: metric buffering, W&B integration, `check_pruning` (Optuna reporting).
- `src/apps/runtime/metrics.py` — generic metric helpers (`add_ll_metrics`, `log_with_frequency`).
- `src/apps/interface/clustering/metrics.py` — clustering metric helpers (`add_clustering_metrics` → NMI/ARI/accuracy).

### Directory Structure

```
goal-apps/
├── config/                 # Matplotlib style + Hydra config (base, dataset/, model/)
├── src/apps/               # Core framework — cli/, interface/ (+ clustering/, analyses/), runtime/
├── plugins/
│   ├── datasets/           # MNIST, CIFAR-10, SVHN, Tasic, Neural Traces, 20 Newsgroups
│   └── models/             # HMoG, MFA, K-means, LDA (each plugin registers with ConfigStore)
├── scratch/                # Debug/experiment scripts (not part of the package)
└── runs/                   # Training outputs (created at runtime)
    ├── single/<run_name>/  # Regular training runs
    └── tune/<study_name>/  # Optuna studies / W&B sweeps
        ├── study-config.yaml, study.db
        └── t<N>/           # One directory per trial: run-config.yaml, metrics.joblib, epoch_<N>/
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

### Navigation Map

Where common concerns live across the codebase:

- **Where metrics come from.** Two sources, both write into the same
  `MetricHistory` buffer: (1) *training-loop metrics* emitted from
  `model.train()` via `logger.log_metrics(...)` — e.g. log-likelihood, BIC,
  clustering NMI/ARI, regularization penalties, timing. These are added by
  helpers like `add_ll_metrics`, `add_clustering_metrics`, `l1_l2_regularizer`
  (`src/apps/runtime/metrics.py`, `src/apps/interface/clustering/metrics.py`).
  (2) *Analysis metrics* emitted from `Analysis.process(...)` via
  `analysis.metrics(artifact)`.
- **`metric_names` is auto-composed for clustering models.**
  `ClusteringModel.metric_names(dataset)` merges: `LL_METRIC_KEYS` if the
  model implements `HasLogLikelihood` (runtime-checkable protocol),
  `CLUSTERING_METRIC_KEYS` if `dataset.has_labels`, whatever the subclass
  declares in the `training_metric_keys` property (model-specific stats /
  regularization keys), and the union of `metric_keys` across
  `get_analyses(dataset)`. Subclasses normally only declare
  `training_metric_keys` and never override `metric_names` itself. Used by
  the Optuna create validator to reject typos in `--metric`.
- **Metric keys are frozensets.** Each helper (`add_ll_metrics`,
  `add_clustering_metrics`, `l1_l2_regularizer`, `update_stats` via
  `stats_keys(...)`) ships with a sibling frozenset of the keys it
  produces. Declaring `training_metric_keys` is just unioning those
  siblings. Update the frozenset whenever you rename or add a key.
- **Optuna study artifacts.** One study dir per experiment:
  `runs/tune/<study>/study-config.yaml` (overrides, metric, direction,
  pruner, storage) and `runs/tune/<study>/study.db` (SQLite). Trial runs
  land in `runs/tune/<study>/t<N>/` with the usual `run-config.yaml`,
  `metrics.joblib`, `epoch_<N>/` layout. The `pruner:` key is a nested
  mapping (`{name, n_startup_trials, n_warmup_steps, …}`); legacy bare
  strings (`pruner: median`) still load via back-compat in
  `sweep.py:_normalize_pruner_config`.
- **Divergence-to-prune flow.** `Logger.monitor_params` is called inside
  jitted training steps. On NaN it enters an `io_callback` that flips the
  module-level `_divergence_raised` flag and raises
  `DivergentTrainingError`. JAX wraps that in `XlaRuntimeError` with no
  exception chaining, so `sweep.py`'s trial objective reads
  `Logger.divergence_raised()` in its `except Exception` branch and
  converts to `optuna.TrialPruned` — never string-match on the error.
  Divergence is treated as a pruning signal (not `TrialState.FAIL`)
  because stability regularizers are themselves tuned hyperparameters:
  a diverging config means the sampled regularizer regime can't hold
  this model, i.e. effectively −∞ performance. Pruned trials feed TPE's
  density model so it learns to avoid that neighborhood; `FAIL` would
  discard them. At the catch site, `sweep.py` sets
  `trial.user_attrs["diverged"] = True` so divergence is distinguishable
  from genuine mid-training prunes downstream. Classification goes
  through `is_diverged_trial` in `cli/util.py`, which reads the tag and
  falls back to "PRUNED with no intermediate values" for legacy trials
  that predate the tag. `goal tune optuna import` imports diverged
  PRUNEDs by default (definitive bad-neighborhood signal for TPE);
  `--include-pruned` still gates non-divergent PRUNEDs (fate ambiguous,
  param-coverage only).
- **Atomic checkpoints.** `handler._atomic_dump` does dump-to-`.tmp` +
  `os.replace` for params, metrics, and artifacts so a crashed write
  never leaves a truncated resume target. `save_debug_state` is
  intentionally non-atomic (debug only).
- **Metrics fail fast.** Clustering metrics (`clustering_nmi`,
  `clustering_ari`) do NOT guard degenerate denominators. A single-
  cluster / single-class split genuinely has undefined NMI/ARI, and the
  maintainer prefers `NaN` in the log over a fudged zero that silently
  poisons Optuna. If you find yourself tempted to add `+ eps` to a
  denominator, stop and surface the degeneracy instead.
- **Entropy uses library primitive in mean coordinates.** Entropy
  computations delegate to `Categorical.negative_entropy`, which takes
  *mean coordinates* (K-1 dim; the 0th probability is dropped by
  `from_probs` and reconstructed inside the library as `1 - sum(means)`).
  Do not hand-roll `jnp.log` / `xlogy` loops in metric code — use
  `cat.negative_entropy(cat.from_probs(probs))`. Natural coordinates
  (log-odds) would give identical entropy via `dual_potential`, but
  clustering probabilities are already in mean form, so `from_probs` is
  the zero-conversion path.
- **Hierarchy filters once, merge reuses.** `CoAssignmentHierarchyAnalysis`
  and `KLHierarchyAnalysis` take `filter_empty_clusters` /
  `min_cluster_size` and persist the resulting `valid_clusters` inside
  the `ClusterHierarchy` artifact. Downstream `CoAssignmentMergeAnalysis`
  and `KLMergeAnalysis` load the hierarchy artifact and read
  `hierarchy.valid_clusters` — they do NOT expose their own filter
  fields and must run at an epoch where the matching hierarchy artifact
  already exists. `OptimalMergeAnalysis` doesn't depend on a hierarchy
  (Hungarian on the raw contingency) and keeps its own filter fields.
  `model.get_analyses()` controls the ordering.

### Testing Protocol (Gold Standard)

For library-wide changes (`src/apps`, trainers, Optuna integration,
config plumbing), the most thorough smoke test is a small Optuna study:

```bash
uv run goal tune optuna create --metric "Clustering/Test ARI" \
    experiment=smoke dataset=mnist model=mnist-hmog-ld20 \
    use_wandb=false \
    <small search space>
uv run goal tune optuna run smoke  # a handful of trials
uv run goal tune optuna status smoke
```

This exercises config composition, training, analyses, metric logging,
Optuna pruning/reporting, divergence handling, and checkpoint I/O in a
single run. Heavier than `--dry-run`; not required for every change, but
the right default when touching cross-cutting code. Pass
`use_wandb=false` for local verification without creating public W&B
runs — the local metrics buffer and Optuna DB are still populated.

### Batch Size vs Batch Steps

Training supports two distinct strategies controlled by `batch_size` and `batch_steps`:

- **`batch_size=null` (default)**: Full-batch approximate EM. The E-step computes expectations over the entire dataset, then `batch_steps` gradient steps are taken for the M-step against those fixed expectations. This is EM via gradient descent.
- **`batch_size=N`**: Mini-batch SGD. Each step samples a mini-batch of N points. Typically used with `batch_steps=1`. Standard stochastic optimization — a fundamentally different approach from the approximate EM.

### HMOG Model Architecture

Hierarchical Mixture of Gaussians — primary model. Inside each cycle it
runs three gradient sub-phases: `lgm` (LGM params, mixture frozen) →
`mix` (mixture params, LGM fixed) → `full` (end-to-end). An optional
`pre` (LGMPreTrainer) phase runs once before cycling begins. See
`plugins/models/hmog/`.

**Gradient EM and `bound_means` whitening**: `FullGradientTrainer.bound_means()` applies `model.whiten_prior()` to the bounded posterior statistics (the EM target). The gradient is `prior_stats - bounded_posterior_stats`, so whitening the target means the model's mean parameters converge toward a whitened state where the prior is N(0,I). This is intentional and correct — it keeps the latent prior from drifting. Do NOT treat this as a bug or remove the `whiten_prior` call. This same principle applies to MFA and any other model using gradient EM with whitening.

### MFA Model Architecture

Mixture of Factor Analyzers — simpler than HMoG. Registered as `mfa`
and `mfa-diagonal`. Single `trainer` phase inside each cycle; k-means
init, no pretraining. Shares the cyclic driver with HMoG (see below).

### Cyclic Training and Optuna Pruning

HMoG and MFA both expose `num_cycles` + `lr_scales` at the model-config
level. The shared driver lives in
`src/apps/interface/clustering/model.py` and:
- writes a checkpoint at the end of every cycle (so Optuna's
  `check_pruning` sees each cycle as a reportable step),
- resumes mid-training by mapping the loaded epoch back to a cycle
  index — cycle boundaries are the only resumable points, mid-cycle
  state is not restored,
- interpolates `lr_scales` (arbitrary-length keypoint list) across
  `num_cycles` so a small set of anchor values drives any number of
  cycles.

Within a cycle, HMoG still runs three sub-phases (`lgm` / `mix` /
`full`); MFA runs a single `trainer` phase. The driver is agnostic to
what happens inside a cycle — the model's `run_cycle` callback owns that.

**Pruning step units.** `Logger.check_pruning` calls
`trial.report(value, epoch)` — the step passed to Optuna is the raw
epoch count, not the cycle index. To give trials N free cycles before
pruning, set `--pruner-warmup-steps` to `N × epochs_per_cycle` (for
HMoG: `lgm.n_epochs + mix.n_epochs + full.n_epochs`; for MFA:
`full.n_epochs`). Change epochs-per-cycle → re-tune the flag.

### Single-Source Seed Convention

All randomness in goal-apps flows from the CLI-set `seed` field on
`RunConfig` (default 0). `initialize_run` threads `seed=cfg.seed`
into `instantiate(cfg.dataset, ...)`, so dataset `load` methods receive
it as a keyword argument. Do NOT add local `random_seed` fields to
dataset configs or model configs — the CLI seed is the single source of
truth. Sklearn components that need a Python `int` seed should derive
it deterministically from the JAX key via
`int(jax.random.randint(key, (), 0, 2**31 - 1))`.

### Configuration Composition

Effective precedence (later wins): base defaults ⟶ Hydra group selection (`dataset=X`, `model=Y`) ⟶ saved `run-config.yaml` (if resuming) ⟶ CLI overrides. On resume, `filter_group_defaults` (in `cli/initialize.py`) drops any `dataset=` / `model=` override because the saved subtree is already composed and a bare scalar would clobber it; subkey overrides like `dataset.n_samples=1000` still take effect. Model parameters are nested under `model` (e.g. `model.n_clusters=60`, `model.full.ent_reg=1e-1`); top-level params like `run_name`, `device`, `jit`, `use_wandb`, `log_level` override directly.

### Hyperparameter Tuning

The `goal tune` command supports two backends:

**W&B sweeps** (`goal tune wandb`): Grid/random sweeps via Weights & Biases. Uses comma-separated overrides (**not** bracket syntax). Supports `--validate` and `--dry-run` flags.

**Optuna** (`goal tune optuna`): TPE-guided search with pruning. Workflow:
1. `goal tune optuna create` — define search space, create study DB + config
2. `goal tune optuna run <study>` — run one trial (submit as SLURM jobs)
3. `goal tune optuna status <study>` — check progress from any node
4. `goal tune optuna reset <study>` — wipe DB, keep config
5. `goal tune optuna clear <study>` — delete entire study
6. `goal tune optuna import <source> <target>` — port historical
   trials from another study, re-extracting the target's metric from
   each source trial's `metrics.joblib`. Tolerates range divergence
   and asymmetric param sets; errors on distribution-family
   conflicts on shared names (e.g. float↔categorical).

Search-space directives in overrides:
- `param=suggest_float:low:high[:log]`
- `param=suggest_int:low:high`
- `param=suggest_categorical:a,b,c`
- `param=suggest_optional:<off>:<inner>` — boolean gate (at
  `param__enabled`) chooses `<off>` vs. the inner directive;
  useful for "maybe apply this regularizer".

Pruner configuration (per-study, persisted in `study-config.yaml`):
- `--pruner` = `median` (default), `percentile`, or `none`.
- `--pruner-startup-trials` — free trials before any pruning (default 3).
- `--pruner-warmup-steps` — **epochs** before a trial is prunable. Set
  to `N × epochs_per_cycle` for N free cycles (see § Cyclic Training).
- `--pruner-interval-steps`, `--pruner-min-trials` — standard Optuna knobs.
- `--pruner-percentile N` — keep top N% at each step (25 = aggressive,
  75 = gentle). Setting this auto-selects `--pruner=percentile`.

Metric validation: `--validate` (default) probes one sampled config,
instantiates model+dataset, and errors if `--metric` isn't in the
model's declared `metric_names`. `--no-validate` skips.

Resume semantics: a preempted trial resumes from the last cycle
boundary checkpoint (mid-cycle state is not restored). Study config
at `runs/tune/<study>/study-config.yaml` is hand-editable.

## Development Patterns

### Adding a New Model / Dataset / Analysis

Copy an existing plugin as a template and adapt it:

- **Model**: copy `plugins/models/mfa/` (simpler than HMoG). Extend
  `Model` or `ClusteringModel`, implement `train`/`analyze`/
  `initialize_model`/`get_analyses`, register with ConfigStore in
  `__init__.py`, add `config/hydra/model/<dataset>-<model>.yaml`.
- **Dataset**: copy `plugins/datasets/mnist/`. Extend `Dataset` or
  `ClusteringDataset`, implement `train_data`/`test_data`/
  `paint_observable` (and `paint_cluster` for clustering), register,
  add `config/hydra/dataset/<name>.yaml`.
- **Analysis**: model-specific go under `plugins/models/<m>/analyses/`;
  reusable under `src/apps/interface/clustering/analyses/`. Extend
  `Analysis[DatasetType, ModelType, ArtifactType]`, implement
  `generate`/`plot`/`artifact_type` (and optionally `metrics`), then
  add an instance to the model's `get_analyses()`.

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

Core: **goal-jax** (local, brings JAX), **Hydra/OmegaConf**, **Typer**, **wandb**, **optuna**, **matplotlib/seaborn**, **scikit-learn**, **joblib**, **scipy**.
Optional extras: **torchvision** + **h5py** (`datasets`), **pytest** (`test`).

