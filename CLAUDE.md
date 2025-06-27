# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **goal-apps**, a high-level applications layer built on top of the **goal-jax** library (located at `../goal-jax`). The project provides machine learning applications, particularly focused on clustering and statistical modeling, with a plugin-based architecture.

## Development Environment

This project uses standard Python packaging with `pip` for dependency management. The project depends on the **goal-jax** library which should be installed in development mode:

```bash
# Install goal-jax in development mode first
cd ../goal-jax
pip install -e .

# Then install goal-apps with dependencies
cd ../goal-apps  
pip install -e .
```

For GPU support, install additional dependencies:
```bash
pip install -e ".[gpu]"
```

For datasets (torchvision):
```bash
pip install -e ".[datasets]"
```

## Using goal-jax in this Context

The **goal-jax** library provides the core geometric optimization framework that this application layer builds upon. Key integration points:

### Installation and Setup
- **goal-jax** must be installed in development mode: `cd ../goal-jax && pip install -e .`
- **goal-jax** provides the core statistical manifold abstractions and models
- This project (`goal-apps`) depends on `goal-jax` as listed in `pyproject.toml:6`

### Core Architecture Integration
- **goal-jax** provides the geometric foundation through `src/goal/geometry/` and `src/goal/models/`
- **goal-apps** builds high-level applications using these primitives
- The plugin system in `plugins/` extends goal-jax models with dataset-specific implementations

### Key Dependencies from goal-jax
- JAX for automatic differentiation and computation
- Optax for optimization algorithms
- Geometric abstractions: `ExponentialFamily`, `Manifold`, coordinate systems (`Natural`/`Mean`)
- Statistical models: Normal, Mixtures, Linear Gaussian Models, etc.

## Common Commands

### Main CLI Tool
The project provides a CLI tool called `goal`:
```bash
goal train dataset=mnist model=hmog
goal analyze run_name=my_experiment  
goal sweep latent_dim=[4,8,12] n_clusters=[4,8,16]
goal plugins list
goal plugins inspect hmog
```

### Testing
Run tests using pytest:
```bash
pip install -e ".[test]"
python -m pytest
```

### Linting and Type Checking
```bash
python -m ruff check src/ plugins/
python -m ruff format src/ plugins/
python -m pyright src/apps plugins/
```

### Plugin Management
List available plugins:
```bash
goal plugins list
```

Inspect plugin configuration:
```bash
goal plugins inspect <plugin-name>
```

## Architecture Overview

### Core Structure
- **`src/apps/`**: Main application code
  - `cli/`: Command-line interface and configuration management
  - `interface/`: Abstract interfaces for datasets, models, analysis
  - `runtime/`: Logging, handlers, and runtime utilities
- **`plugins/`**: Extensible plugin system
  - `datasets/`: Dataset implementations (MNIST, CIFAR-10, SVHN, etc.)
  - `models/`: Model implementations (HMOG - Hierarchical Mixture of Gaussians)
- **`config/`**: Hydra configuration system
  - `hydra/`: Configuration files for datasets and models

### Plugin System
The project uses a plugin-based architecture where:
- Datasets are registered as plugins (see `plugins/datasets/`)
- Models are registered as plugins (see `plugins/models/`)
- Configuration is managed through Hydra with YAML files in `config/hydra/`

### Key Design Patterns
- **Configuration-driven**: Uses Hydra for configuration management
- **Plugin-based**: Extensible through dataset and model plugins
- **Experiment tracking**: Integrates with Weights & Biases (wandb)
- **Reproducible runs**: Results saved to timestamped directories in `runs/`

## Dependencies

### Core Dependencies
- **goal-jax**: The geometric optimization library this project builds upon
- **JAX**: Automatic differentiation and computation backend
- **Hydra**: Configuration management
- **Weights & Biases**: Experiment tracking
- **Typer**: CLI framework

### Optional Dependencies
- **torchvision**: For dataset loading (`[datasets]` extra)
- **jax[cuda12]**: For GPU support (`[gpu]` extra)
- **pytest**: For testing (`[test]` extra)

## Working with goal-jax

When making changes that involve the core statistical models or geometric abstractions:

1. **Understand the coordinate systems**: goal-jax uses `Natural` and `Mean` coordinate systems
2. **Leverage the manifold hierarchy**: Models inherit from `ExponentialFamily` → `Manifold`
3. **Use the type system**: Extensive use of `Point[C, M]` where `C` is coordinate type and `M` is manifold
4. **Consider analytic vs differentiable**: Many operations have both `Analytic*` and `Differentiable*` variants

For more details on goal-jax architecture, see `../goal-jax/CLAUDE.md`.