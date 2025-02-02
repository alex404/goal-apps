"""Training functionality for clustering models."""

import hydra
import jax
from omegaconf import DictConfig

from ..experiments import ExperimentHandler, initialize_jax


def train(cfg: DictConfig) -> None:
    """Train model and save results.

    Args:
        cfg: Configuration containing model, dataset, and training parameters
    """
    print(f"Running experiment: {cfg.experiment}")
    print(f"with JIT: {cfg.jit}")

    # Initialize environment
    initialize_jax(device=cfg.device, disable_jit=not cfg.jit)
    paths = ExperimentHandler(cfg.experiment)
    key = jax.random.PRNGKey(0)

    # Instantiate dataset and model using Hydra
    dataset = hydra.utils.instantiate(cfg.dataset, cache_dir=paths.cache_dir)
    model = hydra.utils.instantiate(cfg.model, data_dim=dataset.data_dim)

    # Train model
    results = model.evaluate(key, dataset)

    # Save results
    paths.save_analysis(results, "training_results")
