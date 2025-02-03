"""Training functionality for clustering models."""

import hydra
from jax import Array
from omegaconf import DictConfig
from rich import print

from ..runtime import RunHandler


def train(key: Array, handler: RunHandler, cfg: DictConfig) -> None:
    """train model and save results."""
    print(f"Run name: {cfg.run_name}")
    print(f"with JIT: {cfg.jit}")

    dataset = hydra.utils.instantiate(cfg.dataset, cache_dir=handler.cache_dir)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, data_dim=dataset.data_dim)

    # Train model
    results = model.evaluate(key, dataset)

    # Save results
    handler.save_analysis(results, "training_results")
