"""Training functionality for clustering models."""

from typing import Any

import hydra
from jax import Array
from omegaconf import DictConfig, OmegaConf
from rich import print as rprint

from ..experiments import ExperimentHandler, initialize_jax
from ..util import format_config_table


def train(key: Array, cfg: DictConfig) -> None:
    """Train model and save results."""
    print(f"Running experiment: {cfg.experiment}")
    print(f"with JIT: {cfg.jit}")

    # Initialize environment
    initialize_jax(device=cfg.device, disable_jit=not cfg.jit)
    paths = ExperimentHandler(cfg.experiment)

    # Get model config and update with runtime values
    model_params = OmegaConf.to_container(cfg.model)

    # Cast to correct type
    params: dict[str, Any] = model_params  # pyright: ignore[reportAssignmentType]

    # Instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset, cache_dir=paths.cache_dir)

    # Update config with runtime value
    params["data_dim"] = dataset.data_dim

    # Show configuration
    target, table = format_config_table("Model", params)
    if target:
        rprint(f"\nImplementation: [blue]{target}[/blue]\n")
    rprint(table)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, data_dim=dataset.data_dim)

    # Train model
    results = model.evaluate(key, dataset)

    # Save results
    paths.save_analysis(results, "training_results")
