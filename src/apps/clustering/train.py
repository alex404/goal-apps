"""Training functionality for clustering models."""

import hydra
from jax import Array
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.panel import Panel

from ..runtime import RunHandler
from ..util import config_tree


def train(key: Array, handler: RunHandler, cfg: DictConfig) -> None:
    """Train model and save results."""
    print(f"Run name: {cfg.run_name}")
    print(f"with JIT: {cfg.jit}")

    # Convert OmegaConf to a regular dictionary (resolve references)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create a tree
    tree = config_tree(config_dict)

    # Print it inside a panel for extra clarity
    print(Panel(tree, title="Hydra Config Overview", border_style="green"))
    dataset = hydra.utils.instantiate(cfg.dataset, cache_dir=handler.cache_dir)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model, data_dim=dataset.data_dim)

    # Train model
    results = model.evaluate(key, dataset)

    # Save results
    handler.save_analysis(results, "training_results")
