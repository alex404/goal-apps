### Imports ###

import os
from pathlib import Path
from typing import Any

import jax
import typer
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigNode, ConfigStore
from omegaconf import DictConfig, OmegaConf
from plugins import register_plugins
from rich import print as rprint
from rich.table import Table

from .clustering.core.config import ClusteringConfig
from .experiments import ExperimentConfig, ExperimentHandler
from .util import format_config_table, get_store_groups

### Preamable ###

cs = ConfigStore.instance()

package_root = Path(__file__).parents[2]

register_plugins()

# CLI configuration
main = typer.Typer(
    help="""CLI for statistical modelling apps built on GOAL (Geometric OptimizAtion Libraries)."""
)
clustering_com = typer.Typer()
main.add_typer(
    clustering_com, name="clustering", help="Command for clustering applications."
)

plugins_com = typer.Typer()
main.add_typer(plugins_com, name="plugins", help="Commands for plugin management.")


### Helper functions ###


def compose_config(
    exp_conf: type[ExperimentConfig], overrides: list[str]
) -> DictConfig:
    """Compose configuration using Hydra."""
    # Register base config
    cs.store(name="config_schema", node=exp_conf)

    with initialize_config_dir(
        version_base="1.3", config_dir=str(package_root / "config")
    ):
        return compose(config_name="config", overrides=overrides)


### Commands ###


# Clustering
overrides = typer.Argument(
    default=None, help="Configuration overrides (e.g., dataset=mnist model=hmog)"
)


@clustering_com.command()
def train(overrides: list[str] = overrides):
    """Train a clustering model.

    Train a model using the specified dataset and model configuration.
    Results are saved to the experiment directory for later analysis.

    Example:
        goal clustering train experiment=my_exp dataset=mnist model=hmog
    """

    cfg = compose_config(ClusteringConfig, overrides)
    key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), byteorder="big"))

    from apps.clustering.train import train

    train(key, cfg)


@clustering_com.command()
def analyze(overrides: list[str] = overrides):
    """Analyze training results.

    Generate visualizations and compute metrics for a trained model.
    Results are saved to the experiment directory.

    Example:
        goal clustering analyze experiment=my_exp
    """
    from apps.clustering.analyze import analyze

    cfg = compose_config(ClusteringConfig, overrides)

    analyze(ExperimentHandler(cfg.experiment))


# Plugins
plugin = typer.Argument(default=None, help="Name of plugin to inspect")


@plugins_com.command(name="list")
def list_plugins():
    """List all available plugins by group."""
    groups = get_store_groups()
    table = Table(title="Available Plugins")
    table.add_column("Type", style="cyan")
    table.add_column("Plugin", style="green")

    for group in ["dataset", "model"]:
        if group in groups:
            items = groups[group]
            table.add_row(group.title(), ", ".join(sorted(items)))

    rprint(table)


@plugins_com.command()
def inspect(plugin: str = plugin):
    """Inspect plugin configuration parameters."""

    for group_name in ["model", "dataset"]:
        group = cs.repo.get(group_name, {})
        if isinstance(group, dict):
            for name, config_node in group.items():  # pyright: ignore[reportUnknownVariableType]
                clean_name: str = name.replace(".yaml", "")  # pyright: ignore[reportUnknownVariableType]
                if clean_name == plugin and isinstance(config_node, ConfigNode):
                    params: dict[str, Any] = OmegaConf.to_container(config_node.node)  # pyright: ignore[reportAssignmentType]
                    if not params:
                        continue

                    target, table = format_config_table(clean_name, params)
                    if target:
                        rprint(f"\nImplementation: [blue]{target}[/blue]\n")
                    rprint(table)
                    return

    rprint(f"[red]Plugin '{plugin}' not found[/red]")


### Main ###

if __name__ == "__main__":
    main()
