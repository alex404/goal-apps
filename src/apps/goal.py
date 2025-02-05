### Imports ###

import os
from pathlib import Path
from typing import Any

import jax
import typer
import wandb
from hydra.core.config_store import ConfigNode, ConfigStore
from omegaconf import OmegaConf
from plugins import register_plugins
from rich import print as rprint
from rich.table import Table

from .configs import ClusteringRunConfig
from .runtime.initialize import initialize_run
from .util import (
    create_sweep_config,
    format_config_table,
    get_store_groups,
    print_sweep_tree,
)

### Preamable ###

package_root = Path(__file__).parents[3]

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


### Commands ###


# Clustering
overrides = typer.Argument(
    default=None, help="Configuration overrides (e.g., dataset=mnist model=hmog)"
)

train_dry_run = typer.Option(False, "--dry-run", help="Print hydra config and exit")


@clustering_com.command()
def train(overrides: list[str] = overrides, dry_run: bool = train_dry_run):
    """Train a clustering model.

    Train a model using the specified dataset and model configuration.
    Results are saved to the runs directory for later analysis.

    Example:
        goal clustering train run_name=my_exp dataset=mnist model=hmog
    """

    handler, dataset, model, logger = initialize_run(ClusteringRunConfig, overrides)
    key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), byteorder="big"))
    if dry_run:
        return

    from apps.clustering.cli import train

    train(key, handler, dataset, model, logger)


sweep_dry_run = typer.Option(False, "--dry-run", help="Print sweep config and exit")
project = typer.Option("goal", "--project", "-p", help="W&B project name")


@clustering_com.command()
def sweep(
    overrides: list[str] = overrides,
    project: str = project,
    dry_run: bool = sweep_dry_run,
):
    """Launch a wandb hyperparameter sweep.

    Example:
        goal clustering sweep latent_dim=[4,8,12,16,20] n_clusters=[4,8,16]
    """
    sweep_config = create_sweep_config(overrides)

    # Create config tree visualization
    print_sweep_tree(sweep_config)

    if not dry_run:
        wandb.sweep(sweep_config, project=project)


@clustering_com.command()
def analyze(overrides: list[str] = overrides):
    """Analyze training results.

    Generate visualizations and compute metrics for a trained model.
    Results are saved to the runs directory.

    Example:
        goal clustering analyze run_name=my_exp
    """
    from apps.clustering.cli import analyze

    handler, _, _, _ = initialize_run(ClusteringRunConfig, overrides)

    analyze(handler)


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

    cs = ConfigStore.instance()

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
