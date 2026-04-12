### Imports ###

### Preamble ###
import logging
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
from .initialize import initialize_run
from .sweep import (
    create_sweep_config,
    run_optuna_study,
    sample_sweep_args,
)
from .util import (
    format_config_table,
    get_store_groups,
    print_sweep_tree,
)

# Very first step: register all plugins
register_plugins()

# Set up logging
log = logging.getLogger(__name__)


### CLI ###

# CLI configuration
main = typer.Typer(
    help="""CLI for statistical modelling apps built on GOAL (Geometric OptimizAtion Libraries)."""
)
plugins_com = typer.Typer()
main.add_typer(plugins_com, name="plugins", help="Commands for plugin management.")


### Commands ###


# Clustering
overrides = typer.Argument(
    default=None, help="Configuration overrides (e.g., dataset=mnist model=hmog)"
)

train_dry_run = typer.Option(False, "--dry-run", help="Print hydra config and exit")


@main.command()
def train(overrides: list[str] = overrides, dry_run: bool = train_dry_run):
    """Train a model on a dataset.

    Train a model using the specified dataset and model configuration.
    Results are saved to the runs directory for later analysis.

    Example:
        goal clustering train run_name=my_exp dataset=mnist model=hmog
    """

    handler, logger, dataset, model, seed = initialize_run(ClusteringRunConfig, overrides)
    key = jax.random.PRNGKey(seed)
    if dry_run:
        return

    # Train model
    log.info("Beginning training...")
    model.train(key, handler, logger, dataset)

    log.info("Training complete.")
    logger.finalize(handler)
    log.info("Logging complete, exiting.")


@main.command()
def analyze(overrides: list[str] = overrides):
    """Analyze training results.

    Generate visualizations and compute metrics for a trained model.
    Results are saved to the runs directory.

    Example:
        goal clustering analyze run_name=my_exp
    """
    handler, logger, dataset, model, seed = initialize_run(ClusteringRunConfig, overrides)
    key = jax.random.PRNGKey(seed)

    # Run analysis
    log.info("Beginning analysis...")
    model.analyze(key, handler, logger, dataset)
    log.info("Analysis complete.")


### Tune Commands ###

tune_app = typer.Typer()
main.add_typer(tune_app, name="tune", help="Hyperparameter tuning commands.")

tune_wandb_dry_run = typer.Option(False, "--dry-run", help="Print sweep config and exit")
tune_wandb_validate = typer.Option(
    False, "--validate", help="Validate an example config from the sweep"
)
tune_wandb_project = typer.Option("goal", "--project", "-p", help="W&B project name")


@tune_app.command(name="wandb")
def tune_wandb(
    overrides: list[str] = overrides,
    project: str = tune_wandb_project,
    dry_run: bool = tune_wandb_dry_run,
    validate: bool = tune_wandb_validate,
):
    """Launch a wandb hyperparameter sweep.

    Example:
        goal tune wandb dataset=mnist model=hmog latent_dim=4,8,12,16 n_clusters=50,100,200
    """
    sweep_config = create_sweep_config(overrides)

    # Create config tree visualization
    print_sweep_tree(sweep_config)

    if validate:
        dry_run = True
        try:
            # Sample one configuration
            sample_args = sample_sweep_args(sweep_config)

            log.info("Validating a sample configuration from the sweep...")
            log.info(f"Sample configuration: {' '.join(sample_args)}")

            # Initialize without actually running the training
            initialize_run(ClusteringRunConfig, sample_args)

            log.info("Sample configuration is valid!")
        except Exception:
            logging.exception("Validation failed:")

    if not dry_run:
        wandb.sweep(sweep_config, project=project)


tune_optuna_metric = typer.Option(..., "--metric", "-m", help="Metric to optimize")
tune_optuna_study = typer.Option(
    None, "--study", "-s", help="Study name (defaults to experiment= override)"
)
tune_optuna_storage = typer.Option(
    None, "--storage", help="Optuna storage URL (default: sqlite in runs/tune/)"
)
tune_optuna_n_trials = typer.Option(50, "--n-trials", "-n", help="Number of trials")
tune_optuna_direction = typer.Option(
    "maximize", "--direction", "-d", help="maximize or minimize"
)
tune_optuna_pruner = typer.Option(
    "median", "--pruner", help="Pruner type: median, none"
)
tune_optuna_dry_run = typer.Option(False, "--dry-run", help="Print config and exit")


@tune_app.command(name="optuna")
def tune_optuna(
    overrides: list[str] = overrides,
    metric: str = tune_optuna_metric,
    study_name: str | None = tune_optuna_study,
    storage: str | None = tune_optuna_storage,
    n_trials: int = tune_optuna_n_trials,
    direction: str = tune_optuna_direction,
    pruner: str = tune_optuna_pruner,
    dry_run: bool = tune_optuna_dry_run,
):
    """Run Optuna hyperparameter optimization.

    Example:
        goal tune optuna --metric "Merging/KL Train ARI" \\
            experiment=pbmc68k-v1 dataset=pbmc68k model=pbmc68k-hmog \\
            model.full.ent_reg=suggest_float:1e-1:2e0:log
    """
    # Resolve study name: --study flag takes precedence, else experiment= override
    if study_name is None:
        for o in overrides:
            if o.startswith("experiment="):
                study_name = o.split("=", 1)[1]
                break
    if study_name is None:
        rprint("[red]Must provide --study or experiment= override[/red]")
        raise typer.Exit(1)

    if dry_run:
        rprint(f"Study: {study_name}")
        rprint(f"Storage: {storage or f'sqlite:///runs/tune/{study_name}/study.db'}")
        rprint(f"Direction: {direction}")
        rprint(f"Metric: {metric}")
        rprint(f"Pruner: {pruner}")
        rprint(f"Trials: {n_trials}")
        rprint(f"Overrides: {overrides}")
        return

    study = run_optuna_study(
        overrides=overrides,
        study_name=study_name,
        metric=metric,
        storage=storage,
        n_trials=n_trials,
        direction=direction,
        pruner_name=pruner,
    )

    rprint(f"\nBest trial: {study.best_trial.number}")
    rprint(f"Best value: {study.best_trial.value}")
    rprint(f"Best params: {study.best_trial.params}")


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
            for name, config_node in group.items():
                clean_name: str = name.replace(".yaml", "")
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
