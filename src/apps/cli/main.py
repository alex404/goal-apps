### Imports ###

### Preamble ###
import logging
from typing import Any

import typer
from rich import print as rprint
from rich.table import Table

from .sweep import (
    clear_optuna_study,
    create_optuna_study,
    create_sweep_config,
    reset_optuna_study,
    run_optuna_trial,
    sample_sweep_args,
)
from .util import (
    format_config_table,
    get_store_groups,
    print_objective_sparkline,
    print_parameter_distributions,
    print_sweep_tree,
)

# Set up logging
log = logging.getLogger(__name__)


def _init_plugins() -> None:
    """Register plugins and heavy dependencies. Call only in commands that need them."""
    from plugins import register_plugins

    register_plugins()


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
    import jax

    _init_plugins()
    from .configs import ClusteringRunConfig
    from .initialize import initialize_run

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
    import jax

    _init_plugins()
    from .configs import ClusteringRunConfig
    from .initialize import initialize_run

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
        _init_plugins()
        from .configs import ClusteringRunConfig
        from .initialize import initialize_run

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
        import wandb

        wandb.sweep(sweep_config, project=project)


optuna_app = typer.Typer()
tune_app.add_typer(optuna_app, name="optuna", help="Optuna hyperparameter optimization.")


@optuna_app.command(name="create")
def optuna_create(
    overrides: list[str] = overrides,
    metric: str = typer.Option(..., "--metric", "-m", help="Metric to optimize"),
    study_name: str | None = typer.Option(
        None, "--study", "-s", help="Study name (defaults to experiment= override)"
    ),
    direction: str = typer.Option(
        "maximize", "--direction", "-d", help="maximize or minimize"
    ),
    pruner: str = typer.Option("median", "--pruner", help="Pruner type: median, none"),
    storage: str | None = typer.Option(
        None, "--storage", help="Optuna storage URL (default: sqlite in runs/tune/)"
    ),
):
    """Create an Optuna study with a search space.

    Example:
        goal tune optuna create --metric "Clustering/Test ARI" \\
            experiment=pbmc68k-search dataset=pbmc68k model=pbmc68k-hmog \\
            model.full.ent_reg=suggest_float:1e-1:3e0:log
    """
    if study_name is None:
        for o in overrides:
            if o.startswith("experiment="):
                study_name = o.split("=", 1)[1]
                break
    if study_name is None:
        rprint("[red]Must provide --study or experiment= override[/red]")
        raise typer.Exit(1)

    study = create_optuna_study(
        overrides=overrides,
        study_name=study_name,
        metric=metric,
        direction=direction,
        pruner_name=pruner,
        storage=storage,
    )
    rprint(f"Created study: {study.study_name}")
    rprint(f"Storage: {study._storage}")
    rprint(f"Run trials with: goal tune optuna run {study_name}")


@optuna_app.command(name="run")
def optuna_run(
    study_name: str = typer.Argument(help="Name of the study to run a trial for"),
    n_trials: int = typer.Option(1, "--n-trials", "-n", help="Trials this worker runs"),
):
    """Run trial(s) against an existing Optuna study.

    Example:
        goal tune optuna run pbmc68k-search
    """
    _init_plugins()
    study = run_optuna_trial(study_name=study_name, n_trials=n_trials)

    try:
        best = study.best_trial
        rprint(f"\nBest trial: {best.number}")
        rprint(f"Best value: {best.value}")
        rprint(f"Best params: {best.params}")
    except ValueError:
        rprint("\nNo completed trials yet.")


@optuna_app.command(name="status")
def optuna_status(
    study_name: str = typer.Argument(help="Name of the study to inspect"),
    top_n: int = typer.Option(10, "--top", "-t", help="Number of top trials to show"),
):
    """Show study status: trial counts, best results, top configurations."""
    import optuna

    from .sweep import _default_storage, _study_config_path

    config_path = _study_config_path(study_name)
    if not config_path.exists():
        rprint(f"[red]Study '{study_name}' not found[/red]")
        raise typer.Exit(1)

    storage = _default_storage(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.trials

    # Count by state
    from optuna.trial import TrialState

    completed = [t for t in trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == TrialState.PRUNED]
    failed = [t for t in trials if t.state == TrialState.FAIL]
    running = [t for t in trials if t.state == TrialState.RUNNING]

    direction = study.direction.name

    rprint(f"\n[bold]Study: {study_name}[/bold]  (direction: {direction.lower()})")
    rprint(f"Trials: {len(completed)} completed, {len(pruned)} pruned, {len(failed)} failed, {len(running)} running")

    if not completed:
        rprint("\nNo completed trials yet.")
        return

    # Best trial
    best = study.best_trial
    rprint(f"\n[bold]Best trial: t{best.number}[/bold] (value: {best.value:.6f})")
    for k, v in best.params.items():
        rprint(f"  {k}: {v}")

    # Top N trials
    ranked = sorted(completed, key=lambda t: t.value or 0, reverse=(direction == "MAXIMIZE"))
    rprint(f"\n[bold]Top {min(top_n, len(ranked))} trials:[/bold]")
    for t in ranked[:top_n]:
        rprint(f"  t{t.number}: {t.value:.6f}")

    # Visualizations
    print_objective_sparkline(completed, direction)
    print_parameter_distributions(completed, best.params)


@optuna_app.command(name="plot")
def optuna_plot(
    study_name: str = typer.Argument(help="Name of the study to plot"),
    pct: float = typer.Option(10.0, "--pct", "-p", help="Top/bottom percentile to highlight"),
    output: str = typer.Option("", "--output", "-o", help="Output path (default: <study>-params.png)"),
):
    """Plot parameter histograms: all trials vs top/bottom pct% by objective."""
    import optuna
    from pathlib import Path

    from .sweep import _default_storage, _study_config_path
    from .util import plot_param_histograms

    config_path = _study_config_path(study_name)
    if not config_path.exists():
        rprint(f"[red]Study '{study_name}' not found[/red]")
        raise typer.Exit(1)

    storage = _default_storage(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)

    from optuna.trial import TrialState
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if not completed:
        rprint("[red]No completed trials.[/red]")
        raise typer.Exit(1)

    direction = study.direction.name
    out_path = Path(output) if output else Path(f"{study_name}-params.png")
    saved = plot_param_histograms(completed, direction, pct=pct, output=out_path)
    rprint(f"Saved to [bold]{saved}[/bold]  ({len(completed)} trials, {direction.lower()})")


@optuna_app.command(name="reset")
def optuna_reset(
    study_name: str = typer.Argument(help="Name of the study to reset"),
):
    """Reset a study: delete the DB but keep config and run directories."""
    reset_optuna_study(study_name)
    rprint(f"Study '{study_name}' reset. Config retained, DB cleared.")


@optuna_app.command(name="clear")
def optuna_clear(
    study_name: str = typer.Argument(help="Name of the study to clear"),
):
    """Clear a study: delete the entire study directory."""
    clear_optuna_study(study_name)
    rprint(f"Study '{study_name}' cleared.")


# Plugins
plugin = typer.Argument(default=None, help="Name of plugin to inspect")


@plugins_com.command(name="list")
def list_plugins():
    """List all available plugins by group."""
    _init_plugins()
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
    _init_plugins()
    from hydra.core.config_store import ConfigNode, ConfigStore
    from omegaconf import OmegaConf

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
