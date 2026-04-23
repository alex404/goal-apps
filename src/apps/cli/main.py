### Imports ###

### Preamble ###
import logging
from typing import Any

import typer
from rich import print as rprint
from rich.table import Table

from .sweep import (
    OptunaImportError,
    OptunaValidationError,
    clear_optuna_study,
    create_optuna_study,
    create_sweep_config,
    default_storage,
    import_trials,
    merge_from_template,
    reset_optuna_study,
    run_optuna_trial,
    validate_optuna_config,
    validate_wandb_config,
)
from .util import (
    format_config_table,
    get_store_groups,
    print_objective_sparkline,
    print_param_distributions_split,
    print_param_pair_coverage,
    print_sweep_tree,
    print_trial_summary,
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

    handler, logger, dataset, model, seed = initialize_run(
        ClusteringRunConfig, overrides
    )
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

    handler, logger, dataset, model, seed = initialize_run(
        ClusteringRunConfig, overrides
    )
    key = jax.random.PRNGKey(seed)

    # Run analysis
    log.info("Beginning analysis...")
    model.analyze(key, handler, logger, dataset)
    log.info("Analysis complete.")


### Tune Commands ###

tune_app = typer.Typer()
main.add_typer(tune_app, name="tune", help="Hyperparameter tuning commands.")

tune_wandb_dry_run = typer.Option(
    False, "--dry-run", help="Print sweep config and exit"
)
tune_wandb_validate = typer.Option(
    True,
    "--validate/--no-validate",
    help="Probe one sampled configuration before launching the sweep",
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

    print_sweep_tree(sweep_config)

    if validate:
        _init_plugins()
        try:
            validate_wandb_config(sweep_config)
        except OptunaValidationError as e:
            rprint(f"[red]Validation failed:[/red]\n{e}")
            raise typer.Exit(1) from e

    if not dry_run:
        import wandb

        wandb.sweep(sweep_config, project=project)


optuna_app = typer.Typer()
tune_app.add_typer(
    optuna_app, name="optuna", help="Optuna hyperparameter optimization."
)


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
    template: str | None = typer.Option(
        None, "--template", "-t", help="Study name whose overrides serve as base values"
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Probe one sampled configuration and verify the metric is produced",
    ),
):
    """Create an Optuna study with a search space.

    Example:
        goal tune optuna create --metric "Clustering/Test ARI" \\
            experiment=pbmc68k-search dataset=pbmc68k model=pbmc68k-hmog \\
            model.full.ent_reg=suggest_float:1e-1:3e0:log
    """
    if template is not None:
        overrides = merge_from_template(template, overrides)

    if study_name is None:
        for o in overrides:
            if o.startswith("experiment="):
                study_name = o.split("=", 1)[1]
                break
    if study_name is None:
        rprint("[red]Must provide --study or experiment= override[/red]")
        raise typer.Exit(1)

    if validate:
        _init_plugins()
        try:
            validate_optuna_config(overrides, metric, study_name)
        except OptunaValidationError as e:
            rprint(f"[red]Validation failed:[/red]\n{e}")
            raise typer.Exit(1) from e

    study = create_optuna_study(
        overrides=overrides,
        study_name=study_name,
        metric=metric,
        direction=direction,
        pruner_name=pruner,
        storage=storage,
    )
    rprint(f"Created study: {study.study_name}")
    rprint(f"Storage: {storage or default_storage(study_name)}")
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
    pct: float = typer.Option(
        10.0, "--pct", "-p", help="Top/bottom percentile to highlight in distributions"
    ),
    n_pairs: int = typer.Option(
        4,
        "--pairs",
        "-n",
        help="Number of top sensitive params for pairwise grids (0 to skip)",
    ),
):
    """Show study status: trial counts, best results, top configurations."""
    import optuna

    from .sweep import study_config_path

    config_path = study_config_path(study_name)
    if not config_path.exists():
        rprint(f"[red]Study '{study_name}' not found[/red]")
        raise typer.Exit(1)

    from omegaconf import OmegaConf

    study_config = OmegaConf.load(config_path)
    metric: str = study_config.metric

    storage = default_storage(study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.trials

    # Count by state
    from optuna.trial import TrialState

    completed = [t for t in trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == TrialState.PRUNED]
    failed = [t for t in trials if t.state == TrialState.FAIL]
    running = [t for t in trials if t.state == TrialState.RUNNING]
    # Early-stopped: pruned by MedianPruner (has performance signal).
    # Diverged: pruned due to instability/missing metric (no intermediate values).
    early_stopped = [t for t in pruned if t.intermediate_values]
    diverged = [t for t in pruned if not t.intermediate_values]
    all_sampled = completed + pruned
    converged = completed + early_stopped

    direction = study.direction.name

    rprint(f"\n[bold]Study: {study_name}[/bold]")
    rprint(f"Metric: {metric}  (direction: {direction.lower()})")
    rprint(
        f"Trials: {len(completed)} completed, {len(early_stopped)} early-stopped, {len(diverged)} diverged, {len(failed)} failed, {len(running)} running"
    )

    if not completed:
        rprint("\nNo completed trials yet.")
        return

    # Best trial, human-rounded params, and top N
    print_trial_summary(completed, direction, pct, top_n)

    # Visualizations
    print_objective_sparkline(completed, direction)
    print_param_distributions_split(
        completed, direction, pct=pct, all_trials=all_sampled
    )
    if n_pairs > 0:
        print_param_pair_coverage(
            completed,
            direction,
            pct=pct,
            n_params=n_pairs,
            all_trials=all_sampled,
            converged_trials=converged,
        )


@optuna_app.command(name="import")
def optuna_import(
    source: str = typer.Argument(help="Name of the source study to import trials from"),
    target: str = typer.Argument(help="Name of the target study (must already exist)"),
    include_pruned: bool = typer.Option(
        False,
        "--include-pruned",
        help="Also import PRUNED trials (param coverage only, no value)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow importing into a target that already has trials",
    ),
):
    """Warm-start a target study from completed trials of a source study.

    Re-extracts the target study's metric from each source trial's
    ``metrics.joblib`` — useful when switching the optimization objective
    without re-running the search. Source study is not mutated.

    Example:
        goal tune optuna import ng20-hmog-coa-nmi ng20-hmog-flat-nmi
    """
    try:
        summary = import_trials(
            source=source, target=target, include_pruned=include_pruned, force=force
        )
    except (FileNotFoundError, OptunaImportError) as e:
        rprint(f"[red]Import failed:[/red]\n{e}")
        raise typer.Exit(1) from e

    rprint(f"\n[bold]Search-space diff[/bold] ({source} → {target})")
    if summary.aligned:
        aligned_list = ", ".join(summary.aligned)
        rprint(f"  [green]aligned[/green] ({len(summary.aligned)}): {aligned_list}")
    if summary.target_only:
        target_list = ", ".join(summary.target_only)
        rprint(
            f"  [yellow]target-only[/yellow] ({len(summary.target_only)}): {target_list}  [dim](no warm-start)[/dim]"
        )
    if summary.source_only:
        source_list = ", ".join(summary.source_only)
        rprint(
            f"  [dim]source-only ({len(summary.source_only)}): {source_list}  (ignored by target sampler)[/dim]"
        )

    rprint(
        f"\n[bold]Imported[/bold]: {summary.imported_complete} completed, {summary.imported_pruned} pruned, {summary.skipped_missing_metric} skipped (missing metric)"
    )


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
