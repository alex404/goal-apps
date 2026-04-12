### Imports ###

import logging
from pathlib import Path
from typing import Any

import jax

log = logging.getLogger(__name__)

### Sweep Management ###


def _parse_args_to_parameters(args: list[str]) -> dict[str, Any]:
    """Parse CLI arguments into a wandb parameters dictionary."""
    parameters: dict[str, Any] = {}

    for arg in args:
        if "=" not in arg:
            continue

        param, value = arg.split("=", 1)
        if "," in value:
            try:
                # Try to parse as list of numbers first
                values = [float(x) if "." in x else int(x) for x in value.split(",")]
                parameters[param] = {"values": values}
            except ValueError:
                # If number parsing fails, treat as list of strings
                values = [x.strip() for x in value.split(",")]
                if len(values) > 1:
                    parameters[param] = {"values": values}
                else:
                    parameters[param] = {"value": value}
        else:
            # Single value
            parameters[param] = {"value": value}

    return parameters


def create_sweep_config(overrides: list[str]) -> dict[str, Any]:
    """Create wandb sweep config from override strings."""
    # Parse all overrides into a single dictionary
    parameters = {}

    # Then parse CLI overrides (these will overwrite any overlapping parameters)
    cli_parameters = _parse_args_to_parameters(overrides)
    parameters.update(cli_parameters)

    return {
        "program": "${program}",
        "method": "grid",
        "parameters": parameters,
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            "apps.cli.main",
            "train",
            "${args_no_hyphens}",
        ],
    }


def sample_sweep_args(sweep_config: dict[str, Any]) -> list[str]:
    """Sample command line arguments from a wandb sweep config."""
    args: list[str] = []

    # Extract parameters
    parameters = sweep_config.get("parameters", {})

    # Sample one value from each parameter
    for param_name, param_config in parameters.items():
        if param_name == "use_wandb":
            # Don't use wandb for validation
            continue

        if "values" in param_config:
            # If it's a list of values, take the first one
            args.append(f"{param_name}={param_config['values'][0]}")
        elif "value" in param_config:
            # If it's a single value, use it
            args.append(f"{param_name}={param_config['value']}")
        elif "min" in param_config and "max" in param_config:
            # If it's a range, take the min value
            args.append(f"{param_name}={param_config['min']}")

    return args


### Optuna ###


def resolve_optuna_overrides(trial: Any, overrides: list[str]) -> list[str]:
    """Resolve overrides containing suggest_* directives into concrete values.

    Syntax:
        param=suggest_float:low:high         -> trial.suggest_float(param, low, high)
        param=suggest_float:low:high:log     -> trial.suggest_float(param, low, high, log=True)
        param=suggest_int:low:high           -> trial.suggest_int(param, low, high)
        param=suggest_categorical:a,b,c      -> trial.suggest_categorical(param, [a, b, c])
        param=value                          -> passed through unchanged

    Returns:
        List of resolved override strings.
    """
    resolved = []
    for override in overrides:
        if "=" not in override:
            resolved.append(override)
            continue

        param, value = override.split("=", 1)

        if value.startswith("suggest_float:"):
            parts = value.split(":")[1:]
            low, high = float(parts[0]), float(parts[1])
            use_log = len(parts) > 2 and parts[2] == "log"
            sampled = trial.suggest_float(param, low, high, log=use_log)
            resolved.append(f"{param}={sampled}")

        elif value.startswith("suggest_int:"):
            parts = value.split(":")[1:]
            low, high = int(parts[0]), int(parts[1])
            sampled = trial.suggest_int(param, low, high)
            resolved.append(f"{param}={sampled}")

        elif value.startswith("suggest_categorical:"):
            choices_str = value.split(":", 1)[1]
            raw_choices = [c.strip() for c in choices_str.split(",")]
            choices: list[Any] = []
            for c in raw_choices:
                try:
                    choices.append(float(c) if "." in c else int(c))
                except ValueError:
                    choices.append(c)
            sampled = trial.suggest_categorical(param, choices)
            resolved.append(f"{param}={sampled}")

        else:
            resolved.append(override)

    return resolved


def run_optuna_study(
    overrides: list[str],
    study_name: str,
    metric: str,
    storage: str | None,
    n_trials: int,
    direction: str,
    pruner_name: str,
) -> Any:
    """Create and run an Optuna study.

    Args:
        overrides: CLI overrides (may contain suggest_* directives)
        study_name: Name for the Optuna study
        metric: Metric key to optimize (e.g. "Merging/KL Train ARI")
        storage: SQLite URL or None for default (runs/tune/{study_name}/study.db)
        n_trials: Number of trials to run
        direction: "maximize" or "minimize"
        pruner_name: Pruner type: "median" or "none"

    Returns:
        The completed Optuna study.
    """
    import optuna

    from .configs import ClusteringRunConfig
    from .initialize import initialize_run

    # Default storage: runs/tune/{study_name}/study.db
    if storage is None:
        study_dir = Path("runs/tune") / study_name
        study_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{study_dir.resolve()}/study.db"

    # Create pruner
    pruners: dict[str, optuna.pruners.BasePruner] = {
        "median": optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        "none": optuna.pruners.NopPruner(),
    }
    if pruner_name not in pruners:
        msg = f"Unknown pruner: {pruner_name}. Choose from: {list(pruners.keys())}"
        raise ValueError(msg)
    pruner = pruners[pruner_name]

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_overrides = resolve_optuna_overrides(trial, overrides)
        trial_overrides.extend([
            "use_optuna=true",
            f"optuna_metric={metric}",
            f"experiment={study_name}",
            f"run_name=t{trial.number}",
            f"sweep_id={study_name}",
        ])

        handler, logger, dataset, model, seed = initialize_run(
            ClusteringRunConfig, trial_overrides, trial=trial
        )
        key = jax.random.PRNGKey(seed)

        try:
            model.train(key, handler, logger, dataset)
            logger.finalize(handler)
        except optuna.TrialPruned:
            logger.finalize(handler)
            raise

        # Buffer is cleared after finalize(), so read final metric from disk
        metrics = handler.load_metrics()
        if metric not in metrics or not metrics[metric]:
            raise optuna.TrialPruned()

        _, final_value = metrics[metric][-1]
        return final_value

    study.optimize(objective, n_trials=n_trials)
    return study
