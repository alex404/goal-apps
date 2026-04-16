### Imports ###

import difflib
import logging
import shutil
from pathlib import Path
from typing import Any

import jax
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

TUNE_DIR = Path("runs/tune")


class OptunaValidationError(Exception):
    """Raised when validation of a proposed Optuna study configuration fails."""


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


def _parse_categorical_choices(choices_str: str) -> list[Any]:
    """Parse a suggest_categorical value list into typed Python values."""
    raw_choices = [c.strip() for c in choices_str.split(",")]
    choices: list[Any] = []
    for c in raw_choices:
        try:
            choices.append(float(c) if "." in c else int(c))
        except ValueError:
            choices.append(c)
    return choices


def _resolve_directive(name: str, value: str, sampler: Any) -> Any:
    """Resolve a single suggest_* directive via ``sampler`` (an object with
    ``suggest_float``, ``suggest_int``, ``suggest_categorical`` methods — e.g.
    an ``optuna.trial.Trial``). Returns the sampled value, or ``None`` if
    ``value`` isn't a directive.
    """
    if value.startswith("suggest_float:"):
        parts = value.split(":")[1:]
        low, high = float(parts[0]), float(parts[1])
        use_log = len(parts) > 2 and parts[2] == "log"
        return sampler.suggest_float(name, low, high, log=use_log)
    if value.startswith("suggest_int:"):
        parts = value.split(":")[1:]
        low, high = int(parts[0]), int(parts[1])
        return sampler.suggest_int(name, low, high)
    if value.startswith("suggest_categorical:"):
        choices_str = value.split(":", 1)[1]
        return sampler.suggest_categorical(
            name, _parse_categorical_choices(choices_str)
        )
    return None


class _FirstValueSampler:
    """Sampler that returns the low/first value for each directive.

    Mirrors the ``sample_sweep_args`` behaviour for wandb: pick a single
    deterministic point from the search space so the config can be
    instantiated without invoking optuna.
    """

    def suggest_float(
        self, _param: str, low: float, _high: float, log: bool = False
    ) -> float:
        del log
        return low

    def suggest_int(self, _param: str, low: int, _high: int) -> int:
        return low

    def suggest_categorical(self, _param: str, choices: list[Any]) -> Any:
        return choices[0]


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
        sampled = _resolve_directive(param, value, trial)
        if sampled is None:
            resolved.append(override)
        else:
            resolved.append(f"{param}={sampled}")
    return resolved


def sample_optuna_overrides(overrides: list[str]) -> list[str]:
    """Resolve suggest_* directives by picking a deterministic sample point.

    Used for validation — no ``optuna.Trial`` required.
    """
    sampler = _FirstValueSampler()
    resolved = []
    for override in overrides:
        if "=" not in override:
            resolved.append(override)
            continue
        param, value = override.split("=", 1)
        sampled = _resolve_directive(param, value, sampler)
        if sampled is None:
            resolved.append(override)
        else:
            resolved.append(f"{param}={sampled}")
    return resolved


def validate_wandb_config(sweep_config: dict[str, Any]) -> None:
    """Validate a proposed wandb sweep by instantiating one sampled point.

    Raises ``OptunaValidationError`` if ``initialize_run`` fails to instantiate
    model + dataset. (Name is shared with the optuna validator for symmetry —
    both are "config validation" errors.)

    The probe run directory is cleaned up on exit.
    """
    from .configs import ClusteringRunConfig
    from .initialize import initialize_run

    sample_args = sample_sweep_args(sweep_config)
    probe_run_name = "__validate_wandb__"
    probe_overrides = [
        *sample_args,
        f"run_name={probe_run_name}",
        "use_wandb=false",
        "use_optuna=false",
        "device=cpu",
    ]
    probe_run_dir = Path("runs") / "single" / probe_run_name

    try:
        try:
            initialize_run(ClusteringRunConfig, probe_overrides)
        except Exception as e:
            raise OptunaValidationError(
                f"Sample configuration failed to initialize: {e}"
            ) from e
        log.info("Validation passed: sample configuration initialized cleanly.")
    finally:
        if probe_run_dir.exists():
            shutil.rmtree(probe_run_dir, ignore_errors=True)


def validate_optuna_config(overrides: list[str], metric: str, study_name: str) -> None:
    """Validate a proposed Optuna study config by probing one sampled point.

    Raises ``OptunaValidationError`` if:
    - ``initialize_run`` fails to instantiate model + dataset, or
    - ``metric`` is not in the model's declared ``metric_names`` for the
      instantiated dataset.

    The probe run directory is cleaned up on exit.
    """
    from .configs import ClusteringRunConfig
    from .initialize import initialize_run

    probe_run_name = f"__validate_{study_name}__"
    probe_overrides = [
        *sample_optuna_overrides(overrides),
        f"run_name={probe_run_name}",
        "use_wandb=false",
        "use_optuna=false",
        "device=cpu",
    ]
    # Compute the probe dir upfront so we can clean it up even if
    # initialize_run raises before returning a handler.
    probe_run_dir = Path("runs") / "single" / probe_run_name

    try:
        _, _, dataset, model, _ = initialize_run(ClusteringRunConfig, probe_overrides)

        declared = model.metric_names(dataset)
        if metric not in declared:
            close = difflib.get_close_matches(metric, sorted(declared), n=5, cutoff=0.4)
            lines = [
                f"Metric {metric!r} is not produced by this model+dataset configuration.",
            ]
            if close:
                lines.append("Did you mean one of:")
                lines.extend(f"  - {c}" for c in close)
            lines.append(
                "Run with --no-validate to skip this check, or enable the analysis that produces the metric."
            )
            raise OptunaValidationError("\n".join(lines))

        log.info(f"Validation passed: metric {metric!r} is declared by the model.")
    finally:
        if probe_run_dir.exists():
            shutil.rmtree(probe_run_dir, ignore_errors=True)


def merge_from_template(template_name: str, new_overrides: list[str]) -> list[str]:
    """Merge new overrides on top of an existing study's overrides."""
    config_path = study_config_path(template_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Template study not found: {config_path}")
    template_config = OmegaConf.load(config_path)
    base: dict[str, str] = {}
    for override in template_config.overrides:
        key, _, val = str(override).partition("=")
        base[key] = val
    for override in new_overrides:
        key, _, val = override.partition("=")
        base[key] = val
    return [f"{k}={v}" for k, v in base.items()]


def _study_dir(study_name: str) -> Path:
    return TUNE_DIR / study_name


def study_config_path(study_name: str) -> Path:
    return _study_dir(study_name) / "study-config.yaml"


def default_storage(study_name: str) -> str:
    return f"sqlite:///{_study_dir(study_name).resolve()}/study.db"


def _make_pruner(pruner_name: str) -> Any:
    import optuna

    pruners: dict[str, optuna.pruners.BasePruner] = {
        "median": optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        "none": optuna.pruners.NopPruner(),
    }
    if pruner_name not in pruners:
        msg = f"Unknown pruner: {pruner_name}. Choose from: {list(pruners.keys())}"
        raise ValueError(msg)
    return pruners[pruner_name]


def create_optuna_study(
    overrides: list[str],
    study_name: str,
    metric: str,
    direction: str,
    pruner_name: str,
    storage: str | None = None,
) -> Any:
    """Create a new Optuna study and save its config for workers.

    Returns:
        The created Optuna study.
    """
    import optuna

    study_dir = _study_dir(study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    # Save study config so workers can join with just the study name
    config = OmegaConf.create(
        {
            "overrides": overrides,
            "metric": metric,
            "direction": direction,
            "pruner": pruner_name,
            "storage": storage,
        }
    )
    config_path = study_config_path(study_name)
    OmegaConf.save(config, config_path)
    log.info(f"Study config saved to {config_path}")

    resolved_storage = storage or default_storage(study_name)
    return optuna.create_study(
        study_name=study_name,
        storage=resolved_storage,
        direction=direction,
        pruner=_make_pruner(pruner_name),
        load_if_exists=True,
    )


def run_optuna_trial(
    study_name: str,
    n_trials: int = 1,
) -> Any:
    """Run trial(s) against an existing Optuna study.

    Loads the study config from disk, connects to the study DB,
    and runs the specified number of trials.

    Returns:
        The Optuna study.
    """
    import optuna

    from apps.runtime import DivergentTrainingError

    from .configs import ClusteringRunConfig
    from .initialize import initialize_run

    # Load study config
    config_path = study_config_path(study_name)
    if not config_path.exists():
        msg = f"Study config not found at {config_path}. Run 'goal tune optuna create' first."
        raise FileNotFoundError(msg)

    config = OmegaConf.load(config_path)
    overrides: list[str] = list(config.overrides)
    metric: str = config.metric
    pruner_name: str = config.pruner
    storage: str = config.storage or default_storage(study_name)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        pruner=_make_pruner(pruner_name),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        trial_overrides = resolve_optuna_overrides(trial, overrides)
        trial_overrides.extend(
            [
                "use_optuna=true",
                f"optuna_metric={metric}",
                f"experiment={study_name}",
                f"run_name=t{trial.number}",
                f"sweep_id={study_name}",
            ]
        )

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
        except DivergentTrainingError as e:
            # Raised directly from analyses (non-finite distance matrices, etc.)
            log.warning(f"Trial {trial.number} diverged: {e}")
            logger.finalize(handler)
            raise optuna.TrialPruned() from e
        except Exception as e:
            # DivergentTrainingError from JAX callbacks gets wrapped in
            # XlaRuntimeError, losing the original type.
            if isinstance(
                e.__cause__, DivergentTrainingError
            ) or "DivergentTrainingError" in str(e):
                log.warning(f"Trial {trial.number} diverged: {e}")
                logger.finalize(handler)
                raise optuna.TrialPruned() from e
            raise

        # Buffer is cleared after finalize(), so read final metric from disk
        metrics = handler.load_metrics()
        if metric not in metrics or not metrics[metric]:
            raise optuna.TrialPruned()

        _, final_value = metrics[metric][-1]
        return final_value

    study.optimize(objective, n_trials=n_trials)
    return study


def reset_optuna_study(study_name: str) -> None:
    """Delete the study DB and all run directories, then recreate from saved config."""
    import optuna

    study_dir = _study_dir(study_name)
    config_path = study_config_path(study_name)

    # Delete everything except the config
    if study_dir.exists():
        for item in study_dir.iterdir():
            if item == config_path:
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        log.info(f"Cleared {study_dir} (config retained)")

    # Recreate study from saved config
    if config_path.exists():
        config = OmegaConf.load(config_path)
        storage = default_storage(study_name)
        optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=config.direction,
            pruner=_make_pruner(config.pruner),
        )
        log.info(f"Recreated study '{study_name}'")


def clear_optuna_study(study_name: str) -> None:
    """Delete the entire study directory."""
    study_dir = _study_dir(study_name)
    if study_dir.exists():
        shutil.rmtree(study_dir)
        log.info(f"Deleted {study_dir}")
    else:
        log.info(f"No study found at {study_dir}")
