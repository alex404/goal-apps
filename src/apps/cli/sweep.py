### Imports ###

import difflib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import joblib
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

TUNE_DIR = Path("runs/tune")


class OptunaValidationError(Exception):
    """Raised when validation of a proposed Optuna study configuration fails."""


class OptunaImportError(Exception):
    """Raised when trial import fails due to a search-space conflict or missing state."""


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


def _parse_scalar(s: str) -> Any:
    """Parse a scalar literal: int if it looks like one, else float, else string."""
    try:
        return float(s) if "." in s or "e" in s or "E" in s else int(s)
    except ValueError:
        return s


def _parse_categorical_choices(choices_str: str) -> list[Any]:
    """Parse a suggest_categorical value list into typed Python values."""
    return [_parse_scalar(c.strip()) for c in choices_str.split(",")]


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
    if value.startswith("suggest_optional:"):
        # suggest_optional:<off_value>:<inner_directive>
        # A boolean gate at "<name>__enabled" decides: False → off_value,
        # True → resolve the inner directive under the same name.
        rest = value[len("suggest_optional:") :]
        off_str, sep, inner = rest.partition(":")
        if not sep:
            msg = f"suggest_optional requires <off_value>:<inner_directive>, got {value!r}"
            raise ValueError(msg)
        gate_name = f"{name}__enabled"
        if sampler.suggest_categorical(gate_name, [False, True]):
            resolved = _resolve_directive(name, inner, sampler)
            if resolved is None:
                msg = f"suggest_optional inner directive not recognized: {inner!r}"
                raise ValueError(msg)
            return resolved
        return _parse_scalar(off_str)
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


def _resolve_overrides(sampler: Any, overrides: list[str]) -> list[str]:
    """Resolve suggest_* directives in overrides via ``sampler``.

    ``sampler`` must expose ``suggest_float``, ``suggest_int``, and
    ``suggest_categorical`` methods (e.g. an ``optuna.trial.Trial`` or
    ``_FirstValueSampler``).
    """
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


def resolve_optuna_overrides(trial: Any, overrides: list[str]) -> list[str]:
    """Resolve overrides containing suggest_* directives into concrete values.

    Syntax:
        param=suggest_float:low:high              -> trial.suggest_float(param, low, high)
        param=suggest_float:low:high:log          -> trial.suggest_float(param, low, high, log=True)
        param=suggest_int:low:high                -> trial.suggest_int(param, low, high)
        param=suggest_categorical:a,b,c           -> trial.suggest_categorical(param, [a, b, c])
        param=suggest_optional:off:<inner>        -> gated on boolean "<param>__enabled":
                                                     False → off, True → resolve <inner>
        param=value                               -> passed through unchanged

    Returns:
        List of resolved override strings.
    """
    return _resolve_overrides(trial, overrides)


def sample_optuna_overrides(overrides: list[str]) -> list[str]:
    """Resolve suggest_* directives by picking a deterministic sample point.

    Used for validation — no ``optuna.Trial`` required.
    """
    return _resolve_overrides(_FirstValueSampler(), overrides)


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


_VALID_PRUNERS = ("median", "percentile", "none")


def _normalize_pruner_config(pruner: Any) -> dict[str, Any]:
    """Coerce a pruner entry from study-config.yaml into a dict.

    Accepts a bare string (legacy shape: ``pruner: median``) or a mapping
    (``pruner: {name: median, n_warmup_steps: 200, ...}``). Returns a
    plain dict with a ``name`` key and any user-supplied kwargs.
    """
    if pruner is None:
        return {"name": "median"}
    if isinstance(pruner, str):
        return {"name": pruner}
    # DictConfig or plain dict — both support dict-iteration
    cfg: dict[str, Any] = {str(k): pruner[k] for k in pruner}
    if "name" not in cfg:
        msg = f"pruner config missing 'name' key: {cfg}"
        raise ValueError(msg)
    return cfg


def _make_pruner(pruner: Any) -> Any:
    """Build an Optuna pruner from a config mapping (or legacy string)."""
    import optuna

    cfg = _normalize_pruner_config(pruner)
    name = cfg["name"]
    if name not in _VALID_PRUNERS:
        msg = f"Unknown pruner: {name}. Choose from: {list(_VALID_PRUNERS)}"
        raise ValueError(msg)

    if name == "none":
        return optuna.pruners.NopPruner()

    kwargs: dict[str, Any] = {}
    for k in ("n_startup_trials", "n_warmup_steps", "interval_steps", "n_min_trials"):
        if k in cfg and cfg[k] is not None:
            kwargs[k] = int(cfg[k])

    if name == "median":
        # Preserve historical defaults when unspecified.
        kwargs.setdefault("n_startup_trials", 3)
        kwargs.setdefault("n_warmup_steps", 1)
        return optuna.pruners.MedianPruner(**kwargs)

    # percentile
    percentile = float(cfg.get("percentile", 25.0))
    kwargs.setdefault("n_startup_trials", 3)
    kwargs.setdefault("n_warmup_steps", 1)
    return optuna.pruners.PercentilePruner(percentile, **kwargs)


def create_optuna_study(
    overrides: list[str],
    study_name: str,
    metric: str,
    direction: str,
    pruner_config: dict[str, Any],
    storage: str | None = None,
) -> Any:
    """Create a new Optuna study and save its config for workers.

    ``pruner_config`` is a mapping with a ``name`` key (one of ``median``,
    ``percentile``, ``none``) and optional kwargs (``n_startup_trials``,
    ``n_warmup_steps``, ``interval_steps``, ``n_min_trials``,
    ``percentile``). Only explicitly-set kwargs should be included — the
    pruner constructor fills in defaults.

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
            "pruner": pruner_config,
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
        pruner=_make_pruner(pruner_config),
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

    from apps.runtime import DivergentTrainingError, Logger

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
    storage: str = config.storage or default_storage(study_name)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        pruner=_make_pruner(config.pruner),
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
            # DivergentTrainingError raised from an io_callback inside JIT is
            # wrapped in XlaRuntimeError with no __cause__. Logger.monitor_params
            # sets a module-level flag before raising, which survives the wrap.
            if Logger.divergence_raised():
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


### Trial Import ###


@dataclass
class ImportSummary:
    """Summary of a trial import operation."""

    aligned: list[str] = field(default_factory=list)
    target_only: list[str] = field(default_factory=list)
    source_only: list[str] = field(default_factory=list)
    imported_complete: int = 0
    imported_pruned: int = 0
    skipped_missing_metric: int = 0
    skipped_nan_value: int = 0


def _collect_source_distributions(source_study: Any) -> dict[str, Any]:
    """Union of ``trial.distributions`` across all source trials.

    Different trials can have different subsets (for conditional params), so
    we take the union — a param is considered present if any trial sampled it.
    """
    dists: dict[str, Any] = {}
    for t in source_study.trials:
        for name, dist in t.distributions.items():
            dists.setdefault(name, dist)
    return dists


def _enumerate_target_distributions(overrides: list[str]) -> dict[str, Any]:
    """Enumerate every distribution the target study may sample, following
    both branches of ``suggest_optional`` so conditional inner params are
    included alongside their gate.

    Returns a mapping of ``param_name -> optuna.distributions.*``.
    """
    dists: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            continue
        name, _, value = override.partition("=")
        _add_directive_distributions(name, value, dists)
    return dists


def _add_directive_distributions(name: str, value: str, dists: dict[str, Any]) -> None:
    """Parse a single directive and register the distribution(s) it produces."""
    import optuna.distributions as od

    if value.startswith("suggest_float:"):
        parts = value.split(":")[1:]
        low, high = float(parts[0]), float(parts[1])
        use_log = len(parts) > 2 and parts[2] == "log"
        dists[name] = od.FloatDistribution(low, high, log=use_log)
    elif value.startswith("suggest_int:"):
        parts = value.split(":")[1:]
        low, high = int(parts[0]), int(parts[1])
        dists[name] = od.IntDistribution(low, high)
    elif value.startswith("suggest_categorical:"):
        choices_str = value.split(":", 1)[1]
        dists[name] = od.CategoricalDistribution(
            _parse_categorical_choices(choices_str)
        )
    elif value.startswith("suggest_optional:"):
        rest = value[len("suggest_optional:") :]
        _off, sep, inner = rest.partition(":")
        if sep:
            dists[f"{name}__enabled"] = od.CategoricalDistribution([False, True])
            _add_directive_distributions(name, inner, dists)


def _dist_family(dist: Any) -> str:
    """Distribution family identifier for diff purposes.

    Two distributions with the same family but different ranges/choices are
    still 'aligned' — TPE tolerates out-of-range historical values. A
    family mismatch (float vs categorical) is a genuine conflict.
    """
    return type(dist).__name__


def _diff_distributions(
    source: dict[str, Any], target: dict[str, Any]
) -> tuple[list[str], list[str], list[str], dict[str, tuple[str, str]]]:
    """Diff source and target distribution dicts.

    Returns (aligned, target_only, source_only, conflicts) where conflicts
    maps name → (source_family, target_family).
    """
    aligned: list[str] = []
    conflicts: dict[str, tuple[str, str]] = {}
    for name, sdist in source.items():
        if name in target:
            sf = _dist_family(sdist)
            tf = _dist_family(target[name])
            if sf == tf:
                aligned.append(name)
            else:
                conflicts[name] = (sf, tf)
    target_only = sorted(set(target) - set(source))
    source_only = sorted(set(source) - set(target))
    return sorted(aligned), target_only, source_only, conflicts


def import_trials(
    source: str,
    target: str,
    include_pruned: bool = False,
    force: bool = False,
) -> ImportSummary:
    """Import completed (and optionally pruned) trials from a source study
    into a target study, re-extracting the target's metric from each
    source trial's on-disk ``metrics.joblib``.

    The source study is never mutated. Conflicts in distribution family
    on shared param names raise ``OptunaImportError``. Divergent ranges
    and asymmetric param sets are tolerated (Optuna/TPE handle sparse
    histories natively).
    """
    import optuna

    source_dir = _study_dir(source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source study dir not found: {source_dir}")

    target_config_path = study_config_path(target)
    if not target_config_path.exists():
        msg = (
            f"Target study config not found at {target_config_path}. "
            "Run 'goal tune optuna create' first."
        )
        raise FileNotFoundError(msg)

    source_study = optuna.load_study(study_name=source, storage=default_storage(source))

    target_config = OmegaConf.load(target_config_path)
    target_storage: str = target_config.storage or default_storage(target)
    target_study = optuna.load_study(
        study_name=target,
        storage=target_storage,
        pruner=_make_pruner(target_config.pruner),
    )

    if len(target_study.trials) > 0 and not force:
        msg = (
            f"Target study '{target}' already has {len(target_study.trials)} trials. "
            "Pass --force to import on top (may blend native with imported history)."
        )
        raise OptunaImportError(msg)

    source_dists = _collect_source_distributions(source_study)
    target_dists = _enumerate_target_distributions(list(target_config.overrides))
    aligned, target_only, source_only, conflicts = _diff_distributions(
        source_dists, target_dists
    )

    if conflicts:
        lines = [
            "Search-space conflicts (distribution family mismatch on shared names):"
        ]
        for name, (sf, tf) in sorted(conflicts.items()):
            lines.append(f"  {name}: source={sf}, target={tf}")
        lines.append(
            "Re-create the target study with a matching family or drop the conflicting param."
        )
        raise OptunaImportError("\n".join(lines))

    target_metric: str = target_config.metric
    summary = ImportSummary(
        aligned=aligned, target_only=target_only, source_only=source_only
    )

    for t in source_study.trials:
        _import_one_trial(
            t, source, source_dir, target_metric, target_study, include_pruned, summary
        )

    return summary


def _import_one_trial(
    trial: Any,
    source_name: str,
    source_dir: Path,
    target_metric: str,
    target_study: Any,
    include_pruned: bool,
    summary: ImportSummary,
) -> None:
    """Import a single source trial into the target study, updating ``summary``."""
    import math

    import optuna
    from optuna.trial import TrialState

    user_attrs = {"imported_from": source_name}

    if trial.state == TrialState.COMPLETE:
        metrics_path = source_dir / f"t{trial.number}" / "metrics.joblib"
        if not metrics_path.exists():
            summary.skipped_missing_metric += 1
            return
        metrics = joblib.load(metrics_path)
        if target_metric not in metrics or not metrics[target_metric]:
            summary.skipped_missing_metric += 1
            return
        _, value = metrics[target_metric][-1]
        value = float(value)
        if not math.isfinite(value):
            summary.skipped_nan_value += 1
            return
        frozen = optuna.trial.create_trial(
            params=trial.params,
            distributions=trial.distributions,
            value=value,
            state=TrialState.COMPLETE,
            user_attrs=user_attrs,
        )
        target_study.add_trial(frozen)
        summary.imported_complete += 1
    elif trial.state == TrialState.PRUNED and include_pruned:
        frozen = optuna.trial.create_trial(
            params=trial.params,
            distributions=trial.distributions,
            state=TrialState.PRUNED,
            user_attrs=user_attrs,
        )
        target_study.add_trial(frozen)
        summary.imported_pruned += 1
