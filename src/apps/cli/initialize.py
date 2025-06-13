"""Shared utilities for GOAL examples."""

import logging
import os
import sys
import traceback
from pathlib import Path
from types import TracebackType

import hydra
import jax
import matplotlib.pyplot as plt
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from jax.lib import xla_bridge
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from ..interface import (
    Dataset,
    Model,
)
from ..runtime import Logger, LogLevel, RunHandler
from .configs import RunConfig
from .util import print_config_tree

### Python Logging ###

logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)

log = logging.getLogger(__name__)

# Custom theme for our logging
THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "red reverse",
        "metric": "green",
        "step": "blue",
        "value": "yellow",
    }
)


### Initialization Helpers ###


def filter_group_defaults(config_dir: str, overrides: list[str]):
    groups = {d.name for d in Path(config_dir).iterdir() if d.is_dir()}
    return [o for o in overrides if "=" not in o or o.split("=", 1)[0] not in groups]


def setup_matplotlib_style() -> None:
    """Load and set the default matplotlib style."""
    style_path = Path(__file__).parents[3] / "config" / "default.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


def setup_logging(run_dir: Path, log_level: LogLevel) -> None:
    """Configure logging for the entire application with pretty formatting."""
    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create console with our theme
    console = Console(theme=THEME)

    # Console handler using Rich
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,  # We'll show this in the format string instead
        rich_tracebacks=True,
        tracebacks_width=None,  # Use full width
        markup=True,  # Enable rich markup in log messages
    )

    def exception_handler(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt exit gracefully
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Optionally, save to a custom file
        error_file = run_dir / "errors.log"
        with open(error_file, "a") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

    sys.excepthook = exception_handler

    # Create formatters
    # Rich handler already handles the time, so we don't include it in the format
    console_format = "%(name)-20s | %(message)s"
    file_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

    level = log_level.value

    console_handler.setFormatter(logging.Formatter(console_format))
    console_handler.setLevel(level)

    # File handler (keeping this as standard logging for clean logs)

    log_file = run_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(file_format))
    file_handler.setLevel(level)

    # Set up root logger
    logging.root.handlers = [console_handler, file_handler]
    logging.root.setLevel(level)


def setup_jax(device: str = "cpu", disable_jit: bool = False) -> None:
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)


def make_run_dir(
    project_root: Path, run_name: str, sweep_id: str | None = None
) -> Path:
    # Compute run_dir at init
    base = project_root / "runs"
    sweep_id = sweep_id or os.environ.get("WANDB_SWEEP_ID")
    run_dir = (
        base / "sweep" / sweep_id / run_name if sweep_id else base / "single" / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


### Core Initialization Function ###


def initialize_run(
    run_type: type[RunConfig],
    overrides: list[str],
) -> tuple[RunHandler, Logger, Dataset, Model[Dataset]]:
    """Initialize a new run with hydra config and wandb logging."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=run_type)
    proot = Path(__file__).parents[3]
    config_dir = str(proot / "config" / "hydra")

    # Initialize hydra config
    with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = hydra.compose(config_name="config", overrides=overrides)

        run_dir = make_run_dir(
            project_root=proot, run_name=cfg.run_name, sweep_id=cfg.sweep_id
        )

        saved_config_path = run_dir / "config.yaml"

        # overrides > saved config > defaults
        if saved_config_path.exists():
            saved_dict = OmegaConf.load(saved_config_path)
            filtered_overrides = filter_group_defaults(config_dir, overrides)
            override_config = OmegaConf.from_dotlist(filtered_overrides)
            cfg = OmegaConf.merge(cfg, saved_dict, override_config)

    # Initialize run handler
    handler = RunHandler(
        run_name=cfg.run_name,
        project_root=proot,
        run_dir=run_dir,
        requested_epoch=cfg.from_epoch,
        from_scratch=cfg.from_scratch,
    )

    logger = Logger(
        run_name=cfg.run_name,
        run_dir=run_dir,
        use_local=cfg.use_local,
        use_wandb=cfg.use_wandb,
        project=cfg.project,
        group=cfg.group,
        job_type=cfg.job_type,
        run_id_override=cfg.run_id,
    )

    logger.set_metric_buffer(handler.load_metrics())

    # Initialize JAX
    setup_jax(device=cfg.device, disable_jit=not cfg.jit)

    setup_matplotlib_style()

    setup_logging(handler.run_dir, log_level=cfg.log_level)

    log.info(f"Run name: {handler.run_name}")
    log.info(f"Project Root: {handler.project_root}")
    log.info(f"Available devices: {jax.devices()}")
    log.info(f"JAX backend: {xla_bridge.get_backend().platform}")
    log.info(f"with JIT: {cfg.jit}")

    log.info("Loading dataset...")
    dataset: Dataset = instantiate(cfg.dataset, cache_dir=handler.cache_dir)
    log.info(f"Loaded dataset with {len(dataset.train_data)} training data points.")

    # Instantiate model
    log.info("Loading model...")
    model: Model[Dataset] = instantiate(cfg.model, data_dim=dataset.data_dim)

    # update cfg with handler run_id

    # wandb is the authory on the run_id
    cfg.model.data_dim = dataset.data_dim

    logger.log_config(OmegaConf.to_container(cfg, resolve=True))  # pyright: ignore[reportArgumentType]
    print_config_tree(OmegaConf.to_yaml(cfg, resolve=True))

    # will return logger as well
    return handler, logger, dataset, model
