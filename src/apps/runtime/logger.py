"""Shared utilities for GOAL examples."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import jax
import wandb
from jax import Array
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from .handler import RunHandler

### Logging ###


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


def setup_logging(run_dir: Path) -> None:
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

    # Create formatters
    # Rich handler already handles the time, so we don't include it in the format
    console_format = "%(name)-20s | %(message)s"
    file_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"

    console_handler.setFormatter(logging.Formatter(console_format))
    console_handler.setLevel(logging.INFO)

    # File handler (keeping this as standard logging for clean logs)
    log_file = run_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(file_format))
    file_handler.setLevel(logging.INFO)

    # Set up root logger
    logging.root.handlers = [console_handler, file_handler]
    logging.root.setLevel(logging.INFO)


### Global state for metric buffering ###


_metric_buffer: dict[str, list[tuple[int, float]]] = {}


### Jit-Compatible Loggers ###


@dataclass(frozen=True)
class JaxLogger(ABC):
    """Interface for metric logging.

    Note: __init__ and finalize take RunHandler and should never be called
    within jitted code. log_metrics and log_image are pure functions suitable
    for jit compilation.
    """

    run_name: str
    run_dir: Path

    @abstractmethod
    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        """Log metrics for current step. Safe for jit."""

    # @abstractmethod
    # def log_image(self, key: str, array: Array) -> None:
    #     """Log image array. Safe for jit."""

    @abstractmethod
    def finalize(self, handler: RunHandler) -> None:
        """Finalize logging and clean up. Not safe for jit."""


@dataclass(frozen=True)
class WandbLogger(JaxLogger):
    """Logger implementation using Weights & Biases.

    Logs metrics in real-time to wandb platform.
    """

    project: str = "goal"
    group: str | None = None
    job_type: str | None = None

    def __post_init__(
        self,
    ) -> None:
        wandb.init(
            project=self.project,
            name=self.run_name,
            group=self.group,
            job_type=self.job_type,
            dir=self.run_dir,
        )

    @override
    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        super().log_metrics(values, step)
        wandb.log({k: float(v) for k, v in values.items()})

    # @override
    # def log_image(self, key: str, array: Array) -> None:
    #     """Log image array data to wandb."""
    #     wandb.log({key: wandb.Image(array)})

    @override
    def finalize(self, handler: RunHandler) -> None:
        """Clean up wandb run."""
        wandb.finish()


@dataclass(frozen=True)
class LocalLogger(JaxLogger):
    """Local filesystem logger with metric buffering.

    Note: Uses global state for metric buffering to maintain compatibility with
    JAX's pure function requirements. Only one logger instance should be active
    at a time.
    """

    def __post_init__(self) -> None:
        # Initialize buffer
        global _metric_buffer
        _metric_buffer.clear()  # Clear existing buffer if any

    @override
    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        def _log_value(metric: str, value: Any, step: int) -> None:
            global _metric_buffer
            float_val = float(value)  # Do the float conversion here
            if metric not in _metric_buffer:
                _metric_buffer[metric] = []
            _metric_buffer[metric].append((step, float_val))
            # Do the string formatting here too
            log = logging.getLogger("jit_logger")
            log.info("Step %4d | %14s | %10.6f", step, metric, float_val)

        for metric, value in values.items():
            jax.debug.callback(_log_value, metric, value, step)

    @override
    def finalize(self, handler: RunHandler) -> None:
        global _metric_buffer
        metrics_dict = {
            metric: [v for _, v in sorted(values)]
            for metric, values in _metric_buffer.items()
        }
        handler.save_json({"metrics": metrics_dict}, "metrics")

        # Clear buffer
        _metric_buffer.clear()


@dataclass(frozen=True)
class NullLogger(JaxLogger):
    """Logger implementation that does nothing."""

    @override
    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        pass

    # @override
    # def log_image(self, key: str, array: Array) -> None:
    #     pass

    @override
    def finalize(self, handler: RunHandler) -> None:
        pass
