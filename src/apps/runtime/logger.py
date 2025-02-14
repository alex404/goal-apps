"""Shared utilities for GOAL examples."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

import jax
import wandb
from jax import Array
from matplotlib.figure import Figure
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from .handler import RunHandler
from .visualization import setup_matplotlib_style

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


### Jit-Compatible Loggers ###

# Helper functions


def wandb_metric_key(name: str) -> str:
    """Convert metric names to pretty wandb format.

    Examples:
        train_ll -> Metrics/Train Log Likelihood
        test_average_bic -> Metrics/Test Average BIC
    """
    # Special case abbreviations
    replacements = {
        "ll": "Log-Likelihood",
        "bic": "BIC",  # Bayesian Information Criterion
    }

    # Split on underscores
    parts = name.split("_")

    # Process each part
    pretty_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # Check if this part plus maybe next part matches a replacement
        for abbrev, full in replacements.items():
            if i < len(parts) - 1 and f"{parts[i]}_{parts[i + 1]}" == abbrev:
                pretty_parts.append(full)
                i += 2
                break
            if part == abbrev:
                pretty_parts.append(full)
                i += 1
                break
        else:
            # No special case found, just capitalize
            pretty_parts.append(part.capitalize())
            i += 1

    return f"Metrics/{' '.join(pretty_parts)}"


# Global state for metric buffering


_metric_buffer: dict[str, list[tuple[int, float]]] = {}


@dataclass(frozen=True)
class JaxLogger(ABC):
    """Interface for metric and figure logging.

    This logger provides methods for logging both metrics and figures, with
    different JIT compatibility:

    JIT-compatible methods:
        - log_metrics: Can be called from within jax.jit-compiled functions

    Non-JIT methods (must be called outside jax.jit):
        - log_figure: For logging matplotlib figures
        - finalize: For cleanup and final logging
    """

    run_name: str
    run_dir: Path

    def __init__(self, handler: RunHandler) -> None:
        """Initialize logger. Must be called outside of jax.jit."""
        object.__setattr__(self, "run_name", handler.name)
        object.__setattr__(self, "run_dir", handler.run_dir)

    def log_metrics(self, values: dict[str, Array], epoch: int) -> None:
        """Log metrics. Safe to call within jax.jit-compiled functions.

        Uses jax.debug.callback to safely extract values from JIT.

        Args:
            values: Dictionary mapping metric names to JAX arrays
            epoch: Current epoch number
        """

        def _log_values(values_dict: dict[str, Array], epoch: int) -> None:
            float_values = {k: float(v) for k, v in values_dict.items()}
            self._log_metrics(float_values, epoch)

        jax.debug.callback(_log_values, values, epoch)

    @abstractmethod
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        """Implementation-specific logging of metric values."""

    def log_figure(
        self, key: str, fig: Figure, epoch: int, handler: RunHandler | None = None
    ) -> None:
        """Log a matplotlib figure. Must be called outside of jax.jit.

        This method handles matplotlib figures which are incompatible with JIT
        compilation. Any figure generation and logging should happen outside
        of JIT-compiled functions.

        Args:
            key: Name/identifier for the figure
            fig: Matplotlib figure to log
            epoch: Current epoch number
        """
        self._log_figure(key, fig, epoch, handler)

    @abstractmethod
    def _log_figure(
        self, key: str, fig: Figure, epoch: int, handler: RunHandler | None
    ) -> None:
        """Implementation-specific logging of figures."""
        ...

    @abstractmethod
    def finalize(self, handler: RunHandler) -> None:
        """Finalize logging and clean up. Must be called outside of jax.jit."""


@dataclass(frozen=True)
class WandbLogger(JaxLogger):
    """Logger implementation using Weights & Biases.

    Logs metrics in real-time to wandb platform.
    """

    project: str
    group: str | None
    job_type: str | None

    def __init__(
        self,
        handler: RunHandler,
        project: str = "goal",
        group: str | None = None,
        job_type: str | None = None,
    ) -> None:
        super().__init__(handler)
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "job_type", job_type)

        wandb.init(
            project=self.project,
            name=self.run_name,
            group=self.group,
            job_type=self.job_type,
            dir=self.run_dir,
        )

        # Define epoch as our x-axis
        wandb.define_metric("epoch")
        # Use epoch as x-axis for all metrics and artifacts
        wandb.define_metric("*", step_metric="epoch")

    @override
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        pretty_values = {wandb_metric_key(key): value for key, value in values.items()}
        wandb.log({"epoch": epoch, **pretty_values})

    @override
    def _log_figure(
        self, key: str, fig: Figure, epoch: int, handler: RunHandler | None
    ) -> None:
        wandb.log({"epoch": epoch, key: wandb.Image(fig, caption=f"Epoch {epoch}")})

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

    def __init__(self, handler: RunHandler) -> None:
        super().__init__(handler)
        setup_matplotlib_style()
        global _metric_buffer
        _metric_buffer.clear()

    @override
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        global _metric_buffer
        log = logging.getLogger(__name__)

        for metric, value in values.items():
            if metric not in _metric_buffer:
                _metric_buffer[metric] = []
            _metric_buffer[metric].append((epoch, value))
            log.info("epoch %4d | %14s | %10.6f", epoch, metric, value)

    @override
    def _log_figure(
        self, key: str, fig: Figure, epoch: int, handler: RunHandler | None = None
    ) -> None:
        if handler is None:
            raise ValueError("LocalLogger requires a RunHandler for saving figures")

        # Save the figure using handler
        handler.save_plot(fig, f"{key}_epoch_{epoch}")

        # Log to console
        log = logging.getLogger(__name__)
        log.info("epoch %4d | %14s | figure", epoch, key)

    @override
    def finalize(self, handler: RunHandler) -> None:
        global _metric_buffer, _figure_buffer

        # Save metrics
        metrics_dict = {
            metric: [v for _, v in sorted(values)]
            for metric, values in _metric_buffer.items()
        }
        handler.save_json(metrics_dict, "metrics")

        # Clear buffers
        _metric_buffer.clear()


@dataclass(frozen=True)
class NullLogger(JaxLogger):
    @override
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        pass

    @override
    def _log_figure(
        self, key: str, fig: Figure, epoch: int, handler: RunHandler | None
    ) -> None:
        pass

    @override
    def finalize(self, handler: RunHandler) -> None:
        pass
