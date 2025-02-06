"""Shared utilities for GOAL examples."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import override

import jax
import numpy as np
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


### Jit-Compatible Loggers ###

# Helper functions


def wandb_metric_key(name: str) -> str:
    """Convert metric names to pretty wandb format.

    Examples:
        train_ll -> Metric/Train Log Likelihood
        test_average_bic -> Metric/Test Average BIC
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

    return f"Metric/{' '.join(pretty_parts)}"


# Visualization strategies


class ArtifactType(Enum):
    IMAGE = "image"
    SERIES = "series"


# Global state for metric buffering


_metric_buffer: dict[str, list[tuple[int, float]]] = {}
_artifacts_buffer: dict[str, list[tuple[int, Array, ArtifactType]]] = {}


@dataclass(frozen=True)
class JaxLogger(ABC):
    """Interface for metric logging.

    Note: This logger is designed to work with JAX's JIT compilation:

    - Constructor and finalize() must be called from outside JIT-compiled code
    - log_metrics() and log_artifact() can be called from within JIT-compiled code
    - Any values passed to these methods will be evaluated at the point where they are logged through JAX's callback system, and may lead to unpredictable evaluation order of logging calls.
    """

    run_name: str
    run_dir: Path

    def __init__(self, handler: RunHandler) -> None:
        object.__setattr__(self, "run_name", handler.name)
        object.__setattr__(self, "run_dir", handler.run_dir)

    def log_metrics(self, values: dict[str, Array], epoch: int) -> None:
        """Log metrics using callbacks to handle traced values."""

        def _log_values(values_dict: dict[str, Array], epoch: int) -> None:
            float_values = {k: float(v) for k, v in values_dict.items()}
            self._log_metrics(float_values, epoch)

        jax.debug.callback(_log_values, values, epoch)

    @abstractmethod
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        """Implementation-specific logging of metric values."""

    def log_artifact(
        self, key: str, artifact: Array, vis_type: ArtifactType, epoch: int
    ) -> None:
        """Log artifact using callbacks to handle traced values."""

        def _log_artifact(key: str, artifact: Array, epoch: int) -> None:
            self._log_artifact(key, artifact, vis_type, epoch)

        jax.debug.callback(_log_artifact, key, artifact, epoch)

    @abstractmethod
    def _log_artifact(
        self, key: str, artifact: Array, vis_type: ArtifactType, epoch: int
    ) -> None: ...

    @abstractmethod
    def finalize(self, handler: RunHandler) -> None:
        """Finalize logging and clean up. Not safe for jit."""


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
        wandb.define_metric("Metric/*", step_metric="epoch")

    @override
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        pretty_values = {wandb_metric_key(key): value for key, value in values.items()}
        wandb.log({"epoch": epoch, **pretty_values})

    @override
    def _log_artifact(
        self, key: str, artifact: Array, vis_type: ArtifactType, epoch: int
    ) -> None:
        if vis_type == ArtifactType.IMAGE:
            artifact_np = np.array(artifact, dtype=np.float32)
            wandb.log({key: wandb.Image(artifact_np, caption=f"Epoch {epoch}")})
        elif vis_type == ArtifactType.SERIES:
            table = wandb.Table(
                data=[[i, float(y)] for i, y in enumerate(artifact)],
                columns=["x", "y"],
            )
            wandb.log({key: wandb.plot.line(table, "x", "y", title=f"Epoch {epoch}")})

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

        # Initialize buffer
        global _metric_buffer
        _metric_buffer.clear()  # Clear existing buffer if any

    @override
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        global _metric_buffer
        log = logging.getLogger("jit_logger")

        for metric, value in values.items():
            if metric not in _metric_buffer:
                _metric_buffer[metric] = []
            _metric_buffer[metric].append((epoch, value))
            log.info("epoch %4d | %14s | %10.6f", epoch, metric, value)

    @override
    def _log_artifact(
        self, key: str, artifact: Array, vis_type: ArtifactType, epoch: int
    ) -> None:
        global _artifacts_buffer
        if key not in _artifacts_buffer:
            _artifacts_buffer[key] = []
        _artifacts_buffer[key].append((epoch, artifact, vis_type))

        # Still log that we received the data
        log = logging.getLogger("jit_logger")
        log.info("epoch %4d | %14s | %s", epoch, key, vis_type.value)

    @override
    def finalize(self, handler: RunHandler) -> None:
        global _metric_buffer, _artifacts_buffer

        # Save metrics
        metrics_dict = {
            metric: [v for _, v in sorted(values)]
            for metric, values in _metric_buffer.items()
        }
        handler.save_json(metrics_dict, "metrics")

        # Save artifact
        artifact_dict = {
            key: {
                f"epoch_{int(epoch)}": {
                    "artifact": artifact.tolist(),  # Convert array to list for JSON
                    "type": vis_type.value,
                }
                for epoch, artifact, vis_type in entries
            }
            for key, entries in _artifacts_buffer.items()
        }
        handler.save_json(artifact_dict, "artifacts")

        # Clear buffers
        _metric_buffer.clear()
        _artifacts_buffer.clear()


@dataclass(frozen=True)
class NullLogger(JaxLogger):
    """Logger implementation that does nothing."""

    @override
    def _log_metrics(self, values: dict[str, float], epoch: int) -> None:
        pass

    @override
    def _log_artifact(
        self, key: str, artifact: Array, vis_type: ArtifactType, epoch: int
    ) -> None:
        pass

    @override
    def finalize(self, handler: RunHandler) -> None:
        pass
