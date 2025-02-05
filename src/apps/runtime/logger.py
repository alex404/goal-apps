"""Shared utilities for GOAL examples."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import jax
import wandb
from jax import Array

from .handler import RunHandler

### Global state for metric buffering ###


_metric_buffer: dict[str, list[tuple[int, float]]] = {}


### Loggers ###


@dataclass(frozen=True)
class Logger(ABC):
    """Interface for metric logging.

    Note: __init__ and finalize take RunHandler and should never be called
    within jitted code. log_metrics and log_image are pure functions suitable
    for jit compilation.
    """

    metrics: list[str]
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
class WandbLogger(Logger):
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
class LocalLogger(Logger):
    """Local filesystem logger with metric buffering.

    Note: Uses global state for metric buffering to maintain compatibility with
    JAX's pure function requirements. Only one logger instance should be active
    at a time.
    """

    def __post_init__(self) -> None:
        # Initialize buffer
        global _metric_buffer
        _metric_buffer.clear()  # Clear existing buffer if any
        _metric_buffer.update({metric: [] for metric in self.metrics})

    @override
    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        def _log_value(metric: str, value: Any, step: int) -> None:
            global _metric_buffer
            float_val = float(value)  # Do the float conversion here
            _metric_buffer[metric].append((step, float_val))
            # Do the string formatting here too
            print(f"{self.run_name} - Step {step}: {metric}={float_val:.6f}")

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
class NullLogger(Logger):
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
