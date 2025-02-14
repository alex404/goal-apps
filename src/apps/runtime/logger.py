"""Shared utilities for GOAL examples."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, override

import jax
import matplotlib.pyplot as plt
import wandb as wandb
from jax import Array
from matplotlib.figure import Figure

from .handler import JSONDict, JSONList, JSONValue, RunHandler
from .visualization import plot_metrics

### Jit-Compatible Loggers ###

# Artifacts


@dataclass(frozen=True)
class Artifact(ABC):
    """Base class for data that can be logged and visualized."""

    @abstractmethod
    def to_json(self) -> JSONValue:
        """Convert artifact data to JSON-serializable format."""
        ...


@dataclass(frozen=True)
class ArrayArtifact(Artifact):
    """Artifact wrapping a JAX array."""

    data: Array

    @override
    def to_json(self) -> JSONValue:
        return self.data.tolist()


# Helpers


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


# Global buffer for metrics
_metric_buffer: list[tuple[int, dict[str, float]]] = []


@dataclass(frozen=True)
class JaxLogger:
    """Logger supporting both local and wandb logging.

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
    use_wandb: bool
    use_local: bool

    def __init__(
        self,
        handler: RunHandler,
        use_wandb: bool,
        use_local: bool,
        project: str,
        group: str | None,
        job_type: str | None,
    ) -> None:
        """Initialize logger with desired logging destinations."""
        object.__setattr__(self, "run_name", handler.name)
        object.__setattr__(self, "run_dir", handler.run_dir)
        object.__setattr__(self, "use_wandb", use_wandb)
        object.__setattr__(self, "use_local", use_local)

        if use_wandb:
            wandb.init(
                project=project,
                name=self.run_name,
                group=group,
                job_type=job_type,
                dir=self.run_dir,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")

        if use_local:
            global _metric_buffer
            _metric_buffer.clear()

    def log_metrics(self, values: dict[str, Array], epoch: int) -> None:
        """Log metrics. Safe to call within jax.jit-compiled functions."""

        def _log_values(values_dict: dict[str, Array], epoch: int) -> None:
            float_values = {k: float(v) for k, v in values_dict.items()}

            if self.use_local:
                global _metric_buffer
                _metric_buffer.append((epoch, float_values))

                log = logging.getLogger(__name__)
                for metric, value in float_values.items():
                    log.info("epoch %4d | %14s | %10.6f", epoch, metric, value)

            if self.use_wandb:
                pretty_keys = {
                    wandb_metric_key(key): value for key, value in float_values.items()
                }
                wandb.log({"epoch": epoch, **pretty_keys})

        jax.debug.callback(_log_values, values, epoch)

    def log_artifact[T: Artifact](
        self,
        handler: RunHandler,
        epoch: int,
        name: str,
        artifact: T,
        plot_artifact: Callable[[T], Figure],
    ) -> None:
        """Log a figure and its data. Must be called outside of jax.jit."""
        fig = plot_artifact(artifact)

        if self.use_local:
            # Save both data and figure
            handler.save_json(
                {"data": artifact.to_json(), "epoch": epoch},
                f"{name}_epoch_{epoch}_data",
            )
            handler.save_plot(fig, f"{name}_epoch_{epoch}")

            log = logging.getLogger(__name__)
            log.info("epoch %4d | %14s | figure", epoch, name)

        if self.use_wandb:
            wandb.log(
                {"epoch": epoch, name: wandb.Image(fig, caption=f"Epoch {epoch}")}
            )

        plt.close(fig)

    def finalize(self, handler: RunHandler) -> None:
        """Finalize logging and clean up. Must be called outside of jax.jit."""
        if self.use_local:
            global _metric_buffer

            # First convert to lists of primitives
            epochs: JSONList = [int(epoch) for epoch, _ in _metric_buffer]
            metrics: JSONList = [
                {
                    k: float(v) for k, v in metrics.items()
                }  # each metrics dict is JSONDict
                for _, metrics in _metric_buffer
            ]

            # Then construct the final dictionary
            json_data: JSONDict = {"epochs": epochs, "metrics": metrics}

            # Save and plot
            handler.save_json(json_data, "metrics")
            fig = plot_metrics(_metric_buffer)
            handler.save_plot(fig, "metrics")

            _metric_buffer.clear()

        if self.use_wandb:
            wandb.finish()
