"""Shared utilities for GOAL examples."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import matplotlib.pyplot as plt
import wandb as wandb
from jax import Array
from jax import numpy as jnp
from jax.experimental import io_callback
from matplotlib.figure import Figure

from .handler import Artifact, MetricDict, MetricHistory, RunHandler

## Preamble ###

log = logging.getLogger(__name__)

### Helpers ###


def plot_metrics(metrics: MetricHistory) -> Figure:
    """Create a summary plot of all metrics over training.

    Args:
        metrics: Dictionary mapping metric names to lists of (epoch, value) pairs

    Returns:
        Figure containing subplots for each metric
    """
    # Create figure
    n_metrics = len(metrics)
    side_length = math.ceil(math.sqrt(n_metrics))
    fig, axes = plt.subplots(
        side_length, side_length, figsize=(6 * side_length, 4 * side_length)
    )

    axes = axes.ravel()

    # Plot each metric
    for ax, (name, values) in zip(axes, metrics.items()):
        epochs, metric_values = zip(*values)
        ax.plot(epochs, metric_values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True)

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


### Jit-Compatible Loggers ###

# Global buffer for metrics
_metric_buffer: MetricHistory = {}


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

    def monitor_params(
        self,
        param_dict: dict[str, Array],
        handler: RunHandler,
        context: str = "update",
    ) -> None:
        """Monitor parameters for NaNs and save state if found."""
        # Check for NaNs using vectorized JAX operations
        has_any_nans = False
        for p in param_dict.values():
            has_any_nans = jnp.logical_or(has_any_nans, jnp.any(jnp.isnan(p)))

        # Define the IO callback function
        def save_debug(params: dict[str, Array]) -> None:
            handler.save_debug_state(params, context)
            log.error(f"NaNs detected in {context}")
            raise ValueError(f"NaN values detected in {context}")

        # Function for no-op case
        def no_op(_: None) -> None:
            pass

        # Only execute the IO callback if NaNs exist
        def call_io(_: Any) -> None:
            io_callback(save_debug, None, param_dict)

        # Use JAX conditional to only execute callback when needed
        jax.lax.cond(has_any_nans, call_io, no_op, None)

    def log_metrics(self, metrics: MetricDict, epoch: Array) -> None:
        """Log metrics. Safe to call within jax.jit-compiled functions."""

        def _log_values(metrics_dict: MetricDict, epoch_array: Array) -> None:
            epoch = int(epoch_array)
            # Convert arrays to floats
            float_metrics = {
                key: (int(level), float(value))
                for key, (level, value) in metrics_dict.items()
            }

            if self.use_local:
                global _metric_buffer
                for key, (level, value) in float_metrics.items():
                    if key not in _metric_buffer:
                        _metric_buffer[key] = []
                    _metric_buffer[key].append((int(epoch), value))
                    log.log(
                        level,
                        "epoch %4d | %14s | %10.6f",
                        epoch,
                        key,
                        value,
                    )

            if self.use_wandb:
                # Single wandb call with all metrics
                wandb.log(
                    {
                        "epoch": epoch,
                        **{key: value for key, (_, value) in float_metrics.items()},
                    }
                )

        jax.debug.callback(_log_values, metrics, epoch)

    def log_artifact[T: Artifact](
        self,
        handler: RunHandler,
        epoch: int,
        artifact: T,
        plot_artifact: Callable[[T], Figure],
    ) -> None:
        """Log a figure and its data. Must be called outside of jax.jit."""
        fig = plot_artifact(artifact)
        name = artifact.__class__.__name__

        if self.use_local:
            handler.save_artifact(epoch, artifact)
            handler.save_artifact_figure(epoch, type(artifact), fig)
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

            handler.save_metrics(_metric_buffer)
            fig = plot_metrics(_metric_buffer)
            handler.save_metrics_figure(fig)
            plt.close(fig)

            _metric_buffer.clear()

        if self.use_wandb:
            wandb.finish()
