"""Shared utilities for GOAL examples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import matplotlib.pyplot as plt
import wandb as wandb
from matplotlib.figure import Figure

from .handler import Artifact, MetricDict, MetricHistory, RunHandler
from .visualization import plot_metrics

## Preamble ###

log = logging.getLogger(__name__)

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

    def log_metrics(self, metrics: MetricDict, epoch: int) -> None:
        """Log metrics. Safe to call within jax.jit-compiled functions."""

        def _log_values(metrics_dict: MetricDict, epoch: int) -> None:
            # Convert arrays to floats
            float_metrics = {
                key: (display_name, int(level), float(value))
                for key, (display_name, level, value) in metrics_dict.items()
            }

            if self.use_local:
                global _metric_buffer
                for key, (display_name, level, value) in float_metrics.items():
                    if key not in _metric_buffer:
                        _metric_buffer[key] = []
                    _metric_buffer[key].append((int(epoch), value))
                    log.log(
                        level,
                        "epoch %4d | %14s | %10.6f",
                        epoch,
                        display_name,
                        value,
                    )

            if self.use_wandb:
                # Single wandb call with all metrics
                wandb.log(
                    {
                        "epoch": epoch,
                        **{
                            display_name: value
                            for _, (display_name, _, value) in float_metrics.items()
                        },
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
