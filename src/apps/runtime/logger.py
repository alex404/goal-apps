"""Logger for JAX-based applications. Isolates logging functionality from the main application logic, allowing for isolated inclusion of logging frameworks like e.g. Weight and Biases."""

import logging
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import wandb
from jax import Array
from jax import numpy as jnp
from jax.experimental import io_callback
from matplotlib.figure import Figure

from .handler import RunHandler
from .util import Artifact, MetricDict, MetricHistory, plot_metrics

## Logging ###

log = logging.getLogger(__name__)

# Global metric buffer
_metric_buffer: MetricHistory = {}

### Artifacts ###


# Global wall clock start time (set when Logger is initialized)
_wall_clock_start: float = 0.0
# Accumulated duration spent in analyses/checkpointing (excluded from training time)
_paused_duration: float = 0.0


@dataclass(frozen=True)
class Logger:
    """Logger supporting both local and wandb logging.

    This logger provides methods for logging both metrics and figures, with
    different JIT compatibility:

    JIT-compatible methods:
        - log_metrics: Can be called from within jax.jit-compiled functions

    Non-JIT methods (must be called outside jax.jit):
        - log_figure: For logging matplotlib figures
        - finalize: For cleanup and final logging

    Wall clock timing is automatically tracked from Logger initialization
    and included in all logged metrics.
    """

    run_name: str
    """Name of the run, used for wandb and local logging."""
    run_dir: Path
    """Directory for this specific run, containing all artifacts and logs."""
    use_local: bool
    """Whether to log metrics and figures locally."""

    # wandb
    use_wandb: bool
    """Whether to log metrics and figures to Weights & Biases."""
    run_id_override: str | None
    """Override for the run ID, if provided."""
    project: str
    """Weights & Biases project name."""
    group: str | None
    """Weights & Biases group name for organizing runs."""
    job_type: str | None
    """Weights & Biases job type for categorizing runs."""

    def __post_init__(self) -> None:
        """Initialize logger with desired logging destinations."""
        global _metric_buffer, _wall_clock_start, _paused_duration

        # Start wall clock timer for tracking training duration
        _wall_clock_start = time.perf_counter()
        _paused_duration = 0.0

        # Clear or load metric buffer based on whether we're resuming
        if self.use_local:
            # Since Logger doesn't have direct access to handler.from_epoch,
            # we'll let the handler load metrics and call set_metric_buffer
            _metric_buffer.clear()

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.project,
                name=self.run_name,
                group=self.group,
                job_type=self.job_type,
                dir=self.run_dir,
                id=self.run_id_override,
                resume="allow",
            )
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")

    @property
    def run_id(self) -> str | None:
        """Get run ID from override or environment."""
        return wandb.run.id if wandb.run else None

    @staticmethod
    def set_metric_buffer(metrics: MetricHistory) -> None:
        """Set the global metric buffer (used when resuming)."""
        global _metric_buffer
        _metric_buffer = metrics.copy()

    @staticmethod
    def get_metric_buffer() -> MetricHistory:
        """Get a copy of the current metric buffer."""
        return _metric_buffer.copy()

    def log_config(self) -> None:
        """Log the configuration dictionary to wandb."""
        # Save config to local file
        if self.use_wandb:
            # Log the entire config dictionary as a single artifact
            wandb.save(
                str(self.run_dir / "run-config.yaml"),
                base_path=self.run_dir,
                policy="now",
            )
            log.info("Configuration logged to Weights & Biases.")

    @staticmethod
    @contextmanager
    def pause_timing() -> Generator[None, None, None]:
        """Context manager that excludes elapsed time from wall clock training time."""
        global _paused_duration
        pause_start = time.perf_counter()
        try:
            yield
        finally:
            _paused_duration += time.perf_counter() - pause_start

    def log_metrics(self, metrics: MetricDict, epoch: Array) -> None:
        """Log metrics. Safe to call within jax.jit-compiled functions."""
        # Capture variables for closure
        use_local = self.use_local
        use_wandb = self.use_wandb

        def _log_values(metrics_dict: MetricDict, epoch_array: Array) -> None:
            global _metric_buffer, _wall_clock_start
            epoch = int(epoch_array)
            # Convert arrays to floats
            float_metrics = {
                key: (int(level), float(value))
                for key, (level, value) in metrics_dict.items()
            }

            # Automatically inject wall clock elapsed time (excluding analysis/checkpoint time)
            elapsed = time.perf_counter() - _wall_clock_start - _paused_duration
            float_metrics["Timing/Wall Clock (s)"] = (logging.INFO, elapsed)

            if use_local:
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

            if use_wandb:
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

    def monitor_params(
        self,
        param_dict: dict[str, Array],
        handler: RunHandler,
        context: str = "update",
    ) -> None:
        """Monitor parameters for NaNs and save state if found. Safe to call within jax.jit-compiled functions."""
        # Check for NaNs using vectorized JAX operations
        has_any_nans = False
        for p in param_dict.values():
            has_any_nans = jnp.logical_or(has_any_nans, jnp.any(jnp.isnan(p)))

        # Define the IO callback function
        def save_debug(params: dict[str, Array]) -> None:
            handler.save_debug_state(params, context)
            log.error(f"NaNs detected in {context}")

            if self.use_wandb and wandb.run:
                wandb.finish(exit_code=1)
            sys.exit(1)

        # Function for no-op case
        def no_op(_: None) -> None:
            pass

        # Only execute the IO callback if NaNs exist
        def call_io(_: Any) -> None:
            io_callback(save_debug, None, param_dict)

        # Use JAX conditional to only execute callback when needed
        jax.lax.cond(has_any_nans, call_io, no_op, None)

    def finalize(self, handler: RunHandler) -> None:
        """Finalize logging and clean up. Must be called outside of jax.jit."""
        global _metric_buffer

        if self.use_local:
            # Save current buffer state
            handler.save_metrics(_metric_buffer)
            fig = plot_metrics(_metric_buffer)
            handler.save_metrics_figure(fig)
            plt.close(fig)

        if self.use_wandb:
            wandb.finish(exit_code=0)

        # Clear buffer after finalization
        _metric_buffer.clear()

    def finish_preempted(self) -> None:
        """Handle SLURM preemption - mark for requeue."""
        if self.use_wandb and wandb.run:
            wandb.mark_preempting()  # pyright: ignore[reportAttributeAccessIssue]
            wandb.finish(exit_code=143)
        sys.exit(143)

    def finish_interrupted(self) -> None:
        """Handle user Ctrl+C - don't requeue."""
        if self.use_wandb and wandb.run:
            wandb.finish(exit_code=130)
        sys.exit(130)
