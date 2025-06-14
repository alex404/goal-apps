"""Manages file IO and organization for a single run in a machine learning experiment."""

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
from jax import Array
from matplotlib.figure import Figure

from .util import Artifact, MetricHistory, to_snake_case

### Logging ###

log = logging.getLogger(__name__)

### Run Handler ###


@dataclass(frozen=True)
class RunHandler:
    """Handles file management and organization for a single run."""

    # Attributes
    run_name: str
    """Name of the run, used for directory naming."""
    project_root: Path
    """Root directory of the project."""
    run_dir: Path
    """Directory for this specific run, containing all artifacts and logs."""
    resume_epoch: int | None
    """Epoch to resume from, or None for auto logic."""
    recompute_artifacts: bool
    """Whether analysis should start from scratch or used existing artifacts."""

    ### Public Properties ###

    @property
    def cache_dir(self) -> Path:
        """Directory for cached data (e.g., datasets)."""
        return self.project_root / ".cache"

    @property
    def metrics_path(self) -> Path:
        """Path to the metrics file for this run."""
        return self.run_dir / "metrics.joblib"

    @property
    def available_epochs(self) -> list[int]:
        """List of epochs available in this run."""
        return sorted(
            int(d.name.split("_")[1])
            for d in self.run_dir.glob("epoch_*")
            if d.is_dir()
        )

    @cached_property
    def resolve_epoch(self) -> int | None:
        """Verified epoch to resume from, or None if starting fresh."""
        if not self.available_epochs:
            log.info("No existing checkpoints found. Starting fresh.")
            return None

        if self.resume_epoch is None:
            latest = max(self.available_epochs)
            log.info(
                f"No epoch specified. Resuming from latest checkpoint: epoch {latest}"
            )
            return latest

        # Special case: explicit reset
        if self.resume_epoch == 0:
            log.info("Epoch 0 requested, starting fresh (explicit reset).")
            return None

        valid_epochs = [e for e in self.available_epochs if e <= self.resume_epoch]
        if valid_epochs:
            resume_epoch = max(valid_epochs)
            if resume_epoch == self.resume_epoch:
                log.info(f"Resuming from requested epoch {resume_epoch}")
            else:
                log.info(
                    f"Resuming from epoch {resume_epoch} (latest checkpoint before requested epoch {self.resume_epoch})"
                )
            return resume_epoch
        log.warning(
            f"No checkpoint found at or before epoch {self.resume_epoch}. Available checkpoints: {sorted(self.available_epochs)}. Starting fresh."
        )
        return None

    ### Public Methods ###

    def save_params(self, params: Array, epoch: int) -> None:
        """Save parameters at a given epoch."""
        path = self._get_params_path(epoch)
        joblib.dump(params, path)

    def load_params(self) -> Array:
        """Load parameters from the resolved epoch.

        Raises:
            ValueError: If from_epoch is set but no params exist at that epoch
            RuntimeError: If from_epoch is None (shouldn't happen after resolution)
        """
        if self.resolve_epoch is None:
            raise RuntimeError("Cannot load params: from_epoch is None (fresh run)")

        return joblib.load(self._get_params_path(self.resolve_epoch))

    def save_metrics(self, metrics: MetricHistory) -> None:
        """Save training metrics."""
        path = self.run_dir / "metrics.joblib"
        joblib.dump(metrics, path)

    def load_metrics(self) -> MetricHistory:
        """Load training metrics from the run directory, up to the resolved epoch."""
        if not self.metrics_path.exists():
            return {}

        full_metrics = joblib.load(self.metrics_path)

        # If resuming, only return metrics up to the resume point
        if self.resolve_epoch is not None:
            return {
                metric_name: [
                    (epoch, value)
                    for epoch, value in values
                    if epoch <= self.resolve_epoch
                ]
                for metric_name, values in full_metrics.items()
            }
        return full_metrics

    def save_metrics_figure(self, fig: Figure) -> None:
        """Save the metrics summary figure."""
        path = self.run_dir / "metrics.png"
        fig.savefig(path, bbox_inches="tight")

    ## Artifact Management
    def save_artifact(self, epoch: int, artifact: Artifact) -> None:
        """Save an artifact at a given epoch."""
        path = self._get_artifact_path(epoch, type(artifact))
        joblib.dump(artifact, path, compress=3)

    def save_artifact_figure(
        self, epoch: int, artifact_class: type[Artifact], fig: Figure
    ) -> None:
        """Save a figure in the same epoch directory as its artifact."""
        path = self._get_plot_path(epoch, artifact_class)
        fig.savefig(path, bbox_inches="tight")

    def load_artifact[T: Artifact](self, epoch: int, artifact_class: type[T]) -> T:
        """Load an artifact from a specific epoch."""
        path = self._get_artifact_path(epoch, artifact_class)
        return joblib.load(path)

    def save_debug_state(
        self,
        param_dict: dict[str, Array],
        context: str,
    ) -> None:
        """Save parameter states for debugging.

        Args:
            param_dict: Dictionary mapping names to parameter states
            context: Description of when the debug state was captured
        """
        debug_dir = self.run_dir / "debug" / context
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Save all parameter states using numpy save (simpler than HDF5 for debug)
        for name, array in param_dict.items():
            np.save(debug_dir / f"{name}.npy", np.array(array))

    ### Private Methods ###

    ## Path Management
    def _get_epoch_dir(self, epoch: int, create: bool = True) -> Path:
        """Get the directory for a specific epoch, optionally creating it."""
        epoch_dir = self.run_dir / f"epoch_{epoch}"
        if create:
            epoch_dir.mkdir(exist_ok=True)
        return epoch_dir

    def _get_artifact_path[T: Artifact](
        self, epoch: int, artifact_class: type[T]
    ) -> Path:
        """Get the path for an artifact file."""
        artifacts_dir = self._get_epoch_dir(epoch) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        return artifacts_dir / f"{to_snake_case(artifact_class.__name__)}.joblib"

    def _get_plot_path[T: Artifact](self, epoch: int, artifact_class: type[T]) -> Path:
        """Get the path for an artifact plot."""
        plots_dir = self._get_epoch_dir(epoch) / "plots"
        plots_dir.mkdir(exist_ok=True)
        return plots_dir / f"{to_snake_case(artifact_class.__name__)}.png"

    def _get_params_path(self, epoch: int) -> Path:
        """Get the path for parameter files."""
        params_dir = self._get_epoch_dir(epoch)
        return params_dir / "params.joblib"
