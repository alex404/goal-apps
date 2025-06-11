"""Shared utilities for GOAL examples."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from jax import Array
from matplotlib.figure import Figure

from .util import Artifact, MetricHistory, to_snake_case

### Logging ###

log = logging.getLogger(__name__)

### Run Handler ###


class RunHandler:
    """Handles file management and organization for a single run."""

    # Attributes
    run_name: str
    """Name of the run, used for directory naming."""
    project_root: Path
    """Root directory of the project."""
    run_dir: Path
    """Directory for this specific run, containing all artifacts and logs."""

    from_epoch: int | None
    """Epoch to resume from, or None for a fresh run."""
    from_scratch: bool
    """Whether analysis should start from scratch or used existing artifacts."""

    # Metric buffer for local runs
    _metric_buffer: MetricHistory

    def __init__(
        self,
        run_name: str,
        project_root: Path,
        run_dir: Path,
        requested_epoch: int | None,
        from_scratch: bool,
    ):
        """Initialize the run handler."""

        # Simple attribute assignments
        self.run_name = run_name
        self.project_root = project_root
        self.run_dir = run_dir
        self.from_scratch = from_scratch

        # Resolve from_epoch
        if not self.available_epochs:
            self.from_epoch = None
            if requested_epoch is not None:
                logging.warning(
                    f"Requested epoch {requested_epoch} does not exist. Starting fresh run.",
                )
        elif requested_epoch is None:
            self.from_epoch = max(self.available_epochs)

        else:
            valid_epochs = [e for e in self.available_epochs if e <= requested_epoch]
            self.from_epoch = max(valid_epochs) if valid_epochs else None

        # Initialize metric buffer
        metrics_path = self.run_dir / "metrics.joblib"

        if not metrics_path.exists():
            self._metric_buffer = {}

        else:
            full_metrics = joblib.load(metrics_path)

            # If resuming, only return metrics up to the resume point
            if self.from_epoch is not None:
                self._metric_buffer = {
                    metric_name: [
                        (epoch, value)
                        for epoch, value in values
                        if epoch <= self.from_epoch
                    ]
                    for metric_name, values in full_metrics.items()
                }
            else:
                self._metric_buffer = {}

    ### Public Properties ###

    @property
    def cache_dir(self) -> Path:
        """Directory for cached data (e.g., datasets)."""
        return self.project_root / ".cache"

    @property
    def available_epochs(self) -> list[int]:
        """List of epochs available in this run."""
        return sorted(
            int(d.name.split("_")[1])
            for d in self.run_dir.glob("epoch_*")
            if d.is_dir()
        )

    @property
    def metric_buffer(self) -> MetricHistory:
        """Get the metric buffer for this run."""
        return self._metric_buffer

    ### Public Methods ###

    def save_params(self, params: Array, epoch: int, name: str = "params") -> None:
        """Save parameters at a given epoch."""
        path = self._get_params_path(epoch, name)
        joblib.dump(params, path)

    def load_params(self, name: str = "params") -> Array:
        """Load parameters from the resolved epoch.

        Raises:
            ValueError: If from_epoch is set but no params exist at that epoch
            RuntimeError: If from_epoch is None (shouldn't happen after resolution)
        """
        if self.from_epoch is None:
            raise RuntimeError("Cannot load params: from_epoch is None (fresh run)")

        return joblib.load(self._get_params_path(self.from_epoch, name))

    def save_metrics(self) -> None:
        """Save training metrics."""
        if self._metric_buffer != {}:
            path = self.run_dir / "metrics.joblib"
            joblib.dump(self._metric_buffer, path)

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

    def _get_params_path(self, epoch: int, name: str) -> Path:
        """Get the path for parameter files."""
        params_dir = self._get_epoch_dir(epoch)
        return params_dir / f"{name}.joblib"
