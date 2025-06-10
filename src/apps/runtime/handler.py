"""Shared utilities for GOAL examples."""

from __future__ import annotations

import os
import re
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from jax import Array
from matplotlib.figure import Figure

type MetricDict = dict[str, tuple[Array, Array]]  # Single snapshot
type MetricHistory = dict[str, list[tuple[int, float]]]  # Time series

### Util ###


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name)
    return name.lower()


### Artifacts ###


@dataclass(frozen=True)
class Artifact(ABC):
    """Base class for data that can be logged and visualized."""


### Run Handler ###


class RunHandler:
    """Handles file management and organization for a single run."""

    name: str
    project_root: Path
    run_dir: Path
    from_epoch: int | None  # None for fresh run
    from_scratch: bool

    def __init__(
        self,
        name: str,
        project_root: Path,
        requested_epoch: int | None,
        from_scratch: bool,
        sweep_id: str | None,
    ):
        self.name = name
        self.project_root = project_root

        # Compute run_dir at init
        base = project_root / "runs"
        if sweep_id is None:
            sweep_id = os.environ.get("WANDB_SWEEP_ID")
        self.run_dir = (
            base / "sweep" / sweep_id / name if sweep_id else base / "single" / name
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Resolve from_epoch
        available = self.get_available_epochs()

        if not available:
            self.from_epoch = None

        elif requested_epoch is None:
            self.from_epoch = max(available)

        else:
            valid_epochs = [e for e in available if e <= requested_epoch]
            self.from_epoch = max(valid_epochs) if valid_epochs else None

        self.from_scratch = from_scratch

    @property
    def cache_dir(self) -> Path:
        """Directory for cached data (e.g., datasets)."""
        return self.project_root / ".cache"

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

    def save_metrics(self, metrics: MetricHistory) -> None:
        """Save training metrics."""
        path = self.run_dir / "metrics.joblib"
        joblib.dump(metrics, path)

    def save_metrics_figure(self, fig: Figure) -> None:
        """Save the metrics summary figure."""
        path = self.run_dir / "metrics.png"
        fig.savefig(path, bbox_inches="tight")

    def load_metrics(self) -> MetricHistory:
        """Load training metrics."""
        path = self.run_dir / "metrics.joblib"

        if not path.exists():
            return {}

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

    def get_available_epochs(self) -> list[int]:
        """Get all epochs where we have analysis results."""
        return sorted(
            [
                int(d.name.split("_")[1])
                for d in self.run_dir.glob("epoch_*")
                if d.is_dir()
            ]
        )
