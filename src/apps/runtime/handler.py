"""Shared utilities for GOAL examples."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import h5py
import numpy as np
import pandas as pd
from jax import Array
from jax import numpy as jnp
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

    @abstractmethod
    def save_to_hdf5(self, file: h5py.File) -> None:
        """Save artifact data to an open HDF5 file."""

    @classmethod
    @abstractmethod
    def load_from_hdf5(cls, file: h5py.File) -> Self:
        """Load artifact data from an open HDF5 file."""


### Run Handler ###


class RunHandler:
    """Handles file management and organization for a single run."""

    name: str
    project_root: Path
    run_dir: Path

    def __init__(self, name: str, project_root: Path):
        self.name = name
        self.project_root = project_root

        # Compute run_dir at init
        base = project_root / "runs"
        sweep_id = os.environ.get("WANDB_SWEEP_ID")
        self.run_dir = (
            base / "sweep" / sweep_id / name if sweep_id else base / "single" / name
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
        self, epoch: int, artifact_class: type[T], suffix: str
    ) -> Path:
        """Get the path for an artifact file with given suffix."""
        epoch_dir = self._get_epoch_dir(epoch)
        return epoch_dir / f"{to_snake_case(artifact_class.__name__)}{suffix}"

    def save_artifact(self, epoch: int, artifact: Artifact) -> None:
        """Save an artifact at a given epoch."""
        path = self._get_artifact_path(epoch, type(artifact), ".h5")
        with h5py.File(path, "w") as f:
            artifact.save_to_hdf5(f)

    def save_artifact_figure(
        self, epoch: int, artifact_class: type[Artifact], fig: Figure
    ) -> None:
        """Save a figure in the same epoch directory as its artifact."""
        path = self._get_artifact_path(epoch, artifact_class, ".png")
        fig.savefig(path, bbox_inches="tight")

    def load_artifact[T: Artifact](self, epoch: int, artifact_class: type[T]) -> T:
        """Load an artifact from a specific epoch."""
        path = self._get_artifact_path(epoch, artifact_class, ".h5")
        with h5py.File(path, "r") as f:
            return artifact_class.load_from_hdf5(f)

    def save_params(
        self, params: Array, epoch: int | None = None, name: str = "params"
    ) -> None:
        """Save parameters at a given epoch."""
        if epoch is None:
            path = self.run_dir / f"{name}.h5"
        else:
            path = self._get_epoch_dir(epoch) / f"{name}.h5"
        with h5py.File(path, "w") as f:
            # Convert JAX array to numpy for storage
            f.create_dataset("params", data=np.array(params))

    def load_params(self, epoch: int | None = None, name: str = "params") -> Array:
        """Load parameters from a specific epoch."""
        if epoch is None:
            path = self.run_dir / f"{name}.h5"
        else:
            path = self._get_epoch_dir(epoch, create=False) / f"{name}.h5"
        with h5py.File(path, "r") as f:
            # Use .get() method which is more likely to be recognized by the type checker
            dataset = f.get("params")
            if dataset is None:
                raise KeyError("params dataset not found in HDF5 file")
            # Convert to numpy array first
            data = np.array(dataset)
            return jnp.array(data)

    def save_metrics(self, metrics: MetricHistory) -> None:
        """Save training metrics to HDF5 using pandas HDFStore."""
        # Convert metrics to pandas DataFrame
        rows: list[dict[str, Any]] = []
        for metric_name, values in metrics.items():
            for epoch, value in values:
                rows.append(
                    {"metric": metric_name, "epoch": int(epoch), "value": float(value)}
                )

        if not rows:
            # Create empty DataFrame if no metrics
            df = pd.DataFrame(columns=["metric", "epoch", "value"])
        else:
            df = pd.DataFrame(rows)

        # Save to HDF5 using pandas
        path = self.run_dir / "metrics.h5"
        df.to_hdf(path, key="metrics", mode="w")

    def save_metrics_figure(self, fig: Figure) -> None:
        """Save the metrics summary figure."""
        path = self.run_dir / "metrics.png"
        fig.savefig(path, bbox_inches="tight")

    def load_metrics(self) -> MetricHistory:
        """Load training metrics from HDF5."""
        path = self.run_dir / "metrics.h5"

        if not path.exists():
            return {}

        df = pd.read_hdf(path, key="metrics")

        # Convert DataFrame back to MetricHistory format
        result: MetricHistory = {}
        for metric_name, group_df in df.groupby("metric"):
            # Ensure metric_name is str
            metric_str = str(metric_name)
            # Sort and convert to list of tuples
            sorted_df = cast(pd.DataFrame, group_df.sort_values("epoch"))  # pyright: ignore[reportCallIssue]
            result[metric_str] = []

            # Iterate through rows safely
            for _, row in sorted_df.iterrows():
                epoch = int(row["epoch"])
                value = float(row["value"])
                result[metric_str].append((epoch, value))

        return result

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
