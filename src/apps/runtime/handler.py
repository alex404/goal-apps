"""Shared utilities for GOAL examples."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
from jax import Array
from matplotlib.figure import Figure

from ..util import to_snake_case

type Metrics = dict[str, list[tuple[int, float]]]

### Loggers ###


### Path and IO Handler ###

# Define the type recursively
type JSONDict = dict[str, Any]
type JSONList = list[Any]
type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONDict | JSONList | JSONPrimitive

# Artifacts


@dataclass(frozen=True)
class Artifact(ABC):
    """Base class for data that can be logged and visualized."""

    @abstractmethod
    def to_json(self) -> JSONValue:
        """Convert artifact data to JSON-serializable format."""

    @classmethod
    @abstractmethod
    def from_json(cls, json_dict: JSONValue) -> Self:
        pass


# Run Handler


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
        return self.project_root / ".cache"

    def _get_epoch_dir(self, epoch: int, create: bool = True) -> Path:
        """Get the directory for a specific epoch, optionally creating it."""
        epoch_dir = self.run_dir / f"epoch_{epoch}"
        if create:
            epoch_dir.mkdir(exist_ok=True)
        return epoch_dir

    def _get_artifact_path(
        self, epoch: int, artifact_class: type[Artifact], suffix: str
    ) -> Path:
        """Get the path for an artifact file with given suffix."""
        epoch_dir = self._get_epoch_dir(epoch)
        return epoch_dir / f"{to_snake_case(artifact_class.__name__)}{suffix}"

    def _save_json(self, data: JSONValue, path: Path) -> None:
        """Save JSON data to a path relative to run_dir."""
        full_path = self.run_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_json(self, path: Path) -> JSONValue:
        """Load JSON data from a path relative to run_dir."""
        full_path = self.run_dir / path
        with open(full_path) as f:
            return json.load(f)

    def save_artifact(self, epoch: int, artifact: Artifact) -> None:
        """Save an artifact at a given epoch."""
        path = self._get_artifact_path(epoch, type(artifact), ".json")
        self._save_json(artifact.to_json(), path)

    def save_artifact_figure(
        self, epoch: int, artifact_class: type[Artifact], fig: Figure
    ) -> None:
        """Save a figure in the same epoch directory as its artifact."""
        path = self._get_artifact_path(epoch, artifact_class, ".png")
        fig.savefig(path, bbox_inches="tight")

    def load_artifact[T: Artifact](self, epoch: int, artifact_class: type[T]) -> T:
        """Load an artifact from a specific epoch."""
        path = self._get_artifact_path(epoch, artifact_class, ".json")
        json_data = self._load_json(path)
        return artifact_class.from_json(json_data)

    def save_params(self, epoch: int, params: Array) -> None:
        """Save parameters at a given epoch."""
        params_list = params.tolist()
        path = self._get_epoch_dir(epoch) / "params.json"
        self._save_json(params_list, path)

    def load_params(self, epoch: int) -> Array:
        """Load parameters from a specific epoch."""
        path = self._get_epoch_dir(epoch, create=False) / "params.json"
        params_list = self._load_json(path)
        return jax.numpy.array(params_list)

    def save_metrics(self, metrics: Metrics) -> None:
        """Save training metrics."""
        self._save_json(metrics, Path("metrics.json"))

    def save_metrics_figure(self, fig: Figure) -> None:
        """Save the metrics summary figure."""
        path = self.run_dir / "metrics.png"
        fig.savefig(path, bbox_inches="tight")

    def load_metrics(self) -> Metrics:
        """Load training metrics."""
        return self._load_json(Path("metrics.json"))  # pyright: ignore[reportReturnType]

    def get_available_epochs(self) -> list[int]:
        """Get all epochs where we have analysis results."""
        return sorted(
            [
                int(d.name.split("_")[1])
                for d in self.run_dir.glob("epoch_*")
                if d.is_dir()
            ]
        )
