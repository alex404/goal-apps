"""Shared utilities for GOAL examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

### Loggers ###


### Path and IO Handler ###

# Define the type recursively
type JSONDict = dict[str, JSONValue]
type JSONList = list[JSONValue]
type JSONPrimitive = str | int | float | bool | None

type JSONValue = JSONDict | JSONList | JSONPrimitive


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
        run_dir = (
            base / "sweep" / sweep_id / name if sweep_id else base / "single" / name
        )

        self.run_dir = run_dir

        run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".cache"

    def save_json(self, results: JSONValue, name: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = self.run_dir / f"{name}.json"
        with open(analysis_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_json(self, name: str) -> JSONValue:
        analysis_path = self.run_dir / f"{name}.json"
        with open(analysis_path) as f:
            return json.load(f)

    def save_plot(self, fig: Figure, name: str = "plot") -> None:
        plot_path = self.run_dir / f"{name}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
