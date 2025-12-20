"""Generative samples analysis for models implementing IsGenerative protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.figure import Figure

from ...runtime import Artifact, RunHandler
from ..analysis import Analysis
from ..dataset import Dataset
from ..model import Model


@dataclass(frozen=True)
class GenerativeSamples(Artifact):
    """Collection of generated samples from the model."""

    samples: Array  # Array of shape (n_samples, data_dim)


@dataclass(frozen=True)
class GenerativeSamplesAnalysis[D: Dataset](Analysis[D, Model[D], GenerativeSamples]):
    """Analysis that generates and visualizes samples from a generative model.

    Works with any model implementing the IsGenerative protocol (generate method).
    """

    n_samples: int = 100

    @property
    @override
    def artifact_type(self) -> type[GenerativeSamples]:
        return GenerativeSamples

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: Any,
        model: Any,
        epoch: int,
        params: Array,
    ) -> GenerativeSamples:
        """Generate samples from the model."""
        samples = model.generate(params, key, self.n_samples)
        return GenerativeSamples(samples=samples)

    @override
    def plot(self, artifact: GenerativeSamples, dataset: Any) -> Figure:
        """Create grid of generated sample visualizations."""
        n_samples = min(36, artifact.samples.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        height, width = dataset.observable_shape

        fig_width = 2 * grid_size * (width / max(height, width))
        fig_height = 2 * grid_size * (height / max(height, width))

        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(fig_width, fig_height), squeeze=False
        )

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                ax = axes[i, j]
                if idx < n_samples:
                    dataset.paint_observable(artifact.samples[idx], ax)
                else:
                    ax.axis("off")

        plt.suptitle("Generated Samples", fontsize=14)
        plt.tight_layout()
        return fig
