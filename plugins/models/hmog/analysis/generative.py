"""Base class for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from goal.geometry import (
    Natural,
    Point,
)
from h5py import File
from jax import Array
from matplotlib.figure import Figure

from apps.plugins import (
    Analysis,
    ClusteringDataset,
)
from apps.runtime.handler import Artifact, RunHandler

from ..base import HMoG

### Generative Examples ###


@dataclass(frozen=True)
class GenerativeExamples(Artifact):
    """Collection of generated samples from the model."""

    samples: Array  # Array of shape (n_samples, data_dim)

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save generated samples to HDF5 file."""
        file.create_dataset("samples", data=np.array(self.samples))

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> GenerativeExamples:
        """Load generated samples from HDF5 file."""
        samples = jnp.array(file["samples"][()])  # pyright: ignore[reportIndexIssue,reportArgumentType]
        return cls(samples=samples)


def generate_examples[M: HMoG](
    model: M,
    params: Point[Natural, M],
    n_samples: int,
    key: Array,
) -> GenerativeExamples:
    """Generate sample examples from the model.

    Args:
        model: HMoG model
        params: Model parameters
        n_samples: Number of samples to generate
        key: Random key for sampling

    Returns:
        Collection of generated samples
    """
    samples = model.observable_sample(key, params, n_samples)
    return GenerativeExamples(samples=samples)


def generative_examples_plotter(
    dataset: ClusteringDataset,
) -> Callable[[GenerativeExamples], Figure]:
    """Create a grid of generated samples visualizations."""

    def plot_generative_examples(examples: GenerativeExamples) -> Figure:
        n_samples = min(
            36, examples.samples.shape[0]
        )  # Limit to 36 samples for display

        # Calculate grid dimensions (approximate square grid)
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        height, width = dataset.observable_shape

        # Scale figure size based on observable shape
        fig_width = 2 * grid_size * (width / max(height, width))
        fig_height = 2 * grid_size * (height / max(height, width))

        # Create figure with subplots
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(fig_width, fig_height), squeeze=False
        )

        # Plot each sample
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                ax = axes[i, j]
                if idx < n_samples:
                    dataset.paint_observable(examples.samples[idx], ax)
                else:
                    ax.axis("off")  # Hide empty plots

        plt.suptitle("Generated Samples from HMoG Model", fontsize=14)
        plt.tight_layout()
        return fig

    return plot_generative_examples


### Analysis ###


@dataclass(frozen=True)
class GenerativeExamplesAnalysis(Analysis[ClusteringDataset, HMoG, GenerativeExamples]):
    """Analysis of cluster prototypes with their members."""

    n_samples: int

    @property
    @override
    def artifact_type(self) -> type[GenerativeExamples]:
        return GenerativeExamples

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: HMoG,
        epoch: int,
        params: Array,
    ) -> GenerativeExamples:
        """Generate collection of clusters with their members."""
        # Convert array to typed point for the model
        typed_params = model.natural_point(params)
        return generate_examples(model, typed_params, self.n_samples, key)

    @override
    def plot(self, artifact: GenerativeExamples, dataset: ClusteringDataset) -> Figure:
        """Create grid of cluster prototype visualizations."""
        return generative_examples_plotter(dataset)(artifact)
