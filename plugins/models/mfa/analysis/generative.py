"""Generative samples analysis for MFA model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import matplotlib.pyplot as plt
from jax import Array
from matplotlib.figure import Figure

from apps.interface import Analysis, ClusteringDataset
from apps.runtime import Artifact, RunHandler

if TYPE_CHECKING:
    from ..model import MFAModel


@dataclass(frozen=True)
class GenerativeSamples(Artifact):
    """Generated samples from the model."""

    samples: Array
    """Generated samples of shape (n_samples, data_dim)."""


@dataclass(frozen=True)
class GenerativeSamplesAnalysis(
    Analysis[ClusteringDataset, "MFAModel", GenerativeSamples]
):
    """Analysis for generating samples from the trained model."""

    n_samples: int = 100
    """Number of samples to generate."""

    @property
    @override
    def artifact_type(self) -> type[GenerativeSamples]:
        """Return artifact type."""
        return GenerativeSamples

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: "MFAModel",
        epoch: int,
        params: Array,
    ) -> GenerativeSamples:
        """Generate samples from the trained model.

        Args:
            key: Random key for sampling
            handler: Run handler for loading params
            dataset: Dataset
            model: MFA model instance
            epoch: Epoch number
            params: Model parameters

        Returns:
            GenerativeSamples artifact
        """
        samples = model.generate(params, key, self.n_samples)
        return GenerativeSamples(samples=samples)

    @override
    def plot(
        self,
        artifact: GenerativeSamples,
        dataset: ClusteringDataset,
    ) -> Figure:
        """Plot grid of generated samples.

        Args:
            artifact: Generated samples to visualize
            dataset: Dataset

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            img = artifact.samples[i].reshape(28, 28)
            ax.imshow(img, cmap="gray")
            ax.axis("off")

        plt.tight_layout()
        return fig
