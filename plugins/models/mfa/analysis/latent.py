"""Latent space projection analysis for MFA model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from matplotlib.figure import Figure

from apps.interface import Analysis, ClusteringDataset
from apps.runtime import Artifact, RunHandler

from .base import get_responsibilities

if TYPE_CHECKING:
    from ..model import MFAModel


@dataclass(frozen=True)
class LatentProjections(Artifact):
    """Latent factor projections of data."""

    latent_means: Array
    """Posterior latent means of shape (n_samples, latent_dim)."""

    cluster_assignments: Array
    """Cluster assignments of shape (n_samples,)."""

    true_labels: Array | None
    """True labels if available, of shape (n_samples,)."""


@dataclass(frozen=True)
class LatentProjectionsAnalysis(
    Analysis[ClusteringDataset, "MFAModel", LatentProjections]
):
    """Analysis for projecting data into latent factor space."""

    @property
    @override
    def artifact_type(self) -> type[LatentProjections]:
        """Return artifact type."""
        return LatentProjections

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: "MFAModel",
        epoch: int,
        params: Array,
    ) -> LatentProjections:
        """Compute latent projections of the training data.

        Args:
            key: Random key
            handler: Run handler for loading params
            dataset: Dataset to project
            model: MFA model instance
            epoch: Epoch number
            params: Model parameters

        Returns:
            LatentProjections artifact
        """
        # Get cluster assignments
        responsibilities = get_responsibilities(model.mfa, params, dataset.train_data)
        assignments = jnp.argmax(responsibilities, axis=1)

        # Project to latent space (posterior means)
        def get_latent_mean(x: Array) -> Array:
            """Get posterior latent mean for one sample."""
            # Compute posterior in latent space
            posterior_params = model.mfa.posterior_at(params, x)

            # Extract observable part from CompleteMixture
            # CompleteMixture structure: (observable_params, interaction_params, categorical_params)
            lat_obs, _, _ = model.mfa.lat_man.split_coords(posterior_params)

            # lat_obs is in the observable space of the latent mixture (Normal parameters)
            # Convert to mean coordinates to get the latent mean
            return model.mfa.lat_man.obs_man.to_mean(lat_obs)

        latent_means = jax.vmap(get_latent_mean)(dataset.train_data)

        true_labels = getattr(dataset, "train_labels", None)

        return LatentProjections(
            latent_means=latent_means,
            cluster_assignments=assignments,
            true_labels=true_labels,
        )

    @override
    def plot(
        self,
        artifact: LatentProjections,
        dataset: ClusteringDataset,
    ) -> Figure:
        """Plot 2D latent space visualization.

        Args:
            artifact: Latent projections to visualize
            dataset: Dataset

        Returns:
            Matplotlib figure
        """
        fig, axes = (
            plt.subplots(1, 2, figsize=(12, 5))
            if artifact.true_labels is not None
            else (plt.subplots(1, 1, figsize=(6, 5))[0], [plt.gca(), None])
        )
        ax1, ax2 = axes if isinstance(axes, tuple) else (axes, None)

        # Plot colored by cluster assignment
        scatter1 = ax1.scatter(
            artifact.latent_means[:, 0],
            artifact.latent_means[:, 1],
            c=artifact.cluster_assignments,
            cmap="tab10",
            alpha=0.5,
            s=1,
        )
        ax1.set_title("Latent Space (by cluster)")
        ax1.set_xlabel("Latent Dimension 1")
        ax1.set_ylabel("Latent Dimension 2")
        plt.colorbar(scatter1, ax=ax1, label="Cluster")

        # Plot colored by true labels
        if ax2 is not None:
            scatter2 = ax2.scatter(
                artifact.latent_means[:, 0],
                artifact.latent_means[:, 1],
                c=artifact.true_labels,
                cmap="tab10",
                alpha=0.5,
                s=1,
            )
            ax2.set_title("Latent Space (by true label)")
            ax2.set_xlabel("Latent Dimension 1")
            ax2.set_ylabel("Latent Dimension 2")
            plt.colorbar(scatter2, ax=ax2, label="True Label")

        plt.tight_layout()
        return fig
