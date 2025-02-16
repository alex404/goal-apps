"""Base class for HMoG implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, TypedDict, override

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from goal.geometry import (
    Natural,
    Point,
    PositiveDefinite,
)
from goal.models import (
    DifferentiableHMoG,
)
from jax import Array
from matplotlib.figure import Figure

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import Artifact, JSONDict, JSONValue, RunHandler
from apps.runtime.logger import JaxLogger


class HMoGMetrics(TypedDict):
    train_ll: Array
    test_ll: Array
    train_average_bic: Array


@dataclass(frozen=True)
class Prototypes(Artifact):
    prototypes: list[Array]

    @override
    def to_json(self) -> JSONDict:
        return {
            "prototypes": [p.tolist() for p in self.prototypes],
        }

    @classmethod
    @override
    def from_json(cls, json_dict: JSONDict) -> Prototypes:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls([jnp.array(p) for p in json_dict["prototypes"]])


@dataclass(frozen=True)
class Divergences(Artifact):
    """Artifact wrapping a JAX array."""

    data: Array

    @override
    def to_json(self) -> JSONValue:
        return self.data.tolist()

    @classmethod
    @override
    def from_json(cls, json_dict: JSONValue) -> Divergences:
        return cls(jax.numpy.array(json_dict))


# Artifact Creation


def get_component_divergences[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> Divergences:
    # Split into likelihood and mixture parameters
    mix_params = model.prior(params)

    # Extract components from mixture
    comp_lats, _ = model.upr_hrm.split_natural_mixture(mix_params)

    # For each pair of components, compute the KL divergence

    # Function that computes KL divergence between two components
    def kl_div_between_components(i: Array, j: Array) -> Array:
        # Get the mean parameters for component i
        comp_i = model.upr_hrm.comp_man.get_replicate(comp_lats, i)
        comp_i_mean = model.upr_hrm.obs_man.to_mean(comp_i)

        # Get the natural parameters for component j
        comp_j = model.upr_hrm.comp_man.get_replicate(comp_lats, j)

        # Compute KL divergence between components
        return model.upr_hrm.obs_man.relative_entropy(comp_i_mean, comp_j)

    idxs = jnp.arange(model.upr_hrm.n_categories)

    # Function that computes KL divergence from one component to all others
    def kl_div_from_one_to_all(i: Array) -> Array:
        return jax.vmap(kl_div_between_components, in_axes=(None, 0))(i, idxs)

    # Compute all pairwise KL divergences
    return Divergences(jax.vmap(kl_div_from_one_to_all)(idxs))


def get_component_prototypes[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> Prototypes:
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.upr_hrm.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes = []
    for i in range(comp_lats.shape[0]):
        # Get latent mean for this component
        with model.lwr_hrm as lh:
            comp_lat_params = model.upr_hrm.comp_man.get_replicate(
                comp_lats, jnp.asarray(i)
            )
            lwr_hrm_params = lh.join_conjugated(lkl_params, comp_lat_params)
            lwr_hrm_means = lh.to_mean(lwr_hrm_params)
            lwr_hrm_obs = lh.split_params(lwr_hrm_means)[0]
            obs_means = lh.obs_man.split_mean_second_moment(lwr_hrm_obs)[0].array

        prototypes.append(obs_means)

    return Prototypes(prototypes)


def prototypes_plotter(
    dataset: ClusteringDataset,
) -> Callable[[Prototypes], Figure]:
    def plot_prototypes(prototypes: Prototypes) -> Figure:
        n_prots = len(prototypes.prototypes)

        obs_arts = [dataset.observable_artifact(p) for p in prototypes.prototypes]
        shape = obs_arts[0].shape

        n_cols = math.ceil(math.sqrt(n_prots))
        n_rows = math.ceil(n_prots / n_cols)

        height, width = shape
        wh = (width + height) / 2
        height = 2 * height / wh
        width = 2 * width / wh
        figsize = (width * n_cols, height * n_rows)

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=figsize, constrained_layout=True
        )

        # Handle single subplot case
        if n_prots == 1:
            dataset.paint_observable(obs_arts[0], axes)
            return fig

        # Paint the observables using axes.flat iterator
        for ax, obs_art in zip(axes.flat, obs_arts):
            dataset.paint_observable(obs_art, ax)

        # Remove empty subplots
        for ax in axes.flat[n_prots:]:
            fig.delaxes(ax)

        return fig

    return plot_prototypes


def plot_divergence_matrix(
    divergences: Divergences,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    divs = divergences.data

    im = ax.imshow(divs, cmap="viridis")
    plt.colorbar(im, ax=ax, label="KL Divergence")
    n_rws, _ = divs.shape

    ax.set_xlabel("Component j")
    ax.set_ylabel("Component i")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, n_rws, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rws, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    return fig


def log_artifacts[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    handler: RunHandler,
    dataset: ClusteringDataset,
    logger: JaxLogger,
    model: DifferentiableHMoG[ObsRep, LatRep],
    epoch: int,
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]] | None = None,
) -> None:
    """Generate and save plots from artifacts.

    Args:
        handler: Run handler containing saved artifacts
        dataset: Dataset used for visualization
        logger: Logger for saving artifacts and figures
        model: Model used for analysis and artifact generation
        params: If provided, generate new artifacts from these parameters
        epoch: Specific epoch to analyze, defaults to latest
    """
    # from_scratch if params is provided
    if params is not None:
        prototypes = get_component_prototypes(model, params)
        divergences = get_component_divergences(model, params)
        handler.save_params(epoch, params.array)
    else:
        prototypes = handler.load_artifact(epoch, Prototypes)
        divergences = handler.load_artifact(epoch, Divergences)

    # Plot and save
    plot_prototypes = prototypes_plotter(dataset)
    logger.log_artifact(handler, epoch, prototypes, plot_prototypes)
    logger.log_artifact(handler, epoch, divergences, plot_divergence_matrix)
