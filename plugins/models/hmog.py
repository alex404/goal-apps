"""Base class for HMoG implementations."""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypedDict, override

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from goal.geometry import (
    Diagonal,
    Natural,
    Optimizer,
    OptState,
    Point,
    PositiveDefinite,
    Scale,
)
from goal.models import (
    DifferentiableHMoG,
    DifferentiableMixture,
    FullNormal,
    LinearGaussianModel,
    Normal,
    differentiable_hmog,
)
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.figure import Figure
from omegaconf import MISSING

from apps.configs import ClusteringModelConfig
from apps.plugins import (
    ClusteringDataset,
    ClusteringModel,
)
from apps.runtime.handler import JSONDict, RunHandler
from apps.runtime.logger import ArrayArtifact, Artifact, JaxLogger

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


# Specify config
class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


@dataclass
class HMoGConfig(ClusteringModelConfig):
    """Configuration for Hierarchical Mixture of Gaussians model.

    Model Architecture:
        latent_dim: Dimension of latent space [default: 10]
        n_clusters: Number of mixture components [default: 10]
        data_dim: Dimension of input data [set by dataset]
        obs_rep: Representation type for observations. Options: scale, diagonal, positive_definite [default: diagonal]
        lat_rep: Representation type for latents. Options: scale, diagonal, positive_definite [default: diagonal]

    Training Parameters:
        batch_size: Batch size for stage 3 [default: 256]
        stage1_epochs: Number of epochs for EM initialization [default: 100]
        stage2_epochs: Number of epochs for mixture component training [default: 100]
        stage3_epochs: Number of epochs for full model training [default: 100]
        stage2_learning_rate: Learning rate for stage 2 [default: 0.001]
        stage3_learning_rate: Learning rate for stage 3 [default: 0.0003]
    """

    _target_: str = "plugins.models.hmog.HMoGExperiment"
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10
    obs_rep: RepresentationType = RepresentationType.diagonal
    lat_rep: RepresentationType = RepresentationType.diagonal
    batch_size: int = 256
    stage1_epochs: int = 100
    stage2_epochs: int = 100
    stage3_epochs: int = 100
    stage2_learning_rate: float = 1e-3
    stage3_learning_rate: float = 3e-4


# Register config
cs = ConfigStore.instance()
cs.store(group="model", name="hmog", node=HMoGConfig)

# Globals
OBS_JITTER = 1e-7
OBS_MIN_VAR = 1e-6


### Helper Functions ###


def fori[X](lower: int, upper: int, body_fun: Callable[[int, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


# Artifacts


@dataclass(frozen=True)
class Prototypes(Artifact):
    prototypes: list[Array]

    @override
    def to_json(self) -> JSONDict:
        return {
            "prototypes": [p.tolist() for p in self.prototypes],
        }


# Artifact Creation


def get_component_divergences[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> ArrayArtifact:
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
    return ArrayArtifact(jax.vmap(kl_div_from_one_to_all)(idxs))


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


# Artifact plots


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
    divergences: ArrayArtifact,
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


### Stabilizers ###


def bound_mixture_probabilities[Rep: PositiveDefinite](
    model: DifferentiableMixture[FullNormal, Normal[Rep]],
    params: Point[Natural, DifferentiableMixture[FullNormal, Normal[Rep]]],
    min_prob: float = 1e-3,
) -> Point[Natural, DifferentiableMixture[FullNormal, Normal[Rep]]]:
    """Bound mixture probabilities away from 0."""
    comps, cat_params = model.split_natural_mixture(params)

    with model.lat_man as lm:
        cat_means = lm.to_mean(cat_params)
        probs = lm.to_probs(cat_means)
        bounded_probs = jnp.clip(probs, min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_cat_params = lm.to_natural(lm.from_probs(bounded_probs))

    return model.join_natural_mixture(comps, bounded_cat_params)


def bound_hmog_mixture_probabilities[
    ObsRep: PositiveDefinite,
    LatRep: PositiveDefinite,
](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    min_prob: float = 1e-3,
) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
    """Bound mixture probabilities away from 0."""
    lkl_params, mix_params = model.split_conjugated(params)
    bounded_mix_params = bound_mixture_probabilities(
        model.upr_hrm, mix_params, min_prob
    )
    return model.join_conjugated(lkl_params, bounded_mix_params)


### ABC ###


class HMoGMetrics(TypedDict):
    train_ll: Array
    test_ll: Array
    train_average_bic: Array


class HMoGExperiment[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    ClusteringModel
):
    """Experiment framework for HMoGs."""

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        obs_rep: RepresentationType,
        lat_rep: RepresentationType,
        batch_size: int,
        stage1_epochs: int,
        stage2_epochs: int,
        stage3_epochs: int,
        stage2_learning_rate: float,
        stage3_learning_rate: float,
    ) -> None:
        self.batch_size: int = batch_size
        self.stage1_epochs: int = stage1_epochs
        self.stage2_epochs: int = stage2_epochs
        self.stage3_epochs: int = stage3_epochs
        self.stage2_learning_rate: float = stage2_learning_rate
        self.stage3_learning_rate: float = stage3_learning_rate

        obs_rep_type = obs_rep.value
        lat_rep_type = lat_rep.value

        self.model: DifferentiableHMoG[ObsRep, LatRep] = differentiable_hmog(  # pyright: ignore[reportAttributeAccessIssue]
            obs_dim=data_dim,
            obs_rep=obs_rep_type,
            lat_dim=latent_dim,
            n_components=n_clusters,
            lat_rep=lat_rep_type,
        )

        log.info(f"Initialized HMoG model with dimension {self.model.dim}.")

    """Base class for HMoG implementations."""

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        return self.stage1_epochs + self.stage2_epochs + self.stage3_epochs

    @property
    def latent_dim(self) -> int:
        return self.model.lat_man.obs_man.data_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

    # Methods

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize model parameters."""
        keys = jax.random.split(key, 3)
        key_cat, key_comp, key_int = keys

        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_means = self.model.obs_man.regularize_covariance(
            obs_means, OBS_JITTER, OBS_MIN_VAR
        )
        obs_params = self.model.obs_man.to_natural(obs_means)

        with self.model.upr_hrm as uh:
            cat_params = uh.lat_man.initialize(key_cat)
            key_comps = jax.random.split(key_comp, self.n_clusters)
            component_list = [
                uh.obs_man.initialize(key_compi).array for key_compi in key_comps
            ]
            components = jnp.stack(component_list)
            mix_params = uh.join_natural_mixture(
                uh.comp_man.natural_point(components), cat_params
            )

        int_noise = 0.1 * jax.random.normal(key_int, self.model.int_man.shape)
        lkl_params = self.model.lkl_man.join_params(
            obs_params,
            self.model.int_man.point(self.model.int_man.rep.from_dense(int_noise)),
        )
        return self.model.join_conjugated(lkl_params, mix_params).array

    def log_likelihood(
        self, params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], data: Array
    ) -> Array:
        return self.model.average_log_observable_density(params, data)

    @override
    def generate(
        self,
        params: Array,
        key: Array,
        n_samples: int,
    ) -> Array:
        return self.model.observable_sample(
            key, self.model.natural_point(params), n_samples
        )

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        def data_point_cluster(x: Array) -> Array:
            with self.model as m:
                cat_pst = m.lat_man.prior(m.posterior_at(m.natural_point(params), x))
                with m.lat_man.lat_man as lm:
                    probs = lm.to_probs(lm.to_mean(cat_pst))
            return jnp.argmax(probs, axis=-1)

        return jax.vmap(data_point_cluster)(data)

    def log_epoch_metrics(
        self,
        logger: JaxLogger,
        epoch: int,
        hmog_params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        train_sample: Array,
        test_sample: Array,
        log_freq: int = 1,
    ) -> None:
        """Log metrics for an epoch."""

        def compute_metrics():
            epoch_train_ll = self.model.average_log_observable_density(
                hmog_params, train_sample
            )
            epoch_test_ll = self.model.average_log_observable_density(
                hmog_params, test_sample
            )

            n_samps = train_sample.shape[0]
            epoch_train_bic = (
                self.model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll
            )
            metrics = HMoGMetrics(
                train_ll=epoch_train_ll,
                test_ll=epoch_test_ll,
                train_average_bic=epoch_train_bic,
            )
            logger.log_metrics({k: metrics[k] for k in metrics}, epoch + 1)

        def no_op():
            return None

        jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)

    def log_figures(
        self,
        handler: RunHandler,
        logger: JaxLogger,
        dataset: ClusteringDataset,
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        epoch: int,
    ) -> None:
        """Log artifacts - only run a couple times per training."""
        plot_prototypes = prototypes_plotter(dataset)

        prototypes = get_component_prototypes(self.model, params)
        divergences = get_component_divergences(self.model, params)

        logger.log_artifact(
            handler, epoch, "component_prototypes", prototypes, plot_prototypes
        )
        logger.log_artifact(
            handler, epoch, "component_divergences", divergences, plot_divergence_matrix
        )

    @override
    def run_experiment(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        init_key, fit_key = jax.random.split(key)
        params = self.initialize_model(init_key, dataset.train_data)
        final_params = self.fit(
            fit_key, handler, dataset, logger, self.model.natural_point(params)
        )
        handler.save_json(final_params.array.tolist(), "final_params")

    def fit(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
        init_params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Three-stage minibatch training process."""

        train_sample = dataset.train_data
        test_sample = dataset.test_data

        lkl_params0, mix_params0 = self.model.split_conjugated(init_params)

        self.log_epoch_metrics(logger, -1, init_params, train_sample, test_sample)

        # Stage 1: Full-batch EM for LinearGaussianModel
        with self.model.lwr_hrm as lh:
            z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
            lgm_params0 = lh.join_conjugated(lkl_params0, z)

            def stage1_step(
                epoch: int, lgm_params: Point[Natural, LinearGaussianModel[ObsRep]]
            ) -> Point[Natural, LinearGaussianModel[ObsRep]]:
                means = lh.expectation_step(lgm_params, train_sample)
                obs_means, int_means, lat_means = lh.split_params(means)
                obs_means = lh.obs_man.regularize_covariance(
                    obs_means, OBS_JITTER, OBS_MIN_VAR
                )
                means = lh.join_params(obs_means, int_means, lat_means)
                params1 = lh.to_natural(means)
                lkl_params = lh.likelihood_function(params1)
                lgm_params = lh.join_conjugated(lkl_params, z)
                hmog_params = self.model.join_conjugated(lkl_params, mix_params0)
                self.log_epoch_metrics(
                    logger, epoch, hmog_params, train_sample, test_sample
                )
                return lgm_params

            lgm_params1 = fori(0, self.stage1_epochs, stage1_step, lgm_params0)
            lkl_params1 = lh.likelihood_function(lgm_params1)

        # Stage 2: Gradient descent for mixture components
        stage2_optimizer: Optimizer[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ] = Optimizer.adam(self.model.upr_hrm, learning_rate=self.stage2_learning_rate)
        stage2_opt_state = stage2_optimizer.init(mix_params0)

        def stage2_loss(
            params: Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            batch: Array,
        ) -> Array:
            hmog_params = self.model.join_conjugated(lkl_params1, params)
            return -self.model.average_log_observable_density(hmog_params, batch)

        def stage2_minibatch_step(
            carry: tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            ],
            batch: Array,
        ) -> tuple[
            tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            ],
            None,
        ]:
            opt_state, params = carry
            grad = self.model.upr_hrm.grad(lambda p: stage2_loss(p, batch), params)

            opt_state, params = stage2_optimizer.update(opt_state, grad, params)
            params = bound_mixture_probabilities(self.model.upr_hrm, params)
            return ((opt_state, params), None)

        def stage2_epoch(
            epoch: int,
            carry: tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
                Array,
            ],
        ) -> tuple[
            OptState,
            Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            Array,
        ]:
            opt_state, params, key = carry

            # Shuffle data and truncate to fit batches evenly
            return_key, shuffle_key = jax.random.split(key)
            n_complete_batches = train_sample.shape[0] // self.batch_size
            n_samples_to_use = n_complete_batches * self.batch_size

            shuffled_indices = jax.random.permutation(
                shuffle_key, train_sample.shape[0]
            )[:n_samples_to_use]
            batched_data = train_sample[shuffled_indices].reshape(
                (n_complete_batches, self.batch_size, -1)
            )

            # Process batches
            (opt_state, params), _ = jax.lax.scan(
                stage2_minibatch_step,
                (opt_state, params),
                batched_data,
                None,
            )

            # Compute full dataset likelihood
            hmog_params = self.model.join_conjugated(lkl_params1, params)

            self.log_epoch_metrics(
                logger, epoch, hmog_params, train_sample, test_sample, log_freq=10
            )

            return (opt_state, params, return_key)

        (_, mix_params1, key) = fori(
            self.stage1_epochs,
            self.stage1_epochs + self.stage2_epochs,
            stage2_epoch,
            (stage2_opt_state, mix_params0, key),
        )

        # Stage 3: Similar structure to stage 2
        params1 = self.model.join_conjugated(lkl_params1, mix_params1)
        self.log_figures(
            handler, logger, dataset, params1, self.stage1_epochs + self.stage2_epochs
        )

        stage3_optimizer: Optimizer[Natural, DifferentiableHMoG[ObsRep, LatRep]] = (
            Optimizer.adam(self.model.upr_hrm, learning_rate=self.stage3_learning_rate)
        )
        stage3_opt_state = stage3_optimizer.init(params1)

        def stage3_loss(
            params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
            batch: Array,
        ) -> Array:
            return -self.model.average_log_observable_density(params, batch)

        def stage3_minibatch_step(
            carry: tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]],
            batch: Array,
        ) -> tuple[
            tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]], None
        ]:
            opt_state, params = carry
            grad = self.model.grad(lambda p: stage3_loss(p, batch), params)
            opt_state, params = stage3_optimizer.update(opt_state, grad, params)
            params = bound_hmog_mixture_probabilities(self.model, params)
            return (opt_state, params), None

        def stage3_epoch(
            epoch: int,
            carry: tuple[
                OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array
            ],
        ) -> tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array]:
            opt_state, params, key = carry

            # Shuffle and batch data
            return_key, shuffle_key = jax.random.split(key)
            n_complete_batches = train_sample.shape[0] // self.batch_size
            n_samples_to_use = n_complete_batches * self.batch_size

            shuffled_indices = jax.random.permutation(
                shuffle_key, train_sample.shape[0]
            )[:n_samples_to_use]
            batched_data = train_sample[shuffled_indices].reshape(
                (n_complete_batches, self.batch_size, -1)
            )

            # Process batches
            (opt_state, params), _ = jax.lax.scan(
                stage3_minibatch_step,
                (opt_state, params),
                batched_data,
            )

            self.log_epoch_metrics(
                logger, epoch, params, train_sample, test_sample, log_freq=10
            )

            return opt_state, params, return_key

        (_, final_params, _) = fori(
            self.stage1_epochs + self.stage2_epochs,
            self.n_epochs,
            stage3_epoch,
            (stage3_opt_state, params1, key),
        )
        self.log_figures(handler, logger, dataset, final_params, self.n_epochs)
        return final_params
