"""Base class for HMoG implementations."""

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, TypedDict, cast, override

import jax
import jax.numpy as jnp
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
from omegaconf import MISSING

from apps.clustering.plugins import (
    ClusteringDataset,
    ClusteringModel,
)
from apps.configs import ClusteringModelConfig
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger


class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


def fori[X](lower: int, upper: int, body_fun: Callable[[int, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


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
        stage1_epochs: Number of epochs for EM initialization [default: 100]
        stage2_epochs: Number of epochs for mixture component training [default: 100]
        stage3_epochs: Number of epochs for full model training [default: 100]
        stage2_batch_size: Batch size for stage 2 [default: 256]
        stage3_batch_size: Batch size for stage 3 [default: 256]
        stage2_learning_rate: Learning rate for stage 2 [default: 0.001]
        stage3_learning_rate: Learning rate for stage 3 [default: 0.0003]
    """

    _target_: str = "plugins.models.hmog.HMoGExperiment"
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10
    obs_rep: RepresentationType = RepresentationType.diagonal
    lat_rep: RepresentationType = RepresentationType.diagonal
    stage1_epochs: int = 100
    stage2_epochs: int = 100
    stage3_epochs: int = 100
    stage2_batch_size: int = 256
    stage3_batch_size: int = 256
    stage2_learning_rate: float = 1e-3
    stage3_learning_rate: float = 3e-4


# Register config
cs = ConfigStore.instance()
cs.store(group="model", name="hmog", node=HMoGConfig)


OBS_JITTER = 1e-7
OBS_MIN_VAR = 1e-6
LAT_JITTER = 1e-6
LAT_MIN_VAR = 1e-4


### Helper routines ###


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
    train_bic: Array


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
        stage1_epochs: int,
        stage2_epochs: int,
        stage3_epochs: int,
        stage2_batch_size: int,
        stage3_batch_size: int,
        stage2_learning_rate: float,
        stage3_learning_rate: float,
    ) -> None:
        self.stage1_epochs: int = stage1_epochs
        self.stage2_epochs: int = stage2_epochs
        self.stage3_epochs: int = stage3_epochs
        self.stage2_learning_rate: float = stage2_learning_rate
        self.stage3_learning_rate: float = stage3_learning_rate
        self.stage2_batch_size: int = stage2_batch_size
        self.stage3_batch_size: int = stage3_batch_size

        obs_rep_type = obs_rep.value
        lat_rep_type = lat_rep.value

        self.model: DifferentiableHMoG[ObsRep, LatRep] = differentiable_hmog(  # pyright: ignore[reportAttributeAccessIssue]
            obs_dim=data_dim,
            obs_rep=obs_rep_type,
            lat_dim=latent_dim,
            n_components=n_clusters,
            lat_rep=lat_rep_type,
        )

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
            components = [uh.obs_man.initialize(key_compi) for key_compi in key_comps]
            mix_params = uh.join_natural_mixture(components, cat_params)

        int_noise = 0.1 * jax.random.normal(key_int, self.model.int_man.shape)
        lkl_params = self.model.lkl_man.join_params(
            obs_params, Point(self.model.int_man.rep.from_dense(int_noise))
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
        return self.model.observable_sample(key, Point(params), n_samples)

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        def data_point_cluster(x: Array) -> Array:
            with self.model as m:
                cat_pst = m.lat_man.prior(m.posterior_at(Point(params), x))
                with m.lat_man.lat_man as lm:
                    probs = lm.to_probs(lm.to_mean(cat_pst))
            return jnp.argmax(probs, axis=-1)

        return jax.vmap(data_point_cluster)(data)

    @override
    def run_experiment(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        params = self.initialize_model(key, dataset.train_images)
        final_params = self.fit(  # pyright: ignore[reportUnknownVariableType]
            logger, Point(params), dataset.train_images, dataset.test_images
        )
        final_params = cast(
            Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], final_params
        )
        handler.save_json(final_params.array.tolist(), "final_params")

    # Add this method to the HMoGBase class
    @override
    def get_component_prototypes(
        self,
        params: Array,
    ) -> Array:
        r"""Extract the mean image for each mixture component.

        For HMoG models, the prototype for each component k is:
        $$
        \mu_k = A_k \mu_k^z + b_k
        $$
        where $A_k, b_k$ are the linear transformation parameters for component k,
        and $\mu_k^z$ is the mean of the latent distribution for component k.

        Returns:
            Array of shape (n_components, obs_dim) containing the mean
            observation for each mixture component.
        """
        # Split into likelihood and mixture parameters
        lkl_params, mix_params = self.model.split_conjugated(Point(params))

        # Extract components from mixture
        comp_lats, _ = self.model.upr_hrm.split_natural_mixture(mix_params)

        # For each component, compute the observable distribution and get its mean
        prototypes = []
        for comp_lat_params in comp_lats:
            # Get latent mean for this component
            with self.model.lwr_hrm as lh:
                lwr_hrm_params = lh.join_conjugated(lkl_params, comp_lat_params)
                lwr_hrm_means = lh.to_mean(lwr_hrm_params)
                lwr_hrm_obs = lh.split_params(lwr_hrm_means)[0]
                obs_means = lh.obs_man.split_mean_second_moment(lwr_hrm_obs)[0].array

            prototypes.append(obs_means)

        return jnp.stack(prototypes)

    def log_epoch_metrics(
        self,
        logger: JaxLogger,
        epoch: int,
        hmog_params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        train_sample: Array,
        test_sample: Array,
    ) -> None:
        """Log metrics for an epoch."""

        epoch_train_ll = self.model.average_log_observable_density(
            hmog_params, train_sample
        )
        epoch_test_ll = self.model.average_log_observable_density(
            hmog_params, test_sample
        )

        epoch_train_bic = -2 * epoch_train_ll + self.model.dim * jnp.log(
            train_sample.shape[0]
        )
        metrics = HMoGMetrics(
            train_ll=epoch_train_ll,
            test_ll=epoch_test_ll,
            train_bic=epoch_train_bic,
        )

        logger.log_metrics({k: metrics[k] for k in metrics}, epoch)

    # def log_prototypes(
    #     self,
    #     handler: RunHandler,
    #     dataset: Dataset,
    #     params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    # ) -> None:
    #     """Log component prototypes using dataset visualization."""
    #     prototypes = self.get_component_prototypes(params)
    #
    #     for i, prototype in enumerate(prototypes):
    #         fig, ax = plt.subplots()
    #         dataset.visualize_observable(prototype, ax=ax)
    #
    #         # handler.log_image(
    #         #     f"prototypes/prototype_{i}", fig, f"Component {i} prototype"
    #         # )
    #         plt.close(fig)
    #

    # @override
    @partial(jax.jit, static_argnums=(0, 1))
    def fit(
        self,
        logger: JaxLogger,
        init_params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        train_sample: Array,
        test_sample: Array,
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Three-stage minibatch training process."""

        self.log_epoch_metrics(logger, 0, init_params, train_sample, test_sample)

        lkl_params0, mix_params0 = self.model.split_conjugated(init_params)
        key = jax.random.PRNGKey(0)

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

            lgm_params1 = fori(1, self.stage1_epochs + 1, stage1_step, lgm_params0)
            lkl_params1 = lh.likelihood_function(lgm_params1)

        # Stage 2: Gradient descent for mixture components
        stage2_optimizer: Optimizer[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ] = Optimizer.adam(learning_rate=self.stage2_learning_rate)
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
            n_complete_batches = train_sample.shape[0] // self.stage2_batch_size
            n_samples_to_use = n_complete_batches * self.stage2_batch_size

            shuffled_indices = jax.random.permutation(
                shuffle_key, train_sample.shape[0]
            )[:n_samples_to_use]
            batched_data = train_sample[shuffled_indices].reshape(
                (n_complete_batches, self.stage2_batch_size, -1)
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
                logger, epoch, hmog_params, train_sample, test_sample
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
        # self.log_prototypes(handler, dataset, params1)

        stage3_optimizer: Optimizer[Natural, DifferentiableHMoG[ObsRep, LatRep]] = (
            Optimizer.adam(learning_rate=self.stage3_learning_rate)
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
            n_complete_batches = train_sample.shape[0] // self.stage3_batch_size
            n_samples_to_use = n_complete_batches * self.stage3_batch_size

            shuffled_indices = jax.random.permutation(
                shuffle_key, train_sample.shape[0]
            )[:n_samples_to_use]
            batched_data = train_sample[shuffled_indices].reshape(
                (n_complete_batches, self.stage3_batch_size, -1)
            )

            # Process batches
            (opt_state, params), _ = jax.lax.scan(
                stage3_minibatch_step,
                (opt_state, params),
                batched_data,
            )

            self.log_epoch_metrics(logger, epoch, params, train_sample, test_sample)

            return opt_state, params, return_key

        (_, final_params, _) = fori(
            self.stage1_epochs + self.stage2_epochs,
            self.n_epochs,
            stage3_epoch,
            (stage3_opt_state, params1, key),
        )
        # self.log_prototypes(handler, dataset, final_params)
        return final_params
