"""Base class for HMoG implementations."""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import partial
from time import time
from typing import Any, override

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
from jax import Array, debug
from omegaconf import MISSING

from apps.clustering.core.common import ProbabilisticResults, evaluate_clustering
from apps.clustering.core.config import ModelConfig
from apps.clustering.core.datasets import SupervisedDataset
from apps.clustering.core.models import Model


class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


@dataclass
class HMoGConfig(ModelConfig):
    """Configuration for Hierarchical Mixture of Gaussians model."""

    _target_: str = "plugins.models.hmog.MinibatchHMoG"
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


def initialize_gmm_components(
    key: Array, data: Array, nor_man: FullNormal, n_components: int
) -> list[Point[Natural, FullNormal]]:
    debug.print(
        "Data stats - Shape: {}, Mean: {}, Std: {}",
        data.shape,
        jnp.mean(data),
        jnp.std(data),
    )

    keys = jax.random.split(key, 2)
    n_samples = data.shape[0]

    # K-means++ initialization
    def kmeans_plus_plus(key: Array):
        centers = [data[jax.random.randint(key, (), 0, n_samples)]]
        for _ in range(n_components - 1):
            dists = jnp.min(
                jnp.array(
                    [jnp.sum((data - center) ** 2, axis=1) for center in centers]
                ),
                axis=0,
            )

            probs = dists / jnp.sum(dists)
            next_idx = jax.random.choice(key, n_samples, p=probs)
            centers.append(data[next_idx])

        return jnp.stack(centers)

    means = kmeans_plus_plus(keys[0])
    debug.print(
        "Means shape: {}, stats: {}", means.shape, [jnp.mean(means), jnp.std(means)]
    )

    def estimate_local_cov(center: Array):
        dists = jnp.sum((data - center) ** 2, axis=1)
        nearest = jnp.argsort(dists)[: n_samples // n_components]
        local_data = data[nearest]
        return jnp.cov(local_data.T) + jnp.eye(data.shape[1]) * 1e-6

    covs = jax.vmap(estimate_local_cov)(means)
    debug.print(
        "Covs shape: {}, eigenvalue ranges: {}",
        covs.shape,
        [
            (jnp.min(jnp.linalg.eigvalsh(c)), jnp.max(jnp.linalg.eigvalsh(c)))
            for c in covs
        ],
    )

    nrms: list[Point[Natural, FullNormal]] = []
    for i in range(n_components):
        mean = nor_man.loc_man.mean_point(means[i])
        cov = nor_man.cov_man.from_dense(covs[i])
        nrm = nor_man.to_natural(nor_man.join_mean_covariance(mean, cov))
        debug.print(
            "Component {} natural params - Shape: {}, Range: [{}, {}]",
            i,
            nrm.array.shape,
            jnp.min(nrm.array),
            jnp.max(nrm.array),
        )
        nrms.append(nrm)

    return nrms


### ABC ###


class HMoGBase[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    Model[Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]], ABC
):
    """Base class for HMoG implementations."""

    # Fields

    model: DifferentiableHMoG[ObsRep, LatRep]
    n_epochs: int

    # Properties

    @property
    @override
    def latent_dim(self) -> int:
        return self.model.lat_man.obs_man.data_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

    # Methods

    @override
    def initialize(
        self, key: Array, data: Array
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
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
        return self.model.join_conjugated(lkl_params, mix_params)

    def log_likelihood(
        self, params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], data: Array
    ) -> Array:
        return self.model.average_log_observable_density(params, data)

    @override
    def generate(
        self,
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        key: Array,
        n_samples: int,
    ) -> Array:
        return self.model.observable_sample(key, params, n_samples)

    @override
    def cluster_assignments(
        self, params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], data: Array
    ) -> Array:
        def data_point_cluster(x: Array) -> Array:
            with self.model as m:
                cat_pst = m.lat_man.prior(m.posterior_at(params, x))
                with m.lat_man.lat_man as lm:
                    probs = lm.to_probs(lm.to_mean(cat_pst))
            return jnp.argmax(probs, axis=-1)

        return jax.vmap(data_point_cluster)(data)

    @override
    def evaluate(
        self,
        key: Array,
        data: SupervisedDataset,
    ) -> ProbabilisticResults:
        start_time = time()
        params = self.initialize(key, data.train_images)
        final_params, train_lls, test_lls = self.fit(
            params, data.train_images, data.test_images
        )

        train_clusters = self.cluster_assignments(final_params, data.train_images)
        test_clusters = self.cluster_assignments(final_params, data.test_images)

        # Get prototypes
        prototypes = self.get_component_prototypes(final_params)

        return ProbabilisticResults(
            model_name=self.__class__.__name__,
            train_log_likelihood=train_lls.tolist(),
            test_log_likelihood=test_lls.tolist(),
            final_train_log_likelihood=float(train_lls[-1]),
            final_test_log_likelihood=float(
                self.log_likelihood(final_params, data.test_images)
            ),
            train_accuracy=float(
                evaluate_clustering(train_clusters, data.train_labels)
            ),
            test_accuracy=float(evaluate_clustering(test_clusters, data.test_labels)),
            latent_dim=self.latent_dim,
            n_clusters=self.n_clusters,
            n_parameters=self.model.dim,
            training_time=time() - start_time,
            prototypes=prototypes.tolist(),  # Add prototypes to results
        )

    # Add this method to the HMoGBase class
    @override
    def get_component_prototypes(
        self,
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
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
        lkl_params, mix_params = self.model.split_conjugated(params)

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


class MinibatchHMoG[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    HMoGBase[ObsRep, LatRep]
):
    """Minibatch training implementation of HMoG."""

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

    @override
    @partial(jax.jit, static_argnums=(0,))
    def fit(
        self,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        train_sample: Array,
        test_sample: Array,
    ) -> tuple[Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array, Array]:
        """Three-stage minibatch training process."""
        lkl_params0, mix_params0 = self.model.split_conjugated(params0)
        key = jax.random.PRNGKey(0)

        # Stage 1: Full-batch EM for LinearGaussianModel
        with self.model.lwr_hrm as lh:
            z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
            lgm_params0 = lh.join_conjugated(lkl_params0, z)

            def stage1_step(
                params: Point[Natural, LinearGaussianModel[ObsRep]], _: Any
            ) -> tuple[
                Point[Natural, LinearGaussianModel[ObsRep]], tuple[Array, Array]
            ]:
                train_ll = lh.average_log_observable_density(params, train_sample)
                test_ll = lh.average_log_observable_density(params, test_sample)
                means = lh.expectation_step(params, train_sample)
                obs_means, int_means, lat_means = lh.split_params(means)
                obs_means = lh.obs_man.regularize_covariance(
                    obs_means, OBS_JITTER, OBS_MIN_VAR
                )
                means = lh.join_params(obs_means, int_means, lat_means)
                params1 = lh.to_natural(means)
                lkl_params = lh.likelihood_function(params1)
                next_params = lh.join_conjugated(lkl_params, z)
                debug.print("Stage 1 Train LL: {}; Test LL: {}", train_ll, test_ll)
                return next_params, (train_ll, test_ll)

            lgm_params1, (train_lls1, test_lls1) = jax.lax.scan(
                stage1_step, lgm_params0, None, length=self.stage1_epochs
            )
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
            carry: tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
                Array,
            ],
            _: Any,
        ) -> tuple[
            tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
                Array,
            ],
            tuple[Array, Array],
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
            epoch_train_ll = self.model.average_log_observable_density(
                hmog_params, train_sample
            )
            epoch_test_ll = self.model.average_log_observable_density(
                hmog_params, test_sample
            )
            debug.print(
                "Stage 2 epoch train LL: {}; test LL: {}", epoch_train_ll, epoch_test_ll
            )

            return (opt_state, params, return_key), (epoch_train_ll, epoch_test_ll)

        (_, mix_params1, key), (train_lls2, test_lls2) = jax.lax.scan(
            stage2_epoch,
            (stage2_opt_state, mix_params0, key),
            None,
            length=self.stage2_epochs,
        )

        # Stage 3: Similar structure to stage 2
        params1 = self.model.join_conjugated(lkl_params1, mix_params1)
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
            carry: tuple[
                OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array
            ],
            _: Any,
        ) -> tuple[
            tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array],
            tuple[Array, Array],
        ]:
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

            # Compute likelihood
            epoch_train_ll = self.model.average_log_observable_density(
                params, train_sample
            )
            epoch_test_ll = self.model.average_log_observable_density(
                params, test_sample
            )
            debug.print(
                "Stage 3 epoch train LL: {}; test LL: {}", epoch_train_ll, epoch_test_ll
            )

            return (opt_state, params, return_key), (epoch_train_ll, epoch_test_ll)

        (_, final_params, _), (train_lls3, test_lls3) = jax.lax.scan(
            stage3_epoch,
            (stage3_opt_state, params1, key),
            None,
            length=self.stage3_epochs,
        )

        train_lls = jnp.concatenate([train_lls1, train_lls2, train_lls3])
        test_lls = jnp.concatenate([test_lls1, test_lls2, test_lls3])
        return final_params, train_lls.ravel(), test_lls.ravel()
