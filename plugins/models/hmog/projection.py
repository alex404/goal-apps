"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, override

import jax
import jax.numpy as jnp
import optax
from goal.geometry import (
    Diagonal,
    Mean,
    Natural,
    Optimizer,
    OptState,
    Point,
    PositiveDefinite,
)
from goal.models import AnalyticMixture, FactorAnalysis, Normal, analytic_hmog
from jax import Array

from apps.plugins import (
    ClusteringDataset,
    ClusteringExperiment,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .analysis.logging import AnalysisArgs, log_artifacts, log_epoch_metrics
from .base import HMoG, Mixture, fori
from .trainers import LGMPreTrainer

# Start logger
log = logging.getLogger(__name__)


def to_analytic(
    lgm: FactorAnalysis,
    mix: Mixture,
    sym: HMoG,
    lgm_params: Point[Natural, FactorAnalysis],
    mix_params: Point[Natural, Mixture],
) -> Point[Natural, HMoG]:
    lkl_params = lgm.split_conjugated(lgm_params)[0]
    cmp_params, cat_params = mix.split_natural_mixture(mix_params)
    trg_man = Normal(
        lgm.lat_dim,
        PositiveDefinite,
    )
    pd_cmp_params = mix.cmp_man.man_map(
        lambda p: mix.cmp_man.rep_man.embed_rep(trg_man, p), cmp_params
    )
    pd_mix_params = sym.upr_hrm.join_natural_mixture(pd_cmp_params, cat_params)
    return sym.join_conjugated(lkl_params, pd_mix_params)


@dataclass(frozen=True)
class ProjectionTrainer:
    """Trainer for projection-based training of mixture models.

    This trainer projects data to the latent space using a pre-trained LGM,
    then trains a mixture model on the projected data.
    """

    # Training hyperparameters
    lr: float
    n_epochs: int
    batch_size: None | int = None
    batch_steps: int = 1000
    grad_clip: float = 1.0

    # Regularization parameters
    l1_reg: float = 0.0
    l2_reg: float = 0.0

    # Parameter bounds
    min_prob: float = 1e-4
    lat_min_var: float = 1e-6
    lat_jitter_var: float = 0.0

    def project_data(
        self, lgm: FactorAnalysis, params: Point[Natural, FactorAnalysis], data: Array
    ) -> Array:
        """Project data to the latent space using the LGM."""

        # Get the posterior for each data point
        def posterior_mean(x: Array) -> Array:
            prms = lgm.posterior_at(params, x)
            with lgm.lat_man as lm:
                return lm.split_mean_covariance(lm.to_mean(prms))[0].array

        # Extract the means of the latent variables

        return jax.lax.map(posterior_mean, data, batch_size=256)

    def bound_mixture_means(
        self, model: Mixture, means: Point[Mean, Mixture]
    ) -> Point[Mean, Mixture]:
        """Apply bounds to posterior statistics for numerical stability."""
        # Split the mixture parameters
        comp_means, prob_means = model.split_mean_mixture(means)

        # Bound component means
        bounded_comp_means = model.cmp_man.man_map(
            lambda m: model.obs_man.regularize_covariance(
                m, self.lat_jitter_var, self.lat_min_var
            ),
            comp_means,
        )

        # Bound probabilities
        probs = model.lat_man.to_probs(prob_means)
        bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_prob_means = model.lat_man.from_probs(bounded_probs)

        # Rejoin the bounded means
        return model.join_mean_mixture(bounded_comp_means, bounded_prob_means)

    def make_regularizer(self) -> Callable[[Point[Natural, Mixture]], Array]:
        """Create a regularizer for the mixture model."""

        def loss_fn(params: Point[Natural, Mixture]) -> Array:
            # L2 regularization
            return self.l2_reg * jnp.sum(jnp.square(params.array))

        return loss_fn

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        lgm: FactorAnalysis,
        projection: Point[Natural, FactorAnalysis],
        logger: JaxLogger,
        epoch_offset: int,
        mix: Mixture,
        params0: Point[Natural, Mixture],
    ) -> Point[Natural, Mixture]:
        """Train a mixture model on data projected to the latent space."""
        # Project the data to the latent space
        train_data = dataset.train_data
        projected_data = self.project_data(lgm, projection, train_data)

        # Setup optimizer
        optim = optax.adam(learning_rate=self.lr)
        optimizer = Optimizer(optim, mix)
        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        # Initialize optimizer state
        opt_state = optimizer.init(params0)

        # Determine batch size and number of batches
        if self.batch_size is None:
            batch_size = projected_data.shape[0]
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = projected_data.shape[0] // batch_size

        # Log training start
        log.info(f"Training mixture on projected data for {self.n_epochs} epochs")

        # Create regularizer
        regularizer = self.make_regularizer()

        # Define batch step function
        def batch_step(
            carry: tuple[OptState, Point[Natural, Mixture]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, Mixture]], Array]:
            opt_state, params = carry

            # Compute posterior statistics for this batch
            posterior_stats = mix.mean_posterior_statistics(params, batch)

            # Apply bounds to posterior statistics
            bounded_posterior_stats = self.bound_mixture_means(mix, posterior_stats)

            # Define the inner step function
            def inner_step(
                carry: tuple[OptState, Point[Natural, Mixture]],
                _: None,
            ) -> tuple[tuple[OptState, Point[Natural, Mixture]], Array]:
                current_opt_state, current_params = carry

                # Compute gradient as difference between posterior and current prior
                prior_stats = mix.to_mean(current_params)
                grad = prior_stats - bounded_posterior_stats

                # Add regularization
                reg_grad = jax.grad(regularizer)(current_params)
                grad = grad + reg_grad

                # Update parameters
                new_opt_state, new_params = optimizer.update(
                    current_opt_state, grad, current_params
                )

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": current_params.array,
                        "updated_params": new_params.array,
                        "batch": batch,
                        "posterior_stats": posterior_stats.array,
                        "bounded_posterior_stats": bounded_posterior_stats.array,
                        "prior_stats": prior_stats.array,
                        "grad": grad.array,
                    },
                    handler,
                    context="PROJECTION",
                )

                return (new_opt_state, new_params), grad.array

            # Run inner steps
            (final_opt_state, final_params), all_grads = jax.lax.scan(
                inner_step,
                (opt_state, params),
                None,
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), all_grads

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, Mixture], Array],
        ) -> tuple[OptState, Point[Natural, Mixture], Array]:
            opt_state, params, epoch_key = carry

            # Split key for shuffling
            shuffle_key, next_key = jax.random.split(epoch_key)

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(
                shuffle_key, projected_data.shape[0]
            )
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_data = projected_data[batched_indices].reshape(
                (n_batches, batch_size, -1)
            )

            # Process all batches
            (opt_state, new_params), _ = jax.lax.scan(
                batch_step, (opt_state, params), batched_data
            )

            ana_hmog = analytic_hmog(
                lgm.obs_dim,
                lgm.obs_rep,
                mix.obs_man.data_dim,
                mix.n_categories,
            )

            hmog_params = to_analytic(lgm, mix, ana_hmog, projection, new_params)

            # Log metrics
            log_epoch_metrics(
                dataset,
                ana_hmog,
                logger,
                hmog_params,
                epoch + epoch_offset,
                {},
                None,
                log_freq=10,
            )

            return opt_state, new_params, next_key

        # Run training loop
        (_, params_final, _) = fori(
            0, self.n_epochs, epoch_step, (opt_state, params0, key)
        )

        return params_final


class ProjectionHMoGExperiment(ClusteringExperiment):
    """Experiment framework for projection-based HMoG training.

    This approach trains a model in two phases:
    1. Use PreTrainer to learn a Linear Gaussian Model (LGM)
    2. Project data to latent space using the trained LGM
    3. Train a mixture model on the projected data

    This follows a common approach in hierarchical clustering where
    dimensionality reduction (e.g., PCA, factor analysis) is applied
    before clustering.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        pre: LGMPreTrainer,
        pro: ProjectionTrainer,
        analysis: AnalysisArgs,
        lgm_noise_scale: float,
        mix_noise_scale: float,
        pretrain: bool,
    ) -> None:
        super().__init__()

        # Create the component models
        self.lgm = FactorAnalysis(
            obs_dim=data_dim,
            lat_dim=latent_dim,
        )

        self.mixture = AnalyticMixture(
            obs_man=Normal(latent_dim, Diagonal),
            n_categories=n_clusters,
        )

        # Create the full model for evaluation
        self.model = analytic_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal,
            lat_dim=latent_dim,
            n_components=n_clusters,
        )

        # Store trainers and configuration
        self.pre = pre
        self.projection = pro
        self.analysis = analysis
        self.lgm_noise_scale = lgm_noise_scale
        self.mix_noise_scale = mix_noise_scale
        self.pretrain = pretrain

        log.info(f"Initialized Projection HMoG model with dimension {self.model.dim}.")

    @property
    @override
    def n_epochs(self) -> int:
        """Calculate total number of epochs."""
        return self.pre.n_epochs + self.projection.n_epochs

    @property
    @override
    def n_clusters(self) -> int:
        """Return the number of clusters in the model."""
        return self.mixture.n_categories

    @override
    def initialize_model(
        self, key: Array, data: Array
    ) -> tuple[Point[Natural, FactorAnalysis], Point[Natural, Mixture]]:
        """Initialize model parameters."""
        key_lgm, key_mix = jax.random.split(key)

        # Initialize LGM
        lgm_params = self.initialize_lgm(key_lgm, data)

        # Initialize mixture
        mix_params = self.initialize_mixture(key_mix)

        # Join to create full model parameters
        return lgm_params, mix_params

    def initialize_lgm(self, key: Array, data: Array) -> Point[Natural, FactorAnalysis]:
        """Initialize LGM parameters."""
        # Initialize observable parameters from data statistics
        obs_means = self.lgm.obs_man.average_sufficient_statistic(data)
        obs_means = self.lgm.obs_man.regularize_covariance(
            obs_means, self.pre.jitter_var, self.pre.min_var
        )
        obs_params = self.lgm.obs_man.to_natural(obs_means)

        # Initialize interaction matrix with small noise
        int_noise = self.lgm_noise_scale * jax.random.normal(
            key, self.lgm.int_man.shape
        )
        int_params = self.lgm.int_man.point(self.lgm.int_man.rep.from_dense(int_noise))

        # Initialize latent parameters with standard normal
        lat_params = self.lgm.lat_man.to_natural(self.lgm.lat_man.standard_normal())

        # Join all parameters
        return self.lgm.join_params(obs_params, int_params, lat_params)

    def initialize_mixture(self, key: Array) -> Point[Natural, Mixture]:
        """Initialize mixture parameters."""
        return self.mixture.initialize(key, shape=self.mix_noise_scale)

    @override
    def generate(
        self,
        params: Array,
        key: Array,
        n_samples: int,
    ) -> Array:
        """Generate samples from the model."""
        return self.model.observable_sample(
            key, self.model.natural_point(params), n_samples
        )

    @override
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Generate analysis artifacts from saved experiment results."""
        raise NotImplementedError(
            "Analysis is not implemented for projection-based training."
        )
        # epoch = (
        #     self.analysis.epoch
        #     if self.analysis.epoch is not None
        #     else max(handler.get_available_epochs())
        # )
        #
        # if self.analysis.from_scratch:
        #     log.info("Recomputing artifacts from scratch.")
        #     params = self.model.natural_point(handler.load_params(epoch))
        #     log_artifacts(handler, dataset, logger, self.model, epoch, params)
        # else:
        #     log.info("Loading existing artifacts.")
        #     log_artifacts(handler, dataset, logger, self.model, epoch)

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Train HMoG model using projection approach."""
        # Split PRNG key for different training phases
        key_init, key_pre, key_proj = jax.random.split(key, 3)

        # Initialize model parameters
        lgm_params = self.initialize_lgm(key_init, dataset.train_data)

        # Track total epochs
        epoch = 0

        # Phase 1: Train the LGM component first
        if self.pretrain or self.pre.n_epochs > 0:
            # Construct path to the pretrained file

            if self.pretrain:
                new_lgm_array = handler.load_params(name="pretrain")
                lgm_params = self.lgm.natural_point(new_lgm_array)

            elif self.pre.n_epochs > 0:
                log.info("Phase 1: Training LGM for projection")
                lgm_params = self.pre.train(
                    key_pre,
                    handler,
                    dataset,
                    self.lgm,
                    logger,
                    epoch,
                    lgm_params,
                )
                epoch += self.pre.n_epochs

            # Save intermediate LGM parameters
            handler.save_params(lgm_params.array, epoch)

        # Phase 2: Train the mixture model on projected data
        if self.projection.n_epochs > 0:
            log.info("Phase 2: Training mixture model on projected data")

            # Initialize mixture parameters
            mix_params = self.initialize_mixture(key_proj)

            # Train mixture on projected data
            trained_mix_params = self.projection.train(
                key_proj,
                handler,
                dataset,
                self.lgm,
                lgm_params,
                logger,
                epoch,
                self.mixture,
                mix_params,
            )

            ana_hmog = analytic_hmog(
                self.lgm.obs_dim,
                self.lgm.obs_rep,
                self.mixture.obs_man.data_dim,
                self.mixture.n_categories,
            )

            final_params = to_analytic(
                self.lgm, self.mixture, ana_hmog, lgm_params, trained_mix_params
            )

            # Save final parameters
            final_epoch = epoch + self.projection.n_epochs

            # Generate artifacts
            log_artifacts(handler, dataset, logger, ana_hmog, final_epoch, final_params)

        log.info("Training complete")
