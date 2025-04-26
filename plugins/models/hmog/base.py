"""Configuration for HMoG implementations."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

import jax
import jax.numpy as jnp
from goal.geometry import (
    AffineMap,
    Diagonal,
    Manifold,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Rectangular,
    Replicated,
    Scale,
)
from goal.models import (
    AnalyticLinearGaussianModel,
    DifferentiableHMoG,
    DifferentiableLinearGaussianModel,
    Euclidean,
    FullNormal,
    Normal,
)
from jax import Array

from apps.configs import STATS_LEVEL
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import MetricDict
from apps.runtime.logger import JaxLogger

# Start logger
log = logging.getLogger(__name__)


### Covariance Reps ###


class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


### HMoG Protocol ###

type HMoG = DifferentiableHMoG[Any, Any]


### Helpers ###


def fori[X](lower: int, upper: int, body_fun: Callable[[Array, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


def relative_entropy_regularization[
    ObsRep: PositiveDefinite,
    PostRep: PositiveDefinite,
](
    lgm: DifferentiableLinearGaussianModel[ObsRep, PostRep],
    batch: Array,
    lkl_params: Point[
        Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
    ],
) -> tuple[Array, Point[Mean, Normal[PostRep]]]:
    z = lgm.lat_man.to_natural(lgm.lat_man.standard_normal())
    rho = lgm.conjugation_parameters(lkl_params)
    lat_params = lgm.pst_lat_emb.translate(-rho, z)
    lat_loc, lat_prs = lgm.con_lat_man.split_location_precision(lat_params)
    dns_prs = lgm.con_lat_man.cov_man.to_dense(lat_prs)
    dia_prs = lgm.lat_man.cov_man.from_dense(dns_prs)
    sub_lat_params = lgm.lat_man.join_location_precision(lat_loc, dia_prs)
    obs_params, int_params = lgm.lkl_man.split_params(lkl_params)
    lgm_params = lgm.join_params(obs_params, int_params, sub_lat_params)
    lgm_means = lgm.mean_posterior_statistics(lgm_params, batch)
    lgm_lat_means = lgm.split_params(lgm_means)[2]
    return lgm.lat_man.relative_entropy(lgm_lat_means, z), lgm_lat_means


def relative_entropy_regularization_full[
    ObsRep: PositiveDefinite,
    PostRep: PositiveDefinite,
](
    lgm: DifferentiableLinearGaussianModel[ObsRep, PostRep],
    batch: Array,
    lkl_params: Point[
        Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
    ],
) -> tuple[AnalyticLinearGaussianModel[ObsRep], Array, Point[Mean, FullNormal]]:
    # Relative entropy regularization
    ana_lgm = AnalyticLinearGaussianModel(lgm.obs_dim, lgm.obs_rep, lgm.lat_dim)
    z = ana_lgm.lat_man.to_natural(ana_lgm.lat_man.standard_normal())
    lgm_params = ana_lgm.join_conjugated(lkl_params, z)
    lgm_means = ana_lgm.mean_posterior_statistics(lgm_params, batch)
    lgm_lat_means = ana_lgm.split_params(lgm_means)[2]
    re_loss = ana_lgm.lat_man.relative_entropy(lgm_lat_means, z)
    return ana_lgm, re_loss, lgm_lat_means


### Logging ###


def log_epoch_metrics[H: HMoG](
    dataset: ClusteringDataset,
    hmog_model: H,
    logger: JaxLogger,
    params: Point[Natural, H],
    epoch: Array,
    batch_grads: None | Point[Mean, Replicated[H]] = None,
    log_freq: int = 1,
) -> None:
    """Log metrics for an epoch."""
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> None:
        # Core Performance Metrics

        epoch_train_ll = hmog_model.average_log_observable_density(params, train_data)
        epoch_test_ll = hmog_model.average_log_observable_density(params, test_data)

        n_samps = train_data.shape[0]
        epoch_scaled_bic = (
            -(hmog_model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll) / 2
        )
        info = jnp.array(logging.INFO)

        metrics: MetricDict = {
            "Performance/Train Log-Likelihood": (
                info,
                epoch_train_ll,
            ),
            "Performance/Test Log-Likelihood": (
                info,
                epoch_test_ll,
            ),
            "Performance/Scaled BIC": (
                info,
                epoch_scaled_bic,
            ),
        }

        # Raw Parameter Statistics

        stats = jnp.array(STATS_LEVEL)

        def update_parameter_stats[M: Manifold](
            name: str, params: Point[Natural, M]
        ) -> None:
            array = params.array
            metrics.update(
                {
                    f"Params/{name} Min": (
                        stats,
                        jnp.min(array),
                    ),
                    f"Params/{name} Median": (
                        stats,
                        jnp.median(array),
                    ),
                    f"Params/{name} Max": (
                        stats,
                        jnp.max(array),
                    ),
                }
            )

        obs_params, lwr_int_params, upr_params = hmog_model.split_params(params)
        obs_loc_params, obs_prs_params = hmog_model.obs_man.split_params(obs_params)
        lat_params, upr_int_params, cat_params = hmog_model.upr_hrm.split_params(
            upr_params
        )
        lat_loc_params, lat_prs_params = hmog_model.upr_hrm.obs_man.split_params(
            lat_params
        )

        update_parameter_stats("Obs Location", obs_loc_params)
        update_parameter_stats("Obs Precision", obs_prs_params)
        update_parameter_stats("Obs Interaction", lwr_int_params)
        update_parameter_stats("Lat Location", lat_loc_params)
        update_parameter_stats("Lat Precision", lat_prs_params)
        update_parameter_stats("Lat Interaction", upr_int_params)
        update_parameter_stats("Categorical", cat_params)

        # Regularization Metrics

        # Compute latent distribution statistics
        # Get latent distribution in mean coordinates for analysis
        lgm = hmog_model.lwr_hrm
        lkl_params = lgm.lkl_man.join_params(obs_params, lwr_int_params)
        ana_lgm, re_loss, lgm_lat_means = relative_entropy_regularization_full(
            lgm, train_data, lkl_params
        )
        lat_mean, lat_cov = ana_lgm.lat_man.split_mean_covariance(lgm_lat_means)
        lat_mean_array = lat_mean.array
        lat_cov_array = ana_lgm.lat_man.cov_man.to_dense(lat_cov)

        # Add latent distribution metrics
        metrics.update(
            {
                "Regularization/Loading Sparsity": (
                    stats,
                    jnp.mean(jnp.abs(lwr_int_params.array) < 1e-6),
                ),
                "Regularization/Latent KL to Z": (
                    stats,
                    re_loss,
                ),
                # Mean vector summary
                "Regularization/Latent Mean Norm": (
                    stats,
                    jnp.linalg.norm(lat_mean_array),
                ),
                "Regularization/Latent Mean Min": (
                    stats,
                    jnp.min(lat_mean_array),
                ),
                "Regularization/Latent Mean Max": (
                    stats,
                    jnp.max(lat_mean_array),
                ),
                # Variance summary
                "Regularization/Latent Var Min": (
                    stats,
                    jnp.min(jnp.diag(lat_cov_array)),
                ),
                "Regularization/Latent Var Max": (
                    stats,
                    jnp.max(jnp.diag(lat_cov_array)),
                ),
                "Regularization/Latent Var Mean": (
                    stats,
                    jnp.mean(jnp.diag(lat_cov_array)),
                ),
                # Eigenvalue analysis
                "Regularization/Latent Eigenvalue Min": (
                    stats,
                    jnp.min(jnp.linalg.eigvalsh(lat_cov_array)),
                ),
                # Eigenvalue analysis
                "Regularization/Latent Eigenvalue Median": (
                    stats,
                    jnp.median(jnp.linalg.eigvalsh(lat_cov_array)),
                ),
                "Regularization/Latent Eigenvalue Max": (
                    stats,
                    jnp.max(jnp.linalg.eigvalsh(lat_cov_array)),
                ),
                # Structure summary
                "Regularization/Latent Off-Diag Magnitude": (
                    stats,
                    jnp.linalg.norm(
                        lat_cov_array - jnp.diag(jnp.diag(lat_cov_array)), "fro"
                    ),
                ),
                "Regularization/Latent Condition Number": (
                    stats,
                    jnp.linalg.cond(
                        lat_cov_array + jnp.eye(lat_cov_array.shape[0])
                    ),  # Small epsilon for stability
                ),
                "Regularization/Latent Effective Rank": (
                    stats,
                    jnp.sum(jnp.linalg.eigvalsh(lat_cov_array) > 1e-6),
                ),
            }
        )

        ### Grad Norms ###

        def update_grad_stats[M: Manifold](name: str, grad_norms: Array) -> None:
            metrics.update(
                {
                    f"Grad Norms/{name} Min": (
                        stats,
                        jnp.min(grad_norms),
                    ),
                    f"Grad Norms/{name} Median": (
                        stats,
                        jnp.median(grad_norms),
                    ),
                    f"Grad Norms/{name} Max": (
                        stats,
                        jnp.max(grad_norms),
                    ),
                }
            )

        def norm_grads(grad: Point[Mean, H]) -> Array:
            obs_grad, lwr_int_grad, upr_grad = hmog_model.split_params(grad)
            obs_loc_grad, obs_prs_grad = hmog_model.obs_man.split_params(obs_grad)
            lat_grad, upr_int_grad, cat_grad = hmog_model.upr_hrm.split_params(upr_grad)
            lat_loc_grad, lat_prs_grad = hmog_model.upr_hrm.obs_man.split_params(
                lat_grad
            )
            return jnp.asarray(
                [
                    jnp.linalg.norm(grad.array)
                    for grad in [
                        obs_loc_grad,
                        obs_prs_grad,
                        lwr_int_grad,
                        lat_loc_grad,
                        lat_prs_grad,
                        upr_int_grad,
                        cat_grad,
                    ]
                ]
            )

        if batch_grads is not None:
            batch_man: Replicated[H] = Replicated(hmog_model, batch_grads.shape[0])
            grad_norms = batch_man.map(norm_grads, batch_grads).T

            update_grad_stats("Obs Location", grad_norms[0])
            update_grad_stats("Obs Precision", grad_norms[1])
            update_grad_stats("Obs Interaction", grad_norms[2])
            update_grad_stats("Lat Location", grad_norms[3])
            update_grad_stats("Lat Precision", grad_norms[4])
            update_grad_stats("Lat Interaction", grad_norms[5])
            update_grad_stats("Categorical", grad_norms[6])

        logger.log_metrics(metrics, epoch + 1)

    def no_op() -> None:
        return None

    jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)


# @dataclass(frozen=True)
# class HMoGWrapper[H: HMoG](Manifold):
#     """Concrete wrapper for HMoG models (both Differentiable and Symmetric)."""
#
#     model: H
#
#     @property
#     @override
#     def dim(self) -> int:
#         return self.model.dim
#
#     @property
#     def shape(self) -> tuple[int, ...]:
#         return (self.dim,)
#
#     # Required properties that forward to the underlying model
#     @property
#     def obs_man(self) -> Any:
#         return self.model.obs_man
#
#     @property
#     def upr_hrm(self) -> Any:
#         return self.model.upr_hrm
#
#     @property
#     def con_upr_hrm(self) -> Any:
#         # Handle the possibility that one model type doesn't have this property
#         return self.model.upr_hrm
#
#     @property
#     def lwr_hrm(self) -> Any:
#         return self.model.lwr_hrm
#
#     @property
#     def lkl_man(self) -> Any:
#         return self.model.lkl_man
#
#     @property
#     def lat_man(self) -> Any:
#         return self.model.lat_man
#
#     # Create points in this manifold
#     def natural_point(self, array: Array) -> Point[Natural, H]:
#         return self.model.natural_point(array)
#
#     def posterior_at(
#         self,
#         params: Point[Natural, H],
#         x: Array,
#     ) -> Point[Natural, Any]:
#         """Compute the posterior of the model at a given point."""
#         return self.model.posterior_at(params, x)
#
#     # def split_conjugated(
#     #     self, params: Point[Natural, Any]
#     # ) -> tuple[
#     #     Point[Natural, Any],
#     #     Point[Natural, Any],
#     # ]: ...
#     #
#     # def prior(self, params: Point[Natural, Any]) -> Point[Natural, Any]: ...
#     #
#     # def split_params[C: Coordinates](
#     #     self, params: Point[C, Any]
#     # ) -> tuple[
#     #     Point[C, Any],
#     #     Point[C, Any],
#     #     Point[C, Any],
#     # ]: ...
#     #
#     # def average_log_observable_density(
#     #     self, params: Point[Natural, Any], xs: Array
#     # ) -> Array: ...
#
#
# # @dataclass(frozen=True)
# # class HMoG(ABC):
# #     """Protocol for Hierarchical Mixture of Gaussians models."""
# #
# #     model: DifferentiableHMoG[Diagonal, Diagonal] | SymmetricHMoG[Diagonal, Diagonal]
# #
# #     @property
# #     def obs_man(self) -> Any: ...
# #
# #     @property
# #     def upr_hrm(self) -> Any: ...
# #
# #     @property
# #     def con_upr_hrm(self) -> Any: ...
# #
# #     @property
# #     def lwr_hrm(self) -> Any: ...
# #
# #     @property
# #     def dim(self) -> int: ...
# #
# #     def posterior_at(
# #         self,
# #         params: Point[
# #             Natural,
# #             DifferentiableHMoG[Diagonal, Diagonal] | SymmetricHMoG[Diagonal, Diagonal],
# #         ],
# #         x: Array,
# #     ) -> Point[Natural, Any]:
# #         """Compute the posterior of the model at a given point."""
# #         return self.model.posterior_at(params, x)
# #
# #     def natural_point(self, params: Array) -> Point[Natural, Any]: ...
# #
# #     def split_conjugated(
# #         self, params: Point[Natural, Any]
# #     ) -> tuple[
# #         Point[Natural, Any],
# #         Point[Natural, Any],
# #     ]: ...
# #
# #     def prior(self, params: Point[Natural, Any]) -> Point[Natural, Any]: ...
# #
# #     def split_params[C: Coordinates](
# #         self, params: Point[C, Any]
# #     ) -> tuple[
# #         Point[C, Any],
# #         Point[C, Any],
# #         Point[C, Any],
# #     ]: ...
# #
# #     def average_log_observable_density(
# #         self, params: Point[Natural, Any], xs: Array
# #     ) -> Array: ...
