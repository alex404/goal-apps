"""Training algorithms for MFA model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from goal.geometry import Optimizer, OptState
from jax import Array

from apps.interface import ClusteringDataset
from apps.runtime import Logger, RunHandler

if TYPE_CHECKING:
    from .model import MFAModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradientTrainer:
    """Gradient descent trainer with AdamW optimizer.

    Follows the training pattern from the goal-jax MFA example, using
    gradient descent with AdamW to minimize negative log-likelihood.
    """

    lr: float
    """Learning rate for AdamW optimizer."""

    n_epochs: int
    """Number of training epochs."""

    batch_size: int | None = None
    """Batch size (None = full batch gradient descent)."""

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: "MFAModel",
        logger: Logger,
        epoch_offset: int,
        params0: Array,
    ) -> Array:
        """Execute gradient descent training loop.

        Args:
            key: Random key
            handler: Run handler
            dataset: Dataset with train_data
            model: MFA model instance
            logger: Logger for metrics
            epoch_offset: Starting epoch number
            params0: Initial parameters

        Returns:
            Final trained parameters
        """
        # Create optimizer
        optimizer = Optimizer.adamw(
            man=model.mfa,
            learning_rate=self.lr,
        )
        opt_state = optimizer.init(params0)

        # Loss function
        def cross_entropy_loss(params: Array) -> Array:
            """Negative log likelihood."""
            return -model.mfa.average_log_observable_density(params, dataset.train_data)

        # Training step
        def train_step(
            carry: tuple[OptState, Array],
            _: Any,
        ) -> tuple[tuple[OptState, Array], dict[str, Array]]:
            """Single gradient descent step."""
            opt_state, params = carry

            # Compute loss and gradients
            loss_val, grads = jax.value_and_grad(cross_entropy_loss)(params)

            # Update parameters
            opt_state, params = optimizer.update(opt_state, grads, params)

            # Metrics for logging (keep as JAX arrays for scan)
            metrics = {
                "train/loss": loss_val,
                "train/log_likelihood": -loss_val,
            }

            return (opt_state, params), metrics

        # JIT compile the training loop
        @jax.jit
        def run_epochs(
            opt_state: OptState,
            params: Array,
        ) -> tuple[Array, dict[str, Array]]:
            """Run all epochs."""
            (_, final_params), metrics_list = jax.lax.scan(
                train_step,
                (opt_state, params),
                None,
                length=self.n_epochs,
            )
            return final_params, metrics_list

        # Execute training
        log.info(f"Running {self.n_epochs} epochs of gradient descent (lr={self.lr})")
        final_params, all_metrics = run_epochs(opt_state, params0)

        # Log metrics for each epoch
        for i in range(len(all_metrics["train/loss"])):
            epoch = epoch_offset + i + 1
            metrics_at_epoch = {
                name: (jnp.array(logging.INFO), all_metrics[name][i])
                for name in all_metrics
            }
            logger.log_metrics(metrics_at_epoch, jnp.array(epoch))

        return final_params
