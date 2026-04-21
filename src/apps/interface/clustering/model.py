"""Clustering model abstractions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import override

import jax
import numpy as np
from jax import Array
from omegaconf import MISSING

from ...runtime import LL_METRIC_KEYS, Logger, RunHandler
from ..model import Model, ModelConfig
from ..protocols import HasLogLikelihood
from .dataset import ClusteringDataset
from .metrics import CLUSTERING_METRIC_KEYS

log = logging.getLogger(__name__)


def cycle_lr_schedule(keypoints: Sequence[float], num_cycles: int) -> list[float]:
    """Interpolate ``keypoints`` across ``num_cycles`` cycles.

    Returns ``num_cycles`` learning-rate multipliers by linearly
    interpolating the given keypoints across the cycle range. Used by
    HMoG and MFA to derive per-cycle LR scales from a small list of
    user-provided anchor values.

    - ``keypoints = []`` → all 1.0
    - ``keypoints = [s]`` → all ``s``
    - ``len(keypoints) > num_cycles`` → subsampled with a warning
    """
    n = len(keypoints)

    if n == 0:
        return [1.0] * num_cycles
    if n == 1:
        return [keypoints[0]] * num_cycles

    if num_cycles < n:
        log.warning(f"Too many keypoints ({n}) for {num_cycles} cycles. Subsampling.")
        indices = np.linspace(0, n - 1, num=num_cycles).round().astype(int)
        return [keypoints[i] for i in indices]

    x_keypoints = np.linspace(0, num_cycles - 1, num=n)
    x_full = np.arange(num_cycles)
    schedule = np.interp(x_full, x_keypoints, keypoints)
    return schedule.tolist()


@dataclass
class ClusteringModelConfig(ModelConfig):
    """Base configuration for clustering models."""

    _target_: str
    data_dim: int = MISSING
    n_clusters: int = MISSING


class ClusteringModel(Model[ClusteringDataset], ABC):
    """Abstract base class for clustering models.

    This is a minimal interface. Additional capabilities are provided
    via protocols (HasLogLikelihood, IsGenerative, HasSoftAssignments, etc.)

    ``metric_names`` auto-composes based on what the subclass actually
    does:
      - ``LL_METRIC_KEYS`` if the model implements ``HasLogLikelihood``
      - ``CLUSTERING_METRIC_KEYS`` if ``dataset.has_labels``
      - per-subclass ``training_metric_keys`` for anything else the training
        loop emits (update_stats diagnostics, regularization penalties, etc.)
      - the union of ``metric_keys`` across ``get_analyses(dataset)``

    Subclasses typically only need to declare ``training_metric_keys`` —
    they don't need to override ``metric_names`` itself.
    """

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        """Return the number of clusters in the model."""

    @property
    def training_metric_keys(self) -> frozenset[str]:
        """Metric keys emitted by the training loop beyond the LL and
        clustering baselines.

        Do NOT include ``LL_METRIC_KEYS`` or ``CLUSTERING_METRIC_KEYS``
        here — ``metric_names`` adds them automatically based on
        ``HasLogLikelihood`` and ``dataset.has_labels``. Do include
        per-model stats/regularization keys (``HMOG_TRAINING_METRIC_KEYS``,
        ``L1_L2_METRIC_KEYS``, ``PRECISION_REG_METRIC_KEYS``, etc.).
        """
        return frozenset()

    @override
    def metric_names(self, dataset: ClusteringDataset) -> frozenset[str]:
        """Every metric key this model's pipeline can emit for ``dataset``."""
        keys = self.training_metric_keys
        if isinstance(self, HasLogLikelihood):
            keys = keys | LL_METRIC_KEYS
        if dataset.has_labels:
            keys = keys | CLUSTERING_METRIC_KEYS
        for analysis in self.get_analyses(dataset):
            keys = keys | analysis.metric_names
        return keys

    @abstractmethod
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters.

        Args:
            params: Model parameters
            data: Data array of shape (n_samples, data_dim)

        Returns:
            Array of cluster assignments with shape (n_samples,)
        """


### Shared cyclic-training orchestration ###


def run_cyclic_training(
    model: ClusteringModel,
    key: Array,
    handler: RunHandler,
    logger: Logger,
    dataset: ClusteringDataset,
    params: Array,
    *,
    num_cycles: int,
    lr_scales: Sequence[float],
    epochs_per_cycle: int,
    cycle_start_epoch: int,
    current_epoch: int,
    run_cycle: Callable[[Array, Array, int, float, int], tuple[Array, int]],
) -> tuple[Array, int]:
    """Orchestrate a cyclic training loop with resume + checkpoint plumbing.

    Handles the boilerplate shared by any clustering model that trains in
    ``num_cycles`` cycles of fixed ``epochs_per_cycle`` — resume-to-correct-
    cycle detection, per-cycle key splitting, and calling
    ``model.process_checkpoint`` at each cycle boundary so Optuna can prune
    between cycles.

    ``run_cycle(cycle_key, params, cycle_idx, lr_scale, epoch_offset)``
    executes the work of one cycle (which may itself be multi-phase) and
    returns ``(new_params, new_epoch)``. The helper only orchestrates —
    it neither advances the epoch counter itself nor prescribes how cycle
    work is structured internally.

    Args:
        model: Model being trained (used for ``process_checkpoint``).
        key: PRNG key, split once into ``num_cycles`` per-cycle keys.
        handler: Run handler for checkpoint I/O.
        logger: Metrics logger.
        dataset: Dataset (passed through to ``process_checkpoint``).
        params: Initial params (before the first cycle to be run).
        num_cycles: Total number of cycles the model runs end-to-end.
        lr_scales: Per-cycle learning-rate multipliers. Length must equal
            ``num_cycles``.
        epochs_per_cycle: Epochs that each cycle consumes. Used only to
            derive ``current_cycle`` from ``current_epoch``.
        cycle_start_epoch: Epoch index at which cycle 0 begins. For a
            model with pretraining this equals ``pre.n_epochs``; for a
            model with no pretraining it is 0.
        current_epoch: The epoch currently loaded from checkpoint (0 for
            fresh starts). Used to compute which cycle to resume at.
        run_cycle: Callable executing a single cycle.

    Returns:
        ``(final_params, final_epoch)`` after all remaining cycles have
        run.
    """
    assert len(lr_scales) == num_cycles, (
        f"lr_scales length ({len(lr_scales)}) must match num_cycles ({num_cycles})"
    )

    cycle_keys = list(jax.random.split(key, num_cycles))

    if current_epoch <= cycle_start_epoch:
        current_cycle = 0
    else:
        current_cycle = (current_epoch - cycle_start_epoch) // epochs_per_cycle

    epoch = max(current_epoch, cycle_start_epoch)

    for cycle in range(current_cycle, num_cycles):
        log.info("Starting cycle %d/%d", cycle + 1, num_cycles)
        lr_scale = lr_scales[cycle]
        log.info("Learning rate scale: %.3f", lr_scale)
        params, epoch = run_cycle(cycle_keys[cycle], params, cycle, lr_scale, epoch)
        model.process_checkpoint(key, handler, logger, dataset, model, epoch, params)
        log.info("Completed cycle %d/%d", cycle + 1, num_cycles)

    return params, epoch
