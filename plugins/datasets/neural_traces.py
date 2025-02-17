"""Neural trace dataset implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes

from apps.configs import ClusteringDatasetConfig
from apps.plugins import ClusteringDataset, ObservableArtifact


@dataclass
class NeuralTracesConfig(ClusteringDatasetConfig):
    """Configuration for neural traces dataset.

    Parameters:
        dataset_name: Name of the preprocessed dataset to load
        chirp_len: Length of chirp response trace
        bar_len: Length of bar response trace
        feature_len: Number of additional features
        train_split: Fraction of data to use for training
        random_seed: Seed for train/test split
    """

    _target_: str = "plugins.datasets.neural_traces.NeuralTracesDataset"
    dataset_name: str = "all-rgc"
    chirp_len: int = 249
    bar_len: int = 32
    feature_len: int = 3
    train_split: float = 0.8
    random_seed: int = 42


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="neural_traces", node=NeuralTracesConfig)


class NeuralTracesDataset(ClusteringDataset):
    """Neural traces dataset for clustering analysis."""

    _train_data: Array
    _test_data: Array

    def __init__(
        self,
        cache_dir: Path,
        dataset_name: str = "all-rgc",
        chirp_len: int = 249,
        bar_len: int = 32,
        feature_len: int = 3,
        train_split: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        """Load neural traces dataset.

        Args:
            cache_dir: Directory for caching data
            dataset_name: Name of the preprocessed dataset to load
            chirp_len: Length of chirp response trace
            bar_len: Length of bar response trace
            feature_len: Number of additional features
            train_split: Fraction of data to use for training
            random_seed: Seed for train/test split

        Returns:
            Loaded neural traces dataset
        """
        self.cache_dir: Path = cache_dir
        self.dataset_name: str = dataset_name
        self.chirp_len: int = chirp_len
        self.bar_len: int = bar_len
        self.feature_len: int = feature_len

        # Create cache directories
        dataset_dir = cache_dir / dataset_name
        raw_dir = dataset_dir / "raw"
        processed_dir = dataset_dir / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Check if processed files exist
        train_cache = processed_dir / "train_data.npy"
        test_cache = processed_dir / "test_data.npy"

        if train_cache.exists() and test_cache.exists():
            # Load cached split
            train_data = np.load(train_cache)
            test_data = np.load(test_cache)
        else:
            # Load and preprocess raw data
            df = pd.read_pickle(raw_dir / f"{dataset_name}.pkl")

            # Quality control (as in preprocess.py)
            quality_mask = (df.chirp_qidx > 0.35) | (df.bar_qidx > 0.6)
            selected_df = df[quality_mask]

            # Concatenate data
            chirp_matrix = np.vstack(selected_df["preproc_chirp"].values)
            bar_matrix = np.vstack(selected_df["preproc_bar"].values)
            features = ["bar_ds_pvalue", "bar_os_pvalue", "roi_size_um2"]
            feature_matrix = selected_df[features].values
            raw_data = np.hstack([chirp_matrix, bar_matrix, feature_matrix])

            # Randomly split data
            rng = np.random.default_rng(random_seed)
            n_samples = len(raw_data)
            indices = rng.permutation(n_samples)
            split_idx = int(n_samples * train_split)

            train_data = raw_data[indices[:split_idx]]
            test_data = raw_data[indices[split_idx:]]

            # Cache the split
            np.save(train_cache, train_data)
            np.save(test_cache, test_data)

        # Convert to JAX arrays
        self._train_data = jnp.array(train_data)
        self._test_data = jnp.array(test_data)

    @property
    @override
    def train_data(self) -> Array:
        return self._train_data

    @property
    @override
    def test_data(self) -> Array:
        return self._test_data

    @property
    @override
    def data_dim(self) -> int:
        return self.chirp_len + self.bar_len + self.feature_len

    @override
    def observable_artifact(self, observable: Array) -> ObservableArtifact:
        """Convert neural trace to visualization-friendly format.

        Args:
            observable: Neural trace array of shape (data_dim,)

        Returns:
            ObservableArtifact containing the chirp response for visualization
        """
        # Extract chirp response
        chirp_response = observable[: self.chirp_len]
        y_height = np.round(self.chirp_len / 2)
        return ObservableArtifact(obs=chirp_response, shape=(y_height, self.chirp_len))

    @override
    @staticmethod
    def paint_observable(observable: ObservableArtifact, axes: Axes) -> None:
        """Visualize a single neural trace.

        Args:
            observable: ObservableArtifact containing the trace to visualize
            axes: Matplotlib axes to plot on
        """
        # Plot the time series
        obs = observable.obs
        axes.plot(obs, color="black")
        axes.set_xticks([])  # Remove x-axis ticks for cleaner visualization
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
