"""Neural trace dataset implementation."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec

from apps.configs import ClusteringDatasetConfig
from apps.plugins import ClusteringDataset


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

    _target_: str = field(
        default="plugins.datasets.neural_traces.NeuralTracesDataset", init=False
    )
    dataset_name: str
    use_8hz_chirp: bool
    chirp_len: int
    bar_len: int
    feature_len: int
    train_split: float
    random_seed: int


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
        dataset_name: str,
        use_8hz_chirp: bool,
        chirp_len: int,
        bar_len: int,
        feature_len: int,
        train_split: float,
        random_seed: int,
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
        self.use_8hz_chirp: bool = use_8hz_chirp
        self.chirp_len: int = chirp_len
        self.bar_len: int = bar_len
        self.feature_len: int = feature_len

        # Create cache directories
        dataset_dir = cache_dir / "neural-traces"
        raw_dir = dataset_dir / "raw"
        processed_dir = dataset_dir / "processed" / dataset_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Check if processed files exist
        chirp_type = "8hz" if use_8hz_chirp else "preproc"
        train_cache = processed_dir / f"train_data_{chirp_type}.npy"
        test_cache = processed_dir / f"test_data_{chirp_type}.npy"

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

            chirp_col = "chirp_8Hz_average_norm" if use_8hz_chirp else "preproc_chirp"

            if chirp_col not in selected_df.columns:
                raise ValueError(
                    f"Column {chirp_col} not found in dataset. Available columns: {', '.join(selected_df.columns)}"
                )

            chirp_matrix = np.vstack(selected_df[chirp_col].values)

            # Concatenate data
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

    @property
    @override
    def observable_shape(self) -> tuple[int, int]:
        """Return the shape for visualizing a single neural trace."""
        # Using an appropriate aspect ratio for neural traces (height, width)
        data_dim = self.data_dim
        return (data_dim // 3, data_dim)  # Approximation for visualization

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        """Return the shape for visualizing a cluster of neural traces."""
        # Slightly larger than observable_shape to account for members
        obs_height, obs_width = self.observable_shape
        return (math.ceil(obs_height * 1.2), math.ceil(obs_width * 1.5))

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Visualize a neural trace prototype and its cluster members with transparency."""

        # Count members and set title
        n_members = members.shape[0]

        axes.axis("off")
        axes.set_title(f"Cluster {cluster_id} (Size: {n_members})")

        # Limit the number of members to display
        display_members = members
        max_display_members = 500
        if n_members > max_display_members:
            indices = np.random.choice(n_members, max_display_members, replace=False)
            display_members = members[indices]
            axes.set_title(
                f"Cluster {cluster_id} (Size: {n_members}, Display: {max_display_members})"
            )

        n_display_members = display_members.shape[0]

        # Prepare figure geometry and axes
        subplot_spec = axes.get_subplotspec()
        fig = axes.get_figure()

        if subplot_spec is None:
            raise ValueError("paint_cluster requires a subplot")

        assert isinstance(fig, Figure)

        # Create a grid within the provided axes
        gs = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=subplot_spec,
            width_ratios=[1.5, 1, 0.5],
            wspace=0.15,
        )

        # Create subplots for each component
        chirp_ax = Axes(fig, gs[0, 0])
        bar_ax = Axes(fig, gs[0, 1])
        feat_ax = Axes(fig, gs[0, 2])

        # Add the subplots to the figure
        axes.figure.add_subplot(chirp_ax)
        axes.figure.add_subplot(bar_ax)
        axes.figure.add_subplot(feat_ax)

        # Compute alpha based on number of traces
        base_alpha = min(0.4, 10.0 / max(1, n_display_members))

        # First plot all member traces with transparency
        for i in range(n_display_members):
            trace = display_members[i]
            chirp_response = trace[: self.chirp_len]
            bar_response = trace[self.chirp_len : self.chirp_len + self.bar_len]

            # Plot member chirp response with transparency
            chirp_ax.plot(
                chirp_response, color="black", linewidth=0.5, alpha=base_alpha
            )

            # Plot member bar response with transparency
            bar_ax.plot(bar_response, color="black", linewidth=0.5, alpha=base_alpha)

        # Then plot the prototype on top with full opacity
        proto_chirp = prototype[: self.chirp_len]
        proto_bar = prototype[self.chirp_len : self.chirp_len + self.bar_len]
        proto_features = prototype[self.chirp_len + self.bar_len :]

        # Plot prototype chirp response
        chirp_ax.plot(proto_chirp, color="red", linewidth=1.5)
        chirp_ax.set_xticks([])
        chirp_ax.spines["top"].set_visible(False)
        chirp_ax.spines["right"].set_visible(False)

        # Plot prototype bar response
        bar_ax.plot(proto_bar, color="red", linewidth=1.5)
        bar_ax.set_xticks([])
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)

        # Show features as text
        feat_ax.axis("off")
        feature_names = ["Bar DS p-value", "Bar OS p-value", "ROI size (μm²)"]
        feature_text = "\n".join(
            f"{name}: {value:.3f}" for name, value in zip(feature_names, proto_features)
        )
        feat_ax.text(
            0.05,
            0.5,
            feature_text,
            fontsize=7,  # Smaller font
            va="center",
            ha="left",
            transform=feat_ax.transAxes,
        )

    @override
    def paint_observable(self, observable: Array, axes: Axes) -> None:
        """Visualize a single neural trace."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        # Turn off the main axes
        axes.set_axis_off()

        subplot_spec = axes.get_subplotspec()
        fig = axes.get_figure()

        if subplot_spec is None:
            raise ValueError("paint_observable requires a subplot")

        assert isinstance(fig, Figure)

        # Create a grid within the provided axes
        gs = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=subplot_spec,
            width_ratios=[1.5, 1, 0.5],
            wspace=0.15,
        )

        # Create subplots for each component
        chirp_ax = Axes(fig, gs[0, 0])
        bar_ax = Axes(fig, gs[0, 1])
        feat_ax = Axes(fig, gs[0, 2])

        # Add the subplots to the figure
        axes.figure.add_subplot(chirp_ax)
        axes.figure.add_subplot(bar_ax)
        axes.figure.add_subplot(feat_ax)

        # Extract data
        chirp_len = 249
        bar_len = 32
        chirp_response = observable[:chirp_len]
        bar_response = observable[chirp_len : chirp_len + bar_len]
        features = observable[chirp_len + bar_len :]

        chirp_ax.plot(chirp_response, color="black", linewidth=1)
        chirp_ax.set_xticks([])
        chirp_ax.spines["top"].set_visible(False)
        chirp_ax.spines["right"].set_visible(False)

        bar_ax.plot(bar_response, color="black", linewidth=1)
        bar_ax.set_xticks([])
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)

        # Show features as text without background
        feat_ax.axis("off")
        feature_names = ["Bar DS p-value", "Bar OS p-value", "ROI size (μm²)"]
        feature_text = "\n".join(
            f"{name}: {value:.3f}" for name, value in zip(feature_names, features)
        )
        feat_ax.text(
            0.05,
            0.5,
            feature_text,
            fontsize=7,  # Smaller font
            va="center",
            ha="left",
            transform=feat_ax.transAxes,
        )
