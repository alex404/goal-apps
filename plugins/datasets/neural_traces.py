"""Neural trace dataset implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn
from h5py import File, Group
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from apps.interface import Analysis, ClusteringDataset, ClusteringDatasetConfig
from apps.runtime import Artifact, MetricDict, RunHandler


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
        default="plugins.datasets.neural_traces.NeuralTracesDataset.load", init=False
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


@dataclass(frozen=True)
class NeuralTracesDataset(ClusteringDataset):
    """Neural traces dataset for clustering analysis."""

    dataset_name: str
    use_8hz_chirp: bool
    chirp_len: int
    bar_len: int
    feature_len: int
    _train_data: Array
    _test_data: Array

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        dataset_name: str,
        use_8hz_chirp: bool,
        chirp_len: int,
        bar_len: int,
        feature_len: int,
        train_split: float,
        random_seed: int,
    ) -> NeuralTracesDataset:
        """Load neural traces dataset.

        Args:
            cache_dir: Directory for caching data
            dataset_name: Name of the preprocessed dataset to load
            use_8hz_chirp: Whether to use the 8Hz chirp
            chirp_len: Length of chirp response trace
            bar_len: Length of bar response trace
            feature_len: Number of additional features
            train_split: Fraction of data to use for training
            random_seed: Seed for train/test split

        Returns:
            Loaded neural traces dataset
        """
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
            assert isinstance(df, pd.DataFrame), "Expected DataFrame from pickle"

            # Quality control (as in preprocess.py)
            quality_mask = (df.chirp_qidx > 0.35) | (df.bar_qidx > 0.6)
            selected_df = df.loc[quality_mask]

            chirp_col = "chirp_8Hz_average_norm" if use_8hz_chirp else "preproc_chirp"

            if chirp_col not in selected_df.columns:
                raise ValueError(
                    f"Column {chirp_col} not found in dataset. Available columns: {', '.join(selected_df.columns)}"
                )

            chirp_matrix = np.vstack(list(selected_df[chirp_col]))

            # Concatenate data
            bar_matrix = np.vstack(list(selected_df["preproc_bar"]))
            feature_cols = ["bar_ds_pvalue", "bar_os_pvalue", "roi_size_um2"]
            feature_matrix = selected_df[feature_cols].to_numpy()
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
        jax_train_data = jnp.array(train_data)
        jax_test_data = jnp.array(test_data)

        # Create and return the immutable instance
        return cls(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            use_8hz_chirp=use_8hz_chirp,
            chirp_len=chirp_len,
            bar_len=bar_len,
            feature_len=feature_len,
            _train_data=jax_train_data,
            _test_data=jax_test_data,
        )

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
    def has_labels(self) -> bool:
        return False

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
        max_display_members = 1000
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
        chirp_ax = Axes(fig, gs[0, 0])  # pyright: ignore[reportArgumentType]
        bar_ax = Axes(fig, gs[0, 1])  # pyright: ignore[reportArgumentType]
        feat_ax = Axes(fig, gs[0, 2])  # pyright: ignore[reportArgumentType]

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
        chirp_ax = Axes(fig, gs[0, 0])  # pyright: ignore[reportArgumentType]
        bar_ax = Axes(fig, gs[0, 1])  # pyright: ignore[reportArgumentType]
        feat_ax = Axes(fig, gs[0, 2])  # pyright: ignore[reportArgumentType]

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

    @override
    def get_dataset_analyses(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> dict[str, Analysis[ClusteringDataset, Any, BadenBerens]]:
        """Return RGC-specific analyses."""

        # Load the raw dataframe for analyses that need it
        raw_df = pd.read_pickle(
            self.cache_dir / "neural-traces" / "raw" / f"{self.dataset_name}.pkl"
        )
        assert isinstance(raw_df, pd.DataFrame), "Expected DataFrame from pickle"

        return {
            "baden_berens_comparison": BadenBerensAnalysis(raw_df),
        }


@dataclass(frozen=True)
class BadenBerens(Artifact):
    """Baden-Berens style analysis of retinal ganglion cell clustering."""

    # Cluster assignments and metadata
    cluster_assignments: Array  # (n_cells,) cluster IDs
    celltype_labels: Array  # (n_cells,) ground truth cell type IDs

    # Response traces by cluster
    cluster_chirp_traces: Array  # (n_clusters, n_timepoints)
    cluster_bar_traces: Array  # (n_clusters, n_directions, n_timepoints)
    cluster_noise_traces: Array  # (n_clusters, n_timepoints) if available

    # Response metrics by cluster - NOW STORING ALL VALUES, NOT JUST MEANS
    cluster_rf_diameter_values: list[Array]  # List of arrays, one per cluster
    cluster_ds_index_values: list[Array]  # List of arrays, one per cluster
    cluster_os_index_values: list[Array]  # List of arrays, one per cluster
    cluster_soma_x: Array  # (n_clusters,) mean soma x position
    cluster_soma_y: Array  # (n_clusters,) mean soma y position

    # Cluster properties
    cluster_sizes: Array  # (n_clusters,) number of cells per cluster
    cluster_names: list[str]  # Descriptive names for each cluster

    # Hierarchical clustering
    linkage_matrix: Array  # Scipy linkage matrix for dendrogram

    # Quality metrics
    ari_score: float
    nmi_score: float

    def save_to_hdf5(self, file: File) -> None:
        """Save Baden-Berens analysis to HDF5."""
        file.create_dataset(
            "cluster_assignments", data=np.array(self.cluster_assignments)
        )
        file.create_dataset("celltype_labels", data=np.array(self.celltype_labels))

        file.create_dataset(
            "cluster_chirp_traces", data=np.array(self.cluster_chirp_traces)
        )
        file.create_dataset(
            "cluster_bar_traces", data=np.array(self.cluster_bar_traces)
        )
        file.create_dataset(
            "cluster_noise_traces", data=np.array(self.cluster_noise_traces)
        )

        # Save metric value lists as groups
        rf_group = file.create_group("cluster_rf_diameter_values")
        for i, vals in enumerate(self.cluster_rf_diameter_values):
            rf_group.create_dataset(f"cluster_{i}", data=np.array(vals))

        ds_group = file.create_group("cluster_ds_index_values")
        for i, vals in enumerate(self.cluster_ds_index_values):
            ds_group.create_dataset(f"cluster_{i}", data=np.array(vals))

        os_group = file.create_group("cluster_os_index_values")
        for i, vals in enumerate(self.cluster_os_index_values):
            os_group.create_dataset(f"cluster_{i}", data=np.array(vals))

        file.create_dataset("cluster_soma_x", data=np.array(self.cluster_soma_x))
        file.create_dataset("cluster_soma_y", data=np.array(self.cluster_soma_y))

        file.create_dataset("cluster_sizes", data=np.array(self.cluster_sizes))
        file.create_dataset("linkage_matrix", data=np.array(self.linkage_matrix))

        # Save cluster names as attributes
        for i, name in enumerate(self.cluster_names):
            file.attrs[f"cluster_name_{i}"] = name
        file.attrs["n_clusters"] = len(self.cluster_names)

        file.attrs["ari_score"] = self.ari_score
        file.attrs["nmi_score"] = self.nmi_score

    @classmethod
    def load_from_hdf5(cls, file: File) -> BadenBerens:
        """Load Baden-Berens analysis from HDF5."""
        # Load arrays
        cluster_assignments = jnp.array(file["cluster_assignments"][()])  # pyright: ignore[reportIndexIssue]
        celltype_labels = jnp.array(file["celltype_labels"][()])  # pyright: ignore[reportIndexIssue]

        cluster_chirp_traces = jnp.array(file["cluster_chirp_traces"][()])  # pyright: ignore[reportIndexIssue]
        cluster_bar_traces = jnp.array(file["cluster_bar_traces"][()])  # pyright: ignore[reportIndexIssue]
        cluster_noise_traces = jnp.array(file["cluster_noise_traces"][()])  # pyright: ignore[reportIndexIssue]

        # Load metric value lists
        n_clusters = int(file.attrs.get("n_clusters", 0))

        rf_group = file["cluster_rf_diameter_values"]
        assert isinstance(rf_group, Group)
        cluster_rf_diameter_values = [
            jnp.array(rf_group[f"cluster_{i}"]) for i in range(n_clusters)
        ]

        ds_group = file["cluster_ds_index_values"]
        assert isinstance(ds_group, Group)
        cluster_ds_index_values = [
            jnp.array(ds_group[f"cluster_{i}"]) for i in range(n_clusters)
        ]

        os_group = file["cluster_os_index_values"]
        assert isinstance(os_group, Group)
        cluster_os_index_values = [
            jnp.array(os_group[f"cluster_{i}"]) for i in range(n_clusters)
        ]

        cluster_soma_x = jnp.array(file["cluster_soma_x"][()])  # pyright: ignore[reportIndexIssue]
        cluster_soma_y = jnp.array(file["cluster_soma_y"][()])  # pyright: ignore[reportIndexIssue]

        cluster_sizes = jnp.array(file["cluster_sizes"][()])  # pyright: ignore[reportIndexIssue]
        linkage_matrix = jnp.array(file["linkage_matrix"][()])  # pyright: ignore[reportIndexIssue]

        # Load cluster names
        cluster_names = [str(file.attrs[f"cluster_name_{i}"]) for i in range(n_clusters)]

        ari_score = float(file.attrs.get("ari_score", 0.0))
        nmi_score = float(file.attrs.get("nmi_score", 0.0))

        return cls(
            cluster_assignments=cluster_assignments,
            celltype_labels=celltype_labels,
            cluster_chirp_traces=cluster_chirp_traces,
            cluster_bar_traces=cluster_bar_traces,
            cluster_noise_traces=cluster_noise_traces,
            cluster_rf_diameter_values=cluster_rf_diameter_values,
            cluster_ds_index_values=cluster_ds_index_values,
            cluster_os_index_values=cluster_os_index_values,
            cluster_soma_x=cluster_soma_x,
            cluster_soma_y=cluster_soma_y,
            cluster_sizes=cluster_sizes,
            cluster_names=cluster_names,
            linkage_matrix=linkage_matrix,
            ari_score=ari_score,
            nmi_score=nmi_score,
        )


@dataclass(frozen=True)
class BadenBerensAnalysis(Analysis[ClusteringDataset, Any, BadenBerens]):
    """Baden-Berens style analysis comparing clustering with ground truth cell types."""

    raw_df: pd.DataFrame  # Raw dataframe with all cell information

    @property
    @override
    def artifact_type(self) -> type[BadenBerens]:
        return BadenBerens

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> BadenBerens:
        """Generate Baden-Berens analysis from clustering results."""

        # Get cluster assignments
        assignments = model.cluster_assignments(params, dataset.train_data)
        n_clusters = model.n_clusters

        neural_dataset = dataset
        assert isinstance(neural_dataset, NeuralTracesDataset)

        # Use stored prototypes and parse them into components
        stored_prototypes = model.get_cluster_prototypes(handler, epoch)
        cluster_chirp_traces, cluster_bar_traces = self._parse_stored_prototypes(
            stored_prototypes, neural_dataset
        )

        # Use stored cluster members to extract feature values
        stored_members = model.get_cluster_members(handler, epoch)
        cluster_sizes = jnp.array([len(members) for members in stored_members])

        # Extract feature values from cluster members
        cluster_ds_pvalue_values = []
        cluster_os_pvalue_values = []
        cluster_roi_size_values = []

        chirp_len = neural_dataset.chirp_len
        bar_len = neural_dataset.bar_len

        for members in stored_members:
            if len(members) > 0:
                # Extract features from the end of each member's data
                features = members[:, chirp_len + bar_len :]  # Shape: (n_members, 3)
                cluster_ds_pvalue_values.append(features[:, 0])
                cluster_os_pvalue_values.append(features[:, 1])
                cluster_roi_size_values.append(features[:, 2])
            else:
                cluster_ds_pvalue_values.append(jnp.array([]))
                cluster_os_pvalue_values.append(jnp.array([]))
                cluster_roi_size_values.append(jnp.array([]))

        # Get celltype labels if available (otherwise use dummy labels)
        if hasattr(self, "raw_df") and "celltype" in self.raw_df.columns:
            # Try to get labels, but if we can't match them, use dummy labels
            quality_mask = (self.raw_df.chirp_qidx > 0.35) | (
                self.raw_df.bar_qidx > 0.6
            )
            filtered_df = self.raw_df.loc[quality_mask].reset_index(drop=True)
            n_train = len(assignments)
            if len(filtered_df) >= n_train:
                celltype_labels = np.asarray(filtered_df["celltype"])[:n_train]
            else:
                celltype_labels = np.zeros(n_train)  # Dummy labels
        else:
            celltype_labels = np.zeros(len(assignments))  # Dummy labels

        # For spatial coordinates, we'll use zeros as placeholders
        cluster_soma_x = jnp.zeros(n_clusters)
        cluster_soma_y = jnp.zeros(n_clusters)

        # Generate cluster names based on chirp responses
        cluster_names = self._generate_simple_cluster_names(cluster_chirp_traces)

        # Use stored hierarchical clustering
        linkage_matrix = model.get_cluster_hierarchy(handler, epoch)

        # Calculate quality metrics
        ari_score = adjusted_rand_score(
            np.array(celltype_labels), np.array(assignments)
        )
        nmi_score = normalized_mutual_info_score(
            np.array(celltype_labels), np.array(assignments)
        )

        # Create noise traces placeholder
        cluster_noise_traces = jnp.zeros((n_clusters, 100))

        return BadenBerens(
            cluster_assignments=assignments,
            celltype_labels=jnp.array(celltype_labels),
            cluster_chirp_traces=cluster_chirp_traces,
            cluster_bar_traces=cluster_bar_traces,
            cluster_noise_traces=cluster_noise_traces,
            cluster_rf_diameter_values=cluster_roi_size_values,  # Reuse field for ROI size
            cluster_ds_index_values=cluster_ds_pvalue_values,  # Store p-values
            cluster_os_index_values=cluster_os_pvalue_values,  # Store p-values
            cluster_soma_x=cluster_soma_x,
            cluster_soma_y=cluster_soma_y,
            cluster_sizes=cluster_sizes,
            cluster_names=cluster_names,
            linkage_matrix=linkage_matrix,
            ari_score=float(ari_score),
            nmi_score=float(nmi_score),
        )

    def _parse_stored_prototypes(
        self, prototypes: list[Array], dataset: NeuralTracesDataset
    ) -> tuple[Array, Array]:
        """Parse stored prototypes into chirp and bar components."""
        chirp_traces = []
        bar_traces = []

        for proto in prototypes:
            # Parse based on dataset structure: [chirp | bar | features]
            chirp_response = proto[: dataset.chirp_len]
            bar_response = proto[
                dataset.chirp_len : dataset.chirp_len + dataset.bar_len
            ]

            chirp_traces.append(chirp_response)
            # Replicate bar response for all directions (placeholder)
            n_directions = 8
            directional_bar = jnp.tile(bar_response, (n_directions, 1))
            bar_traces.append(directional_bar)

        return jnp.array(chirp_traces), jnp.array(bar_traces)

    @override
    def plot(self, artifact: BadenBerens, dataset: ClusteringDataset) -> Figure:
        """Create Baden-Berens style visualization."""
        n_clusters = len(artifact.cluster_names)

        # Scale figure height with number of clusters
        fig_height = max(12, n_clusters * 0.3)  # ~0.3 inches per cluster
        fig = plt.figure(figsize=(20, fig_height))

        # Adjust grid to account for variable height
        gs = GridSpec(
            n_clusters,
            4,
            figure=fig,
            width_ratios=[0.8, 2.5, 1.5, 1.5],
            hspace=0.02,
            wspace=0.15,
        )

        # A. Dendrogram (spans all rows) - GET THE LEAF ORDER
        ax_dendro = fig.add_subplot(gs[:, 0])
        dendrogram_leaf_order = self._plot_dendrogram(ax_dendro, artifact)

        # B. Response traces (spans all rows) - USE THE LEAF ORDER
        ax_chirp = fig.add_subplot(gs[:, 1])
        ax_bar = fig.add_subplot(gs[:, 2])
        self._plot_response_traces(ax_chirp, ax_bar, artifact, dendrogram_leaf_order)

        # C. Response metrics - USE THE LEAF ORDER
        self._plot_cluster_histograms(fig, gs, artifact, dendrogram_leaf_order)

        # Overall title - simplified without metrics
        fig.suptitle(
            "RGC Functional Classification",
            fontsize=16,
            y=0.99,
        )

        plt.subplots_adjust(top=0.96)
        return fig

    def _plot_dendrogram(self, ax: Axes, artifact: BadenBerens) -> list[int]:
        """Plot hierarchical clustering dendrogram."""
        # Create dendrogram with colors matching clusters
        # Ensure linkage matrix is numpy float64
        linkage_matrix = np.array(artifact.linkage_matrix, dtype=np.float64)

        dendro = sch.dendrogram(
            linkage_matrix,
            orientation="left",
            ax=ax,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="k",
        )

        # Get the reordered indices
        leaves = dendro["leaves"]
        if leaves is None:
            leaves = list(range(len(artifact.cluster_names)))

        # Color the leaf nodes according to cluster colors
        ax.set_xlabel("Distance")
        ax.set_title("Hierarchical Clustering")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])

        return leaves

    def palette(self, n_colors: int):
        """Generate a color palette for clusters."""
        return seaborn.cubehelix_palette(
            n_colors=n_colors, start=0.5, rot=-1.5, dark=0.3, light=0.7, reverse=True
        )

    def _plot_response_traces(
        self, ax_chirp: Axes, ax_bar: Axes, artifact: BadenBerens, leaf_order: list[int]
    ) -> None:
        """Plot chirp and bar response traces above baseline."""
        n_clusters = len(leaf_order)

        # Calculate trace spacing
        trace_spacing = 1.0 / n_clusters
        traces_min = jnp.min(artifact.cluster_chirp_traces)
        traces_max = jnp.max(artifact.cluster_chirp_traces)
        traces_range = traces_max - traces_min

        # Set2 color palette for clusters
        palette = self.palette(n_clusters)

        # Plot chirp responses
        for i, cluster_idx in enumerate(leaf_order):
            color = palette[i]
            y_position = 1.0 - (i + 0.5) * trace_spacing

            trace = artifact.cluster_chirp_traces[cluster_idx]
            trace_baselined = trace - jnp.min(trace)
            scale_factor = trace_spacing * 0.9
            trace_scaled = (trace_baselined / traces_range) * scale_factor
            trace_final = trace_scaled + y_position

            # Create time axis
            time_points = np.linspace(0, 1, len(trace))

            # Plot trace
            ax_chirp.plot(time_points, trace_final, color=color, linewidth=1.2)

            # Fill only positive deviations from baseline
            trace_final_np = np.asarray(trace_final)
            ax_chirp.fill_between(
                time_points,
                y_position,
                trace_final_np,
                where=(trace_final_np > y_position).tolist(),
                color=color,
                alpha=0.3,
                linewidth=0,
            )

            # Add cluster label
            ax_chirp.text(
                -0.02,
                y_position,
                f"{cluster_idx}",
                ha="right",
                va="center",
                fontsize=8,
                transform=ax_chirp.transData,
            )

        # Style chirp axis
        ax_chirp.set_xlim(0, 1)
        ax_chirp.set_ylim(0, 1)
        ax_chirp.set_xlabel("Time (normalized)")
        ax_chirp.set_title("8Hz Chirp Response")
        ax_chirp.spines["top"].set_visible(False)
        ax_chirp.spines["right"].set_visible(False)
        ax_chirp.set_yticks([])

        traces_min = jnp.min(artifact.cluster_bar_traces)
        traces_max = jnp.max(artifact.cluster_bar_traces)
        traces_range = traces_max - traces_min

        # Plot bar responses similarly
        for i, cluster_idx in enumerate(leaf_order):
            color = palette[i]
            y_position = 1.0 - (i + 0.5) * trace_spacing

            trace = jnp.mean(artifact.cluster_bar_traces[cluster_idx], axis=0)
            trace_min = jnp.min(trace)
            trace_baselined = trace - trace_min

            # Scale trace
            scale_factor = trace_spacing * 0.9
            trace_scaled = (trace_baselined / traces_range) * scale_factor

            trace_final = trace_scaled + y_position

            time_points = np.linspace(0, 1, len(trace))

            trace_final_bar_np = np.asarray(trace_final)
            ax_bar.plot(time_points, trace_final_bar_np, color=color, linewidth=1.2)
            ax_bar.fill_between(
                time_points,
                y_position,
                trace_final_bar_np,
                where=(trace_final_bar_np > y_position).tolist(),
                color=color,
                alpha=0.3,
                linewidth=0,
            )

        # Style bar axis
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xlabel("Time (normalized)")
        ax_bar.set_title("Moving Bar Response")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.set_yticks([])

    def _plot_cluster_histograms(
        self, fig: Figure, gs: GridSpec, artifact: BadenBerens, leaf_order: list[int]
    ) -> None:
        """Plot individual histograms for each cluster's metrics."""
        # Update labels to match what we're actually plotting
        metrics_info = [
            ("roi_size", "ROI Size (μm²)", (0, 400)),
            ("ds_pvalue", "DS p-value", (0, 1)),
            ("os_pvalue", "OS p-value", (0, 1)),
        ]
        palette = self.palette(len(leaf_order))

        # Collect all values across all clusters for background histograms
        all_roi_values = jnp.concatenate(
            [vals for vals in artifact.cluster_rf_diameter_values if len(vals) > 0]
        )
        all_ds_values = jnp.concatenate(
            [vals for vals in artifact.cluster_ds_index_values if len(vals) > 0]
        )
        all_os_values = jnp.concatenate(
            [vals for vals in artifact.cluster_os_index_values if len(vals) > 0]
        )
        all_values_dict = {
            "roi_size": all_roi_values,
            "ds_pvalue": all_ds_values,
            "os_pvalue": all_os_values,
        }

        # Create histograms for each cluster
        for i, cluster_idx in enumerate(leaf_order):
            color = palette[i]
            cluster_idx = int(cluster_idx)

            # Create subplot spanning the metrics column
            outer_ax = fig.add_subplot(gs[i, 3])
            outer_ax.axis("off")

            inner_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[i, 3], wspace=0.25)

            for j, (metric_key, label, xlim) in enumerate(metrics_info):
                ax_hist = fig.add_subplot(inner_gs[0, j])

                # Get cluster-specific values
                if metric_key == "roi_size":
                    cluster_vals = artifact.cluster_rf_diameter_values[cluster_idx]
                elif metric_key == "ds_pvalue":
                    cluster_vals = artifact.cluster_ds_index_values[cluster_idx]
                else:
                    cluster_vals = artifact.cluster_os_index_values[cluster_idx]

                # Plot background histogram of all values
                all_vals = all_values_dict[metric_key]
                bins: list[float] = np.linspace(xlim[0], xlim[1], 20).tolist()
                if len(all_vals) > 0:
                    ax_hist.hist(
                        np.asarray(all_vals),
                        bins=bins,
                        density=True,
                        alpha=0.2,
                        color="gray",
                        edgecolor="none",
                        label="All cells",
                    )

                # Plot cluster-specific histogram
                if len(cluster_vals) > 0:
                    ax_hist.hist(
                        np.asarray(cluster_vals),
                        bins=bins,
                        density=True,
                        alpha=0.7,
                        color=color,
                        edgecolor="none",
                        label=f"Cluster {cluster_idx}",
                    )
                ax_hist.set_xlim(xlim)
                ax_hist.spines["top"].set_visible(False)
                ax_hist.spines["right"].set_visible(False)
                ax_hist.spines["left"].set_visible(False)
                ax_hist.set_yticks([])

                if j == 0:  # ROI size
                    ax_hist.set_xticks([0, 200, 400])
                    ax_hist.set_xticklabels(["0", "200", "400"], fontsize=7)
                else:  # p-values
                    ax_hist.set_xticks([0, 0.5, 1])
                    ax_hist.set_xticklabels(["0", "0.5", "1"], fontsize=7)

                if i == len(leaf_order) - 1:
                    ax_hist.set_xlabel(label, fontsize=8)

            # Add cluster size on the right
            n_cells = int(artifact.cluster_sizes[cluster_idx])
            outer_ax.text(
                0.95,
                0.5,
                f"n={n_cells}",
                transform=outer_ax.transAxes,
                fontsize=8,
                ha="right",
                va="center",
            )

    def _generate_simple_cluster_names(self, chirp_traces: Array) -> list[str]:
        """Generate simple descriptive names for clusters based on chirp response."""
        names = []
        for i in range(len(chirp_traces)):
            # Basic ON/OFF classification from chirp response
            trace = chirp_traces[i]
            on_response = jnp.mean(trace[100:150])
            off_response = jnp.mean(trace[150:200])

            if off_response > on_response + 0.1:
                base_name = "OFF"
            elif on_response > off_response + 0.1:
                base_name = "ON"
            else:
                base_name = "ON-OFF"

            names.append(f"{base_name} {i}")

        return names

    @override
    def metrics(self, artifact: BadenBerens) -> MetricDict:
        """Return quality metrics."""
        # Import here to ensure proper type
        return {}
