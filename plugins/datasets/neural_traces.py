"""Neural trace dataset implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from h5py import File
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from apps.configs import ClusteringDatasetConfig
from apps.plugins import Analysis, ClusteringDataset, ClusteringExperiment
from apps.runtime.handler import Artifact, MetricDict


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
    def get_dataset_analyses(self) -> dict[str, Analysis[Self, Any, Any]]:
        """Return RGC-specific analyses."""

        # Load the raw dataframe for analyses that need it
        raw_df = pd.read_pickle(
            self.cache_dir / "neural-traces" / "raw" / f"{self.dataset_name}.pkl"
        )

        return {
            "baden_berens_comparison": BadenBerensComparisonAnalysis(raw_df),
        }


"""Baden-Berens style analyses for neural traces dataset."""


@dataclass(frozen=True)
class BadenBerensComparison(Artifact):
    """Comparison between clustering results and reference cell types."""

    # Core comparison data
    confusion_matrix: Array  # (n_clusters, n_celltypes)
    cluster_to_celltype: dict[int, dict[int, float]]  # cluster -> {celltype: fraction}
    celltype_to_clusters: dict[int, dict[int, float]]  # celltype -> {cluster: fraction}

    # Averaged response profiles by cluster
    cluster_chirp_responses: Array  # (n_clusters, n_timepoints)
    cluster_bar_responses: Array  # (n_clusters, n_timepoints)

    # Response metrics by cluster
    cluster_metrics: dict[str, Array]  # metric_name -> (n_clusters,)

    # Metadata
    cluster_sizes: Array  # Number of cells per cluster
    celltype_labels: list[int]  # Unique cell type IDs
    ari_score: float
    nmi_score: float

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save comparison data to HDF5."""
        file.create_dataset("confusion_matrix", data=np.array(self.confusion_matrix))
        file.create_dataset(
            "cluster_chirp_responses", data=np.array(self.cluster_chirp_responses)
        )
        file.create_dataset(
            "cluster_bar_responses", data=np.array(self.cluster_bar_responses)
        )
        file.create_dataset("cluster_sizes", data=np.array(self.cluster_sizes))

        # Save dictionaries as groups
        c2ct_group = file.create_group("cluster_to_celltype")
        for cluster_id, celltype_dict in self.cluster_to_celltype.items():
            cluster_group = c2ct_group.create_group(str(cluster_id))
            for celltype, fraction in celltype_dict.items():
                cluster_group.attrs[str(celltype)] = fraction

        # Save metrics
        metrics_group = file.create_group("cluster_metrics")
        for metric_name, values in self.cluster_metrics.items():
            metrics_group.create_dataset(metric_name, data=np.array(values))

        # Save metadata
        file.attrs["celltype_labels"] = self.celltype_labels
        file.attrs["ari_score"] = self.ari_score
        file.attrs["nmi_score"] = self.nmi_score

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> BadenBerensComparison:
        """Load comparison data from HDF5."""
        # Load array datasets
        confusion_matrix = jnp.array(file["confusion_matrix"][()])  # pyright: ignore[reportArgumentType,reportIndexIssue]
        cluster_chirp_responses = jnp.array(file["cluster_chirp_responses"][()])  # pyright: ignore[reportArgumentType,reportIndexIssue]
        cluster_bar_responses = jnp.array(file["cluster_bar_responses"][()])  # pyright: ignore[reportArgumentType,reportIndexIssue]
        cluster_sizes = jnp.array(file["cluster_sizes"][()])  # pyright: ignore[reportArgumentType,reportIndexIssue]

        # Load cluster_to_celltype mapping
        cluster_to_celltype = {}
        c2ct_group = file["cluster_to_celltype"]
        for cluster_id_str in c2ct_group.keys():
            cluster_id = int(cluster_id_str)
            cluster_group = c2ct_group[cluster_id_str]
            celltype_dict = {}
            for celltype_str in cluster_group.attrs.keys():
                celltype = int(celltype_str)
                fraction = float(cluster_group.attrs[celltype_str])
                celltype_dict[celltype] = fraction
            cluster_to_celltype[cluster_id] = celltype_dict

        # Reconstruct celltype_to_clusters from the confusion matrix
        # This is more reliable than saving/loading it separately
        celltype_to_clusters = {}
        unique_celltypes = list(file.attrs["celltype_labels"])
        n_clusters, n_celltypes = confusion_matrix.shape

        for ct_idx, celltype in enumerate(unique_celltypes):
            celltype_size = np.sum(confusion_matrix[:, ct_idx])
            if celltype_size > 0:
                celltype_dict = {}
                for cluster_id in range(n_clusters):
                    fraction = float(
                        confusion_matrix[cluster_id, ct_idx] / celltype_size
                    )
                    if fraction > 0:
                        celltype_dict[cluster_id] = fraction
                celltype_to_clusters[int(celltype)] = celltype_dict

        # Load cluster metrics
        cluster_metrics = {}
        metrics_group = file["cluster_metrics"]
        for metric_name in metrics_group:
            cluster_metrics[metric_name] = jnp.array(metrics_group[metric_name][()])  # pyright: ignore[reportArgumentType,reportIndexIssue]

        # Load metadata attributes
        celltype_labels = list(file.attrs["celltype_labels"])  # pyright: ignore[reportArgumentType]
        ari_score = float(file.attrs["ari_score"])  # pyright: ignore[reportArgumentType]
        nmi_score = float(file.attrs["nmi_score"])  # pyright: ignore[reportArgumentType]

        return cls(
            confusion_matrix=confusion_matrix,
            cluster_to_celltype=cluster_to_celltype,
            celltype_to_clusters=celltype_to_clusters,
            cluster_chirp_responses=cluster_chirp_responses,
            cluster_bar_responses=cluster_bar_responses,
            cluster_metrics=cluster_metrics,
            cluster_sizes=cluster_sizes,
            celltype_labels=celltype_labels,
            ari_score=ari_score,
            nmi_score=nmi_score,
        )


@dataclass(frozen=True)
class BadenBerensComparisonAnalysis(
    Analysis[ClusteringDataset, ClusteringExperiment, BadenBerensComparison]
):
    """Analysis comparing clustering results with reference cell types."""

    raw_df: pd.DataFrame  # Raw dataframe with cell type labels

    @property
    @override
    def artifact_type(self) -> type[BadenBerensComparison]:
        return BadenBerensComparison

    @override
    def generate(
        self,
        experiment: ClusteringExperiment,
        params: Array,
        dataset: ClusteringDataset,
        key: Array,
    ) -> BadenBerensComparison:
        """Generate comparison between clustering and reference cell types."""

        # Get cluster assignments from the model
        assignments = experiment.cluster_assignments(params, dataset.train_data)
        n_clusters = experiment.n_clusters

        # Extract cell types from raw dataframe
        # Assuming the ordering matches between processed dataset and raw df
        cell_types = self.raw_df["celltype"].values[: len(assignments)]
        unique_celltypes = np.unique(cell_types)
        n_celltypes = len(unique_celltypes)

        # Build confusion matrix
        confusion = np.zeros((n_clusters, n_celltypes))
        for cluster_id, celltype in zip(assignments, cell_types):
            celltype_idx = np.where(unique_celltypes == celltype)[0][0]
            confusion[int(cluster_id), celltype_idx] += 1

        # Calculate cluster sizes
        cluster_sizes = np.sum(confusion, axis=1)

        # Build cluster to celltype mapping
        cluster_to_celltype = {}
        for cluster_id in range(n_clusters):
            if cluster_sizes[cluster_id] > 0:
                fractions = confusion[cluster_id] / cluster_sizes[cluster_id]
                cluster_to_celltype[int(cluster_id)] = {
                    int(unique_celltypes[i]): float(fractions[i])
                    for i in range(n_celltypes)
                    if fractions[i] > 0
                }

        # Build celltype to clusters mapping
        celltype_sizes = np.sum(confusion, axis=0)
        celltype_to_clusters = {}
        for ct_idx, celltype in enumerate(unique_celltypes):
            if celltype_sizes[ct_idx] > 0:
                fractions = confusion[:, ct_idx] / celltype_sizes[ct_idx]
                celltype_to_clusters[int(celltype)] = {
                    int(cluster_id): float(fractions[cluster_id])
                    for cluster_id in range(n_clusters)
                    if fractions[cluster_id] > 0
                }

        # Extract average response profiles by cluster
        chirp_responses = []
        bar_responses = []

        for cluster_id in range(n_clusters):
            cluster_mask = assignments == cluster_id
            if np.sum(cluster_mask) > 0:
                # Get indices of cells in this cluster
                cluster_indices = np.where(cluster_mask)[0]

                # Average chirp responses (using 8Hz chirp)
                chirp_data = [
                    self.raw_df["chirp_8Hz_average_norm"].iloc[i]
                    for i in cluster_indices
                ]
                avg_chirp = np.mean(chirp_data, axis=0)
                chirp_responses.append(avg_chirp)

                # Average bar responses
                bar_data = [self.raw_df["preproc_bar"].iloc[i] for i in cluster_indices]
                avg_bar = np.mean(bar_data, axis=0)
                bar_responses.append(avg_bar)
            else:
                # Empty cluster
                chirp_responses.append(np.zeros(259))  # 8Hz chirp length
                bar_responses.append(np.zeros(32))  # bar response length

        # Extract response metrics by cluster
        cluster_metrics = {}
        metric_fields = ["rf_cdia_um", "bar_ds_index", "bar_os_index"]

        for metric in metric_fields:
            metric_values = []
            for cluster_id in range(n_clusters):
                cluster_mask = assignments == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_indices = np.where(cluster_mask)[0]
                    values = [self.raw_df[metric].iloc[i] for i in cluster_indices]
                    # Filter out NaN values
                    valid_values = [v for v in values if not np.isnan(v)]
                    if valid_values:
                        metric_values.append(np.mean(valid_values))
                    else:
                        metric_values.append(np.nan)
                else:
                    metric_values.append(np.nan)
            cluster_metrics[metric] = jnp.array(metric_values)

        # Calculate comparison scores
        ari = adjusted_rand_score(cell_types, assignments)
        nmi = normalized_mutual_info_score(cell_types, assignments)

        return BadenBerensComparison(
            confusion_matrix=jnp.array(confusion),
            cluster_to_celltype=cluster_to_celltype,
            celltype_to_clusters=celltype_to_clusters,
            cluster_chirp_responses=jnp.array(chirp_responses),
            cluster_bar_responses=jnp.array(bar_responses),
            cluster_metrics=cluster_metrics,
            cluster_sizes=jnp.array(cluster_sizes),
            celltype_labels=list(unique_celltypes),
            ari_score=float(ari),
            nmi_score=float(nmi),
        )

    @override
    def plot(
        self, artifact: BadenBerensComparison, dataset: ClusteringDataset
    ) -> Figure:
        """Create Baden-Berens style visualization."""

        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(60, 12, figure=fig, hspace=0.4, wspace=0.3)

        # Create a reordering based on response properties
        # This is a simplified version - you might want more sophisticated ordering
        cluster_order = self._order_clusters_by_response(artifact)

        # Left: Response traces
        ax_chirp = fig.add_subplot(gs[:, 2:4])
        ax_bar = fig.add_subplot(gs[:, 4:6])

        # Plot responses for each cluster
        for i, cluster_id in enumerate(cluster_order):
            y_pos = len(cluster_order) - i - 1

            # Chirp response
            chirp = artifact.cluster_chirp_responses[cluster_id]
            ax_chirp.plot(
                chirp + y_pos * 2, color=self._get_cluster_color(cluster_id, artifact)
            )

            # Bar response
            bar = artifact.cluster_bar_responses[cluster_id]
            ax_bar.plot(
                bar + y_pos * 2, color=self._get_cluster_color(cluster_id, artifact)
            )

        ax_chirp.set_title("8Hz Chirp Response")
        ax_bar.set_title("Bar Response")

        # Right: Response metrics
        ax_rf = fig.add_subplot(gs[0:20, 8:10])
        ax_dsi = fig.add_subplot(gs[20:40, 8:10])
        ax_osi = fig.add_subplot(gs[40:60, 8:10])

        # Plot histograms
        self._plot_metric_histogram(
            ax_rf,
            artifact.cluster_metrics["rf_cdia_um"],
            "RF Diameter (μm)",
            cluster_order,
            artifact,
        )
        self._plot_metric_histogram(
            ax_dsi,
            artifact.cluster_metrics["bar_ds_index"],
            "DS Index",
            cluster_order,
            artifact,
        )
        self._plot_metric_histogram(
            ax_osi,
            artifact.cluster_metrics["bar_os_index"],
            "OS Index",
            cluster_order,
            artifact,
        )

        # Add confusion matrix
        ax_conf = fig.add_subplot(gs[45:60, 0:2])
        self._plot_confusion_summary(ax_conf, artifact)

        plt.suptitle(
            f"Clustering Analysis (ARI={artifact.ari_score:.3f}, NMI={artifact.nmi_score:.3f})"
        )

        return fig

    @override
    def metrics(self, artifact: BadenBerensComparison) -> MetricDict:
        """Return comparison metrics."""
        return {
            "Comparison/ARI Score": (jnp.array(0), jnp.array(artifact.ari_score)),
            "Comparison/NMI Score": (jnp.array(0), jnp.array(artifact.nmi_score)),
            "Comparison/Mean Cluster Size": (
                jnp.array(0),
                jnp.mean(artifact.cluster_sizes),
            ),
        }

    def _order_clusters_by_response(self, artifact: BadenBerensComparison) -> list[int]:
        """Order clusters by response properties for visualization."""
        # Simple ordering by ON/OFF index - you can make this more sophisticated
        on_off_indices = []
        for i in range(len(artifact.cluster_chirp_responses)):
            chirp = artifact.cluster_chirp_responses[i]
            # Simple ON/OFF calculation based on response polarity
            on_off_index = np.mean(chirp[100:150]) - np.mean(chirp[50:100])
            on_off_indices.append(on_off_index)

        # Sort from OFF (negative) to ON (positive)
        return list(np.argsort(on_off_indices))

    def _get_cluster_color(
        self, cluster_id: int, artifact: BadenBerensComparison
    ) -> str:
        """Assign color based on cluster properties."""
        # This is a simplified version - you might want to base this on
        # dominant cell type or response properties
        chirp = artifact.cluster_chirp_responses[cluster_id]
        on_off_index = np.mean(chirp[100:150]) - np.mean(chirp[50:100])

        if on_off_index < -0.5:
            return "red"  # OFF
        if on_off_index > 0.5:
            return "blue"  # ON
        return "green"  # ON-OFF

    def _plot_metric_histogram(
        self,
        ax,
        values: Array,
        title: str,
        cluster_order: list[int],
        artifact: BadenBerensComparison,
    ):
        """Plot histogram of metric values colored by cluster."""
        # Implementation for metric histograms
        pass

    def _plot_confusion_summary(self, ax, artifact: BadenBerensComparison):
        """Plot confusion matrix summary."""
        # Normalize by columns (cell types)
        confusion_norm = artifact.confusion_matrix / np.sum(
            artifact.confusion_matrix, axis=0
        )

        # Show top clusters for each cell type
        sns.heatmap(
            confusion_norm, ax=ax, cmap="YlOrRd", cbar_kws={"label": "Fraction"}
        )
        ax.set_xlabel("Cell Type")
        ax.set_ylabel("Cluster")
