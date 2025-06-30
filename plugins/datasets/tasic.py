"""Tasic et al. (2018) single-cell RNA-seq dataset implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec

from apps.interface import ClusteringDataset, ClusteringDatasetConfig


@dataclass
class TasicConfig(ClusteringDatasetConfig):
    """Configuration for Tasic dataset.

    Parameters:
        n_genes: Target number of genes to keep after Fano factor selection
        use_fano_selection: Whether to use Fano factor ranking for gene selection
        min_expression_threshold: Minimum expression level to consider a gene
        log_transform: Whether to apply log(counts + 1) transformation
        standardize: Whether to center and scale features
        test_fraction: Fraction of data to use for test set
        random_seed: Random seed for train/test split
        n_global_genes: Number of genes to show in global profile plot
        n_marker_genes: Number of cluster-specific marker genes to show
    """

    _target_: str = "plugins.datasets.tasic.TasicDataset.load"

    # Gene selection
    n_genes: int = 3000
    use_fano_selection: bool = True
    min_expression_threshold: float = 0.0

    # Preprocessing
    log_transform: bool = True
    standardize: bool = False

    # Data splits
    test_fraction: float = 0.2
    random_seed: int = 42

    # Visualization
    n_global_genes: int = 100  # Genes to show in global profile
    n_marker_genes: int = 8  # Cluster-specific markers to show


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="tasic", node=TasicConfig)


@dataclass(frozen=True)
class TasicDataset(ClusteringDataset):
    """Tasic et al. (2018) single-cell RNA-seq dataset for mouse visual cortex."""

    # Configuration
    n_genes: int
    use_fano_selection: bool
    min_expression_threshold: float
    log_transform: bool
    standardize: bool
    n_global_genes: int
    n_marker_genes: int

    # Data
    _train_data: Array
    _test_data: Array
    _train_labels: Array
    _test_labels: Array

    # Metadata for visualization
    _gene_names: list[str]  # Names of selected genes
    _cell_type_names: list[str]  # Names of cell types
    _global_gene_indices: Array  # Indices of genes for global profile
    _gene_means: Array  # Mean expression per gene (for marker calculation)
    _gene_stds: Array  # Std expression per gene (for marker calculation)

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        n_genes: int = 3000,
        use_fano_selection: bool = True,
        min_expression_threshold: float = 0.0,
        log_transform: bool = True,
        standardize: bool = True,
        test_fraction: float = 0.2,
        random_seed: int = 42,
        n_global_genes: int = 100,
        n_marker_genes: int = 8,
    ) -> "TasicDataset":
        """Load Tasic dataset.

        Args:
            cache_dir: Directory for caching downloaded data
            n_genes: Target number of genes to keep
            use_fano_selection: Use Fano factor for gene selection
            min_expression_threshold: Minimum expression threshold
            log_transform: Apply log transformation
            standardize: Standardize features
            test_fraction: Fraction for test set
            random_seed: Random seed
            n_global_genes: Number of genes for global profile
            n_marker_genes: Number of marker genes per cluster

        Returns:
            Loaded Tasic dataset
        """
        data_file = cache_dir / "tasic_data.h5"

        # Download data
        _download_tasic_data(data_file)

        # Load data from HDF5 file
        import h5py

        with h5py.File(data_file, "r") as f:
            expression_data = f["expression"][:]  # cells x genes
            gene_names = [
                g.decode() if isinstance(g, bytes) else g for g in f["gene_names"][:]
            ]
            cell_labels = f["cell_labels"][:]
            cell_type_names = [
                c.decode() if isinstance(c, bytes) else c
                for c in f["cell_type_names"][:]
            ]

        # Convert to numpy arrays
        expression_data = np.array(expression_data, dtype=np.float32)
        cell_labels = np.array(cell_labels, dtype=np.int32)

        print(
            f"Loaded Tasic dataset: {expression_data.shape[0]} cells, {expression_data.shape[1]} genes"
        )
        print(f"Found {len(cell_type_names)} cell types")

        # Apply minimum expression threshold
        if min_expression_threshold > 0:
            gene_mask = (expression_data > min_expression_threshold).mean(axis=0) > 0.05
            expression_data = expression_data[:, gene_mask]
            gene_names = [gene_names[i] for i in range(len(gene_names)) if gene_mask[i]]
            print(f"After expression filtering: {len(gene_names)} genes")

        # Gene selection using Fano factor
        if use_fano_selection and n_genes < expression_data.shape[1]:
            gene_means = expression_data.mean(axis=0)
            gene_vars = expression_data.var(axis=0)
            fano_factors = gene_vars / (gene_means + 1e-8)  # Avoid division by zero

            # Select top genes by Fano factor
            top_gene_indices = np.argsort(fano_factors)[-n_genes:]
            expression_data = expression_data[:, top_gene_indices]
            gene_names = [gene_names[i] for i in top_gene_indices]
            print(f"After Fano factor selection: {len(gene_names)} genes")

        # Log transformation
        if log_transform:
            expression_data = np.log1p(expression_data)  # log(1 + x)
            print("Applied log transformation")

        # Standardization
        if standardize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            expression_data = scaler.fit_transform(expression_data)
            print("Applied standardization")

        # Convert to JAX arrays
        expression_data = jnp.array(expression_data)
        cell_labels = jnp.array(cell_labels)

        # Create train/test split
        n_cells = expression_data.shape[0]
        rng = np.random.RandomState(random_seed)
        test_indices = rng.choice(
            n_cells, size=int(n_cells * test_fraction), replace=False
        )
        train_indices = np.setdiff1d(np.arange(n_cells), test_indices)

        train_data = expression_data[train_indices]
        test_data = expression_data[test_indices]
        train_labels = cell_labels[train_indices]
        test_labels = cell_labels[test_indices]

        # Compute metadata for visualization
        gene_means = train_data.mean(axis=0)
        gene_stds = train_data.std(axis=0)

        # Select genes for global profile (highest variance)
        global_gene_indices = jnp.argsort(gene_stds)[-n_global_genes:]

        return cls(
            cache_dir=cache_dir,
            n_genes=n_genes,
            use_fano_selection=use_fano_selection,
            min_expression_threshold=min_expression_threshold,
            log_transform=log_transform,
            standardize=standardize,
            n_global_genes=n_global_genes,
            n_marker_genes=n_marker_genes,
            _train_data=train_data,
            _test_data=test_data,
            _train_labels=train_labels,
            _test_labels=test_labels,
            _gene_names=gene_names,
            _cell_type_names=cell_type_names,
            _global_gene_indices=global_gene_indices,
            _gene_means=gene_means,
            _gene_stds=gene_stds,
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
        return True

    @property
    @override
    def train_labels(self) -> Array:
        return self._train_labels

    @property
    @override
    def test_labels(self) -> Array:
        return self._test_labels

    @property
    @override
    def data_dim(self) -> int:
        return self._train_data.shape[1]

    @property
    @override
    def observable_shape(self) -> tuple[int, int]:
        # For gene expression, we'll visualize as a 1D profile
        # Height=1 for single profile, width=number of genes in global view
        return (1, self.n_global_genes)

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        # Give more space for the two-panel layout
        return (4, self.n_global_genes + 20)  # Height for profile + markers

    @property
    @override
    def n_classes(self) -> int:
        return len(self._cell_type_names)

    @override
    def paint_observable(self, observable: Array, axes: Axes):
        """Visualize a single cell's gene expression profile."""
        # Show global gene profile
        global_profile = observable[self._global_gene_indices]

        axes.plot(global_profile, color="black", linewidth=1)
        axes.set_xlabel("Gene Index (by Variance)")
        axes.set_ylabel("Expression")
        axes.set_title("Gene Expression Profile")
        axes.grid(True, alpha=0.3)

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Visualize gene expression cluster with two-stage approach."""

        # Turn off the main axes frame
        axes.set_axis_off()

        # Get subplot specification and figure
        subplot_spec = axes.get_subplotspec()
        fig = axes.get_figure()

        if subplot_spec is None:
            raise ValueError("paint_cluster requires a subplot")

        assert isinstance(fig, Figure)

        # Create a grid layout: global profile on left, marker genes on right
        gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, width_ratios=[2, 1], wspace=0.3
        )

        # Create axes for global profile and marker genes
        profile_ax = fig.add_subplot(gs[0, 0])
        marker_ax = fig.add_subplot(gs[0, 1])

        # === LEFT PANEL: Global Gene Profile ===

        # Extract global gene profiles
        prototype_profile = prototype[self._global_gene_indices]

        # Plot member profiles with transparency
        n_members = members.shape[0]
        n_display_members = min(50, n_members)  # Limit for performance

        if n_display_members > 0:
            # Sample members to display
            if n_members > n_display_members:
                indices = np.random.choice(n_members, n_display_members, replace=False)
                display_members = members[indices]
            else:
                display_members = members

            # Plot member profiles with transparency
            alpha = min(0.3, 10.0 / max(1, n_display_members))
            for i in range(display_members.shape[0]):
                member_profile = display_members[i][self._global_gene_indices]
                profile_ax.plot(
                    member_profile, color="gray", alpha=alpha, linewidth=0.5
                )

        # Plot prototype profile on top
        profile_ax.plot(prototype_profile, color="red", linewidth=2, label="Prototype")
        profile_ax.set_xlabel("Gene Rank (by Variance)")
        profile_ax.set_ylabel("Expression")
        profile_ax.set_title(f"Cluster {cluster_id} Profile\n({n_members} cells)")
        profile_ax.grid(True, alpha=0.3)
        profile_ax.legend()

        # === RIGHT PANEL: Cluster-Specific Marker Genes ===

        # Compute marker genes for this cluster
        # Find genes most upregulated in this cluster vs others
        cluster_mean = prototype  # Use prototype as cluster representative

        # Compute z-scores relative to global mean
        z_scores = (cluster_mean - self._gene_means) / (self._gene_stds + 1e-8)

        # Get top marker genes
        top_marker_indices = jnp.argsort(z_scores)[-self.n_marker_genes :]
        marker_expressions = cluster_mean[top_marker_indices]
        marker_names = [self._gene_names[int(i)] for i in top_marker_indices]

        # Create horizontal bar plot
        y_pos = np.arange(len(marker_names))
        marker_ax.barh(y_pos, marker_expressions, color="darkblue", alpha=0.7)
        marker_ax.set_yticks(y_pos)
        marker_ax.set_yticklabels(marker_names, fontsize=8)
        marker_ax.set_xlabel("Expression")
        marker_ax.set_title("Top Markers")
        marker_ax.grid(True, alpha=0.3, axis="x")

        # Adjust layout
        marker_ax.invert_yaxis()  # Highest expression at top


def _download_tasic_data(output_path: Path) -> None:
    """Download real Tasic et al. (2018) single-cell RNA-seq dataset from Allen Institute."""
    from urllib.request import urlretrieve
    from zipfile import ZipFile

    import h5py
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Allen Institute download URL for Tasic 2018 dataset
    ZIP_URL = (
        "https://celltypes.brain-map.org/api/v2/well_known_file_download/694413985"
    )
    temp_dir = output_path.parent / "tasic_temp"
    zip_path = temp_dir / "Tasic2018.zip"

    temp_dir.mkdir(exist_ok=True)

    # Download if not already present
    if not zip_path.exists():
        print("Downloading Tasic 2018 V1 dataset from Allen Institute...")
        urlretrieve(ZIP_URL, str(zip_path))
        print("Download complete.")

    # Extract the data
    print("Extracting dataset...")
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Locate the extracted files - actual structure from Allen Institute
    exp_path = temp_dir / "mouse_VISp_2018-06-14_exon-matrix.csv"
    genes_path = temp_dir / "mouse_VISp_2018-06-14_genes-rows.csv"
    samples_path = temp_dir / "mouse_VISp_2018-06-14_samples-columns.csv"

    if not exp_path.exists() or not genes_path.exists() or not samples_path.exists():
        raise FileNotFoundError(
            f"Expected files not found after extraction. Contents: {list(temp_dir.rglob('*.csv'))}"
        )

    print("Loading expression data...")
    expression_df = pd.read_csv(exp_path, index_col=0)
    print("Loading gene metadata...")
    genes_df = pd.read_csv(genes_path, index_col=0)
    print("Loading sample metadata...")
    samples_df = pd.read_csv(samples_path, index_col=0)

    # Convert to numpy arrays
    expression_data = expression_df.values.astype(
        np.float32
    )  # genes x cells, need to transpose
    expression_data = expression_data.T  # Now cells x genes
    gene_names = list(genes_df.index)  # Gene names from genes file

    # Extract cell type labels from samples metadata
    print("Available metadata columns:", list(samples_df.columns))

    # Priority order for cell type labels (from most specific to least)
    label_columns = [
        "cluster",
        "subclass",
        "class",
        "brain_subregion",
        "cortical_layer_label",
    ]

    cell_labels_str = None
    selected_column = None

    for col in label_columns:
        if col in samples_df.columns:
            unique_count = len(samples_df[col].unique())
            if (
                unique_count > 1 and unique_count < len(samples_df) * 0.8
            ):  # Not too few, not too many
                cell_labels_str = samples_df[col].values
                selected_column = col
                print(f"Using {col} as cell type labels ({unique_count} unique types)")
                break

    if cell_labels_str is None:
        # Fallback: find column with reasonable number of categories
        categorical_cols = samples_df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            unique_count = len(samples_df[col].unique())
            if 2 <= unique_count <= 200:  # Reasonable range for cell types
                cell_labels_str = samples_df[col].values
                selected_column = col
                print(f"Using {col} as cell type labels ({unique_count} unique types)")
                break

        if cell_labels_str is None:
            raise ValueError("No suitable cell type labels found in samples metadata")

    unique_cell_types = sorted(set(cell_labels_str))
    label_to_index = {label: idx for idx, label in enumerate(unique_cell_types)}
    cell_labels = np.array(
        [label_to_index[label] for label in cell_labels_str], dtype=np.int32
    )

    print(
        f"Loaded real Tasic dataset: {expression_data.shape[0]} cells, {expression_data.shape[1]} genes"
    )
    print(f"Found {len(unique_cell_types)} unique cell types")

    # Save to HDF5 format
    with h5py.File(output_path, "w") as f:
        f.create_dataset("expression", data=expression_data)
        f.create_dataset("gene_names", data=[g.encode() for g in gene_names])
        f.create_dataset("cell_labels", data=cell_labels)
        f.create_dataset(
            "cell_type_names", data=[c.encode() for c in unique_cell_types]
        )

    print(f"Saved processed dataset to {output_path}")

    # Cleanup temporary files
    import shutil

    shutil.rmtree(temp_dir)
    print("Cleanup complete.")
