"""Tasic et al. (2018) single-cell RNA-seq dataset implementation."""

from dataclasses import dataclass
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

from apps.interface import ClusteringDataset, ClusteringDatasetConfig


def _download_tasic_data(output_path: Path) -> None:
    """Download Tasic et al. (2018) dataset from Allen Brain Atlas."""
    try:
        import json

        import requests
    except ImportError:
        raise ImportError(
            "requests is required to download Tasic dataset. Install with: pip install requests"
        )

    # Ensure cache directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Allen Brain Atlas API endpoint for Tasic 2018 dataset
    # This is a simplified download - the actual API structure may be more complex
    print("Fetching Tasic dataset from Allen Brain Atlas...")

    try:
        # Download the actual Tasic et al. 2018 dataset from Allen Brain Atlas
        import io
        import zipfile

        import h5py

        print("Downloading Tasic et al. (2018) dataset from Allen Brain Atlas...")

        # Official Allen Brain Atlas URLs for Tasic 2018 dataset
        visp_url = "https://celltypes.brain-map.org/api/v2/well_known_file_download/694413985"  # VISp
        alm_url = "https://celltypes.brain-map.org/api/v2/well_known_file_download/694413179"  # ALM

        all_expression_data = []
        all_gene_names = None
        all_cell_labels = []
        all_cell_ids = []
        region_labels = []

        # Download both brain regions
        for region_name, url in [("VISp", visp_url), ("ALM", alm_url)]:
            print(f"Downloading {region_name} data...")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # The file is a ZIP archive
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find CSV files in the archive
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]

                if not csv_files:
                    raise RuntimeError(f"No CSV files found in {region_name} archive")

                # Load the main expression matrix (usually the largest CSV)
                main_csv = max(csv_files, key=lambda f: z.getinfo(f).file_size)
                print(f"Loading {main_csv} from {region_name}...")

                with z.open(main_csv) as csv_file:
                    # Load expression data
                    region_data = pd.read_csv(csv_file, index_col=0)

                    # Genes are typically rows, cells are columns
                    if all_gene_names is None:
                        all_gene_names = region_data.index.tolist()
                    else:
                        # Ensure gene names match across regions
                        if set(region_data.index) != set(all_gene_names):
                            # Take intersection of genes
                            common_genes = list(
                                set(region_data.index) & set(all_gene_names)
                            )
                            region_data = region_data.loc[common_genes]
                            if len(all_expression_data) > 0:
                                # Filter previous data to same genes
                                for i, prev_data in enumerate(all_expression_data):
                                    all_expression_data[i] = prev_data[
                                        np.isin(all_gene_names, common_genes)
                                    ]
                            all_gene_names = common_genes

                    # Transpose to get cells × genes format
                    region_expression = region_data.T.values.astype(np.float32)
                    all_expression_data.append(region_expression)

                    # Store cell IDs
                    cell_ids = region_data.columns.tolist()
                    all_cell_ids.extend(cell_ids)

                    # Add region labels
                    region_labels.extend([region_name] * len(cell_ids))

                    print(
                        f"{region_name}: {region_expression.shape[0]} cells × {region_expression.shape[1]} genes"
                    )

        # Combine data from both regions
        expression_data = np.vstack(all_expression_data)

        print(
            f"Combined dataset: {expression_data.shape[0]} cells × {expression_data.shape[1]} genes"
        )

        # Download metadata to get cell type labels
        print("Downloading cell metadata...")

        # The metadata is available through the cell types browser
        # We'll create basic cell type labels based on the region for now
        # In the real implementation, you'd want to get the actual cluster assignments

        # For now, assign simple region-based labels
        # TODO: Download actual cell type metadata from the Allen Brain Atlas
        cell_type_names = list(set(region_labels))
        type_to_int = {t: i for i, t in enumerate(cell_type_names)}
        cell_labels = np.array([type_to_int[region] for region in region_labels])

        print(f"Found {len(cell_type_names)} regions: {cell_type_names}")

        # Save to HDF5
        with h5py.File(output_path, "w") as f:
            f.create_dataset("expression", data=expression_data)
            # Handle mixed string/integer gene names
            f.create_dataset("gene_names", data=[str(g).encode() for g in all_gene_names])
            f.create_dataset("cell_labels", data=cell_labels)
            f.create_dataset(
                "cell_type_names", data=[str(t).encode() for t in cell_type_names]
            )
            f.create_dataset("cell_ids", data=[str(c).encode() for c in all_cell_ids])
            f.create_dataset("region_labels", data=[str(r).encode() for r in region_labels])

        print(f"Real Tasic dataset saved to {output_path}")
        print(
            f"Dataset shape: {expression_data.shape[0]} cells × {expression_data.shape[1]} genes"
        )

    except Exception as e:
        raise RuntimeError(f"Failed to download/create Tasic dataset: {e}") from e


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

        # Download data if it doesn't exist
        if not data_file.exists():
            print(f"Downloading Tasic dataset to {data_file}...")
            try:
                _download_tasic_data(data_file)
            except Exception as e:
                raise RuntimeError(
                    f"""Failed to download Tasic dataset automatically.
                    
You can manually download the Tasic et al. (2018) dataset and save it as:
{data_file}

The file should contain:
- 'expression': Expression matrix (cells x genes)
- 'gene_names': Gene names
- 'cell_labels': Cell type labels
- 'cell_type_names': Cell type names

You can download the data from:
https://portal.brain-map.org/atlases-and-data/rnaseq

Or use the Allen Brain Atlas API to fetch the data programmatically.

Original error: {e}
"""
                ) from e

        # Load data from HDF5 file
        try:
            import h5py

            with h5py.File(data_file, "r") as f:
                expression_data = f["expression"][:]  # cells x genes
                gene_names = [
                    g.decode() if isinstance(g, bytes) else g
                    for g in f["gene_names"][:]
                ]
                cell_labels = f["cell_labels"][:]
                cell_type_names = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in f["cell_type_names"][:]
                ]
        except ImportError:
            raise ImportError(
                "h5py is required to load Tasic dataset. Install with: pip install h5py"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Tasic dataset from {data_file}: {e}")

        # Convert to numpy arrays
        expression_data = np.array(expression_data, dtype=np.float32)
        cell_labels = np.array(cell_labels, dtype=np.int32)

        print(
            f"Loaded Tasic dataset: {expression_data.shape[0]} cells, {expression_data.shape[1]} genes"
        )

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
