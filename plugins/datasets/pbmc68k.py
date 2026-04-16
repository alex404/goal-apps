"""PBMC 68k single-cell RNA-seq dataset (Zheng et al. 2017).

Fresh 68k PBMCs from 10x Genomics (Donor A). Cell type labels are assigned
via canonical marker genes — approximate but sufficient for NMI evaluation.

Requires manual download: place the tarball
``fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz``
in the cache directory (default: ``.cache/``).  Download from:
https://www.10xgenomics.com/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0
"""

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import h5py
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from scipy.io import mmread
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from apps.interface import ClusteringDataset, ClusteringDatasetConfig

### Logging ###

log = logging.getLogger(__name__)

### Constants ###

_TARBALL_NAME = "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
_DOWNLOAD_URL = (
    "https://www.10xgenomics.com/datasets/"
    "fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0"
)

# Canonical PBMC marker genes for cell type annotation
_PBMC_MARKERS: dict[str, list[str]] = {
    "CD14+ Monocyte": ["CD14", "LYZ", "S100A8", "S100A9"],
    "FCGR3A+ Monocyte": ["FCGR3A", "MS4A7"],
    "B cell": ["CD79A", "MS4A1", "CD79B"],
    "CD4+ T cell": ["CD3D", "IL7R", "CD4"],
    "CD8+ T cell": ["CD3D", "CD8A", "CD8B"],
    "NK cell": ["NKG7", "GNLY", "NCAM1"],
    "Dendritic": ["FCER1A", "CST3"],
    "Megakaryocyte": ["PPBP", "PF4"],
}

### Config ###


@dataclass
class PBMC68kConfig(ClusteringDatasetConfig):
    """Configuration for PBMC 68k dataset.

    Parameters:
        min_genes_per_cell: Minimum genes detected per cell (QC filter)
        max_genes_per_cell: Maximum genes detected per cell (doublet filter)
        min_cells_per_gene: Minimum cells expressing a gene
        n_genes: Number of highly variable genes to retain
        use_fano_selection: Use Fano factor for HVG selection
        log_transform: Apply log1p transformation after CP10k normalization
        standardize: Center and scale features
        test_fraction: Fraction of data for test set
        random_seed: Random seed for reproducibility
        min_marker_score: Minimum mean marker expression for cell type annotation
        max_cells: Optional cap on number of cells (None for all ~68k)
        n_global_genes: Number of genes in global profile visualization
        n_marker_genes: Number of cluster-specific marker genes to show
    """

    _target_: str = "plugins.datasets.pbmc68k.PBMC68kDataset.load"

    # QC filtering
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 5000
    min_cells_per_gene: int = 10

    # Gene selection
    n_genes: int = 2000
    use_fano_selection: bool = True

    # Preprocessing
    log_transform: bool = True
    standardize: bool = False

    # Data splits
    test_fraction: float = 0.2
    random_seed: int = 42

    # Cell type annotation
    min_marker_score: float = 0.5

    # Memory management
    max_cells: int | None = None

    # Visualization
    n_global_genes: int = 100
    n_marker_genes: int = 8


cs = ConfigStore.instance()
cs.store(group="dataset", name="pbmc68k", node=PBMC68kConfig)

### Dataset ###


@dataclass(frozen=True)
class PBMC68kDataset(ClusteringDataset):
    """PBMC 68k single-cell RNA-seq dataset (Zheng et al. 2017)."""

    # Configuration
    n_genes: int
    n_global_genes: int
    n_marker_genes: int

    # Data
    _train_data: Array
    _test_data: Array
    _train_labels: Array
    _test_labels: Array

    # Metadata for visualization
    _gene_names: list[str]
    _cell_type_names: list[str]
    _global_gene_indices: Array
    _gene_means: Array
    _gene_stds: Array

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        min_genes_per_cell: int = 200,
        max_genes_per_cell: int = 5000,
        min_cells_per_gene: int = 10,
        n_genes: int = 2000,
        use_fano_selection: bool = True,
        log_transform: bool = True,
        standardize: bool = True,
        test_fraction: float = 0.2,
        random_seed: int = 42,
        min_marker_score: float = 0.5,
        max_cells: int | None = None,
        n_global_genes: int = 100,
        n_marker_genes: int = 8,
    ) -> "PBMC68kDataset":
        """Load PBMC 68k dataset with standard scRNA-seq preprocessing."""
        # Build cache key from preprocessing parameters
        cache_key = (
            f"pbmc68k_g{n_genes}"
            f"_qc{min_genes_per_cell}-{max_genes_per_cell}-{min_cells_per_gene}"
        )
        if max_cells is not None:
            cache_key += f"_max{max_cells}"
        processed_path = cache_dir / f"{cache_key}.h5"

        if processed_path.exists():
            log.info("Loading cached PBMC 68k processed data: %s", processed_path)
            with h5py.File(processed_path, "r") as hf:
                f: Any = hf  # h5py stubs don't support dataset subscript
                expression_data = np.array(f["expression"][:], dtype=np.float32)
                gene_names = [
                    g.decode() if isinstance(g, bytes) else g
                    for g in f["gene_names"][:]
                ]
                cell_labels = np.array(f["cell_labels"][:], dtype=np.int32)
                cell_type_names = [
                    c.decode() if isinstance(c, bytes) else c
                    for c in f["cell_type_names"][:]
                ]
        else:
            expression_data, gene_names, cell_labels, cell_type_names = (
                _preprocess_pbmc68k(
                    cache_dir=cache_dir,
                    min_genes_per_cell=min_genes_per_cell,
                    max_genes_per_cell=max_genes_per_cell,
                    min_cells_per_gene=min_cells_per_gene,
                    n_genes=n_genes,
                    use_fano_selection=use_fano_selection,
                    log_transform=log_transform,
                    max_cells=max_cells,
                    random_seed=random_seed,
                    min_marker_score=min_marker_score,
                )
            )

            # Cache processed data
            cache_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(processed_path, "w") as f:
                f.create_dataset("expression", data=expression_data)
                f.create_dataset(
                    "gene_names", data=[g.encode() for g in gene_names]
                )
                f.create_dataset("cell_labels", data=cell_labels)
                f.create_dataset(
                    "cell_type_names",
                    data=[c.encode() for c in cell_type_names],
                )
            log.info("Saved processed cache: %s", processed_path)

        log.info(
            "PBMC 68k: %d cells, %d genes, %d cell types",
            expression_data.shape[0],
            expression_data.shape[1],
            len(cell_type_names),
        )

        # Keep pre-standardization copy for visualization stats
        visu_matrix = expression_data.copy()

        if standardize:
            scaler = StandardScaler()
            expression_data = scaler.fit_transform(expression_data).astype(np.float32)
            log.info("Applied standardization")

        # Convert to JAX arrays
        expression_jax = jnp.array(expression_data)
        cell_labels_jax = jnp.array(cell_labels)

        # Stratified train/test split
        n_cells = expression_jax.shape[0]
        try:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=test_fraction, random_state=random_seed
            )
            ((train_idx, test_idx),) = sss.split(
                np.zeros(n_cells), cell_labels
            )
            log.info(
                "Stratified split: %d train, %d test", len(train_idx), len(test_idx)
            )
        except ValueError as e:
            log.warning("Stratified split failed (%s), using random split", e)
            rng = np.random.RandomState(random_seed)
            test_idx = rng.choice(
                n_cells, size=int(n_cells * test_fraction), replace=False
            )
            train_idx = np.setdiff1d(np.arange(n_cells), test_idx)

        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)

        train_data = expression_jax[train_idx]
        test_data = expression_jax[test_idx]
        train_labels = cell_labels_jax[train_idx]
        test_labels = cell_labels_jax[test_idx]

        # Visualization metadata from pre-standardized matrix
        train_visu = visu_matrix[train_idx]
        gene_means = jnp.array(train_visu.mean(axis=0))
        gene_stds = jnp.array(train_visu.std(axis=0))
        global_gene_indices = jnp.argsort(gene_stds)[-n_global_genes:]

        return cls(
            cache_dir=cache_dir,
            n_genes=n_genes,
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

    # --- Properties ---

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
        return (self.n_marker_genes, self.n_marker_genes)

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        return (1, 1)

    @property
    @override
    def n_classes(self) -> int:
        return len(self._cell_type_names)

    # --- Visualization ---

    def _paint_top_genes(self, observable: Array, axes: Axes, n_genes: int) -> None:
        """Bar chart of top genes by z-score. The useful signal for scRNA-seq."""
        z_scores = (observable - self._gene_means) / (self._gene_stds + 1e-8)
        top_indices = jnp.argsort(z_scores)[-n_genes:]
        top_z = z_scores[top_indices]
        gene_names = [self._gene_names[int(i)] for i in top_indices]

        y_pos = np.arange(n_genes)
        colors = ["#c44e52" if z > 0 else "#4c72b0" for z in np.array(top_z)]
        axes.barh(y_pos, top_z, color=colors, alpha=0.8)
        axes.set_yticks(y_pos)
        axes.set_yticklabels(gene_names, fontsize=7)
        axes.set_xlabel("z-score")
        axes.axvline(0, color="black", linewidth=0.5)
        axes.invert_yaxis()

    @override
    def paint_observable(self, observable: Array, axes: Axes):
        """Top marker genes by z-score — compact, informative for scRNA-seq."""
        self._paint_top_genes(observable, axes, self.n_marker_genes)

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Cluster prototype: top marker genes + size annotation."""
        n_members = members.shape[0]
        self._paint_top_genes(prototype, axes, self.n_marker_genes + 4)
        axes.set_title(f"C{cluster_id} ({n_members})", fontsize=8)


### Preprocessing helpers ###


def _extract_mex(cache_dir: Path) -> Path:
    """Extract the 10x MEX tarball if not already extracted.

    Looks for the tarball in cache_dir.  If the extracted directory already
    exists, skips extraction.  Raises FileNotFoundError with download
    instructions if the tarball is missing.

    Returns:
        Path to the extracted MEX directory (contains barcodes.tsv, genes.tsv, matrix.mtx).
    """
    mex_dir = cache_dir / "pbmc68k_mex"
    if mex_dir.exists():
        log.info("Using cached PBMC 68k MEX data: %s", mex_dir)
        return mex_dir

    tarball = cache_dir / _TARBALL_NAME
    if not tarball.exists():
        raise FileNotFoundError(
            f"PBMC 68k tarball not found at:\n  {tarball}\n\n"
            + f"Download '{_TARBALL_NAME}' from:\n  {_DOWNLOAD_URL}\n"
            + f"and place it in:\n  {cache_dir}/"
        )

    log.info("Extracting PBMC 68k tarball...")
    mex_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(mex_dir, filter="data")
    log.info("Extraction complete: %s", mex_dir)
    return mex_dir


def _read_10x_mex(mex_dir: Path) -> tuple[sp.csc_matrix, list[str]]:
    """Read 10x MEX format (matrix.mtx + genes.tsv + barcodes.tsv).

    Returns:
        matrix: Sparse CSC matrix (genes x cells)
        gene_names: Gene symbol names
    """
    # The tarball extracts to filtered_matrices_mex/hg19/
    inner = mex_dir / "filtered_matrices_mex" / "hg19"
    if not inner.exists():
        # Fallback: try mex_dir directly
        inner = mex_dir

    matrix = sp.csc_matrix(mmread(inner / "matrix.mtx"))
    gene_names: list[str] = []
    with open(inner / "genes.tsv") as f:
        for line in f:
            parts = line.strip().split("\t")
            # genes.tsv: col0 = Ensembl ID, col1 = symbol
            gene_names.append(parts[1] if len(parts) > 1 else parts[0])

    log.info("Read MEX: %d genes x %d cells", matrix.shape[0], matrix.shape[1])
    return matrix, gene_names


def _annotate_cell_types(
    expression: sp.spmatrix,
    gene_names: list[str],
    min_marker_score: float,
) -> tuple[np.ndarray, list[str]]:
    """Annotate cells using canonical PBMC marker genes.

    Runs on the full gene set AFTER log-normalization but BEFORE HVG selection,
    since marker genes may not be among the top HVGs.

    Returns:
        labels: Integer labels (n_cells,)
        cell_type_names: Ordered cell type names (may include "Unknown")
    """
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}
    type_names = list(_PBMC_MARKERS.keys())
    n_cells = expression.shape[0]

    scores = np.zeros((n_cells, len(type_names)), dtype=np.float32)

    for j, cell_type in enumerate(type_names):
        markers = _PBMC_MARKERS[cell_type]
        found_idx = [gene_to_idx[m] for m in markers if m in gene_to_idx]

        if not found_idx:
            log.warning("No marker genes found for %s", cell_type)
            continue

        marker_expr = np.asarray(expression[:, found_idx].todense())
        scores[:, j] = marker_expr.mean(axis=1).ravel()

    best_type = np.argmax(scores, axis=1)
    best_score = scores[np.arange(n_cells), best_type]

    if np.any(best_score < min_marker_score):
        type_names_final = [*type_names, "Unknown"]
        unknown_idx = len(type_names)
        labels = np.where(best_score >= min_marker_score, best_type, unknown_idx)
    else:
        type_names_final = type_names
        labels = best_type

    for i, name in enumerate(type_names_final):
        count = np.sum(labels == i)
        log.info("  %s: %d cells", name, count)

    return labels.astype(np.int32), type_names_final


def _preprocess_pbmc68k(
    cache_dir: Path,
    min_genes_per_cell: int,
    max_genes_per_cell: int,
    min_cells_per_gene: int,
    n_genes: int,
    use_fano_selection: bool,
    log_transform: bool,
    max_cells: int | None,
    random_seed: int,
    min_marker_score: float,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Download and preprocess PBMC 68k from scratch.

    Returns:
        expression: Dense float32 array (n_cells, n_genes) — log-normalized, NOT standardized
        gene_names: HVG names
        cell_labels: Integer labels (n_cells,)
        cell_type_names: Ordered cell type names
    """
    # Extract tarball if needed (user must provide the download)
    mex_dir = _extract_mex(cache_dir)

    # Read 10x MEX format (genes x cells CSC)
    matrix_csc, gene_names = _read_10x_mex(mex_dir)
    log.info("Raw matrix: %d genes x %d cells", *matrix_csc.shape)

    # Transpose to cells x genes (CSR for efficient row operations)
    matrix = matrix_csc.T.tocsr().astype(np.float32)
    log.info("Transposed: %d cells x %d genes", *matrix.shape)

    # Optional subsampling
    if max_cells is not None and matrix.shape[0] > max_cells:
        rng = np.random.RandomState(random_seed)
        idx = np.sort(rng.choice(matrix.shape[0], max_cells, replace=False))
        matrix = matrix[idx]
        log.info("Subsampled to %d cells", max_cells)

    # QC: filter cells by gene count
    genes_per_cell = np.asarray((matrix > 0).sum(axis=1)).ravel()
    cell_mask = (genes_per_cell >= min_genes_per_cell) & (
        genes_per_cell <= max_genes_per_cell
    )
    matrix = matrix[cell_mask]
    log.info("After cell QC: %d cells", matrix.shape[0])

    # QC: filter genes by cell count
    cells_per_gene = np.asarray((matrix > 0).sum(axis=0)).ravel()
    gene_mask = cells_per_gene >= min_cells_per_gene
    matrix = matrix[:, gene_mask]
    gene_names = [gene_names[i] for i in range(len(gene_names)) if gene_mask[i]]
    log.info("After gene QC: %d genes", matrix.shape[1])

    # CP10k normalization (sparse-friendly row scaling)
    libsize = np.asarray(matrix.sum(axis=1)).ravel()
    libsize[libsize == 0] = 1.0
    scaling = sp.diags(1e4 / libsize)
    matrix = sp.csr_matrix(scaling @ matrix)
    log.info("Applied CP10k normalization")

    # log1p (operates on stored nonzeros only; log1p(0) = 0 preserves sparsity)
    if log_transform:
        matrix.data = np.log1p(matrix.data)
        log.info("Applied log1p transformation")

    # Cell type annotation (on full gene set, before HVG selection)
    cell_labels, cell_type_names = _annotate_cell_types(
        matrix, gene_names, min_marker_score
    )

    # Fano factor HVG selection
    if use_fano_selection and n_genes < matrix.shape[1]:
        gene_means = np.asarray(matrix.mean(axis=0)).ravel()
        gene_sq_means = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        gene_vars = gene_sq_means - gene_means**2
        fano = gene_vars / (gene_means + 1e-8)

        top_idx = np.argsort(fano)[-n_genes:]
        top_idx.sort()
        matrix = matrix[:, top_idx]
        gene_names = [gene_names[i] for i in top_idx]
        log.info("After Fano factor selection: %d genes", len(gene_names))

    # Convert to dense
    expression = matrix.toarray().astype(np.float32)
    log.info(
        "Dense matrix: %s (%.0f MB)", expression.shape, expression.nbytes / 1e6
    )

    return expression, gene_names, cell_labels, cell_type_names
