"""20 Newsgroups dataset implementation."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from apps.interface import ClusteringDataset, ClusteringDatasetConfig

### Logging ###

log = logging.getLogger(__name__)


@dataclass
class NewsgroupsConfig(ClusteringDatasetConfig):
    """Configuration for 20 Newsgroups dataset.

    Parameters:
        categories: Specific newsgroup categories to include (None for all 20)
        remove: Text parts to remove (standard: headers, footers, quotes)
        max_features: Maximum number of features for TF-IDF (None for all)
        min_df: Minimum document frequency for TF-IDF (sklearn standard: 2)
        max_df: Maximum document frequency for TF-IDF (sklearn standard: 0.95)
        random_seed: Random seed for reproducibility
        use_count_vectorizer: Use count vectorization instead of TF-IDF
        n_top_words: Number of top words to show in visualization
    """

    _target_: str = "plugins.datasets.newsgroups.NewsgroupsDataset.load"

    # Data selection
    categories: list[str] | None = None  # None for all 20 categories
    remove: list[str] = field(
        default_factory=lambda: ["headers", "footers", "quotes"]
    )  # Standard ML benchmark preprocessing

    # Feature extraction - standard sklearn parameters
    max_features: int | None = None  # None for all features
    min_df: int = 2  # Standard sklearn default
    max_df: float = 0.95  # Standard sklearn default

    # Reproducibility
    random_seed: int = 42

    # Vectorization method
    use_count_vectorizer: bool = False  # False for TF-IDF, True for count

    # Visualization
    n_top_words: int = 10


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="newsgroups", node=NewsgroupsConfig)


@dataclass(frozen=True)
class NewsgroupsDataset(ClusteringDataset):
    """20 Newsgroups text classification dataset."""

    # Configuration
    max_features: int
    min_df: int
    max_df: float
    n_top_words: int

    # Data
    _train_data: Array
    _test_data: Array
    _train_labels: Array
    _test_labels: Array

    # Metadata
    _feature_names: list[str]  # TF-IDF feature names (words)
    _target_names: list[str]  # Category names

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        categories: list[str] | None,
        remove: list[str],
        max_features: int | None,
        min_df: int,
        max_df: float,
        random_seed: int,
        use_count_vectorizer: bool,
        n_top_words: int,
    ) -> "NewsgroupsDataset":
        """Load 20 Newsgroups dataset using standard train/test splits.

        Args:
            cache_dir: Directory for caching downloaded data
            categories: Specific categories to include
            remove: What to remove from text
            max_features: Maximum TF-IDF features (None for all features)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            random_seed: Random seed
            n_top_words: Top words for visualization

        Returns:
            Loaded 20 Newsgroups dataset with official splits
        """
        # Use provided remove list (empty by default for standard benchmarking)

        log.info("Loading 20 Newsgroups dataset...")

        # Check for cached processed data
        cache_file = cache_dir / "newsgroups_raw.npz"

        if cache_file.exists():
            log.info("Loading cached newsgroups data...")
            cached = np.load(cache_file, allow_pickle=True)
            train_texts = cached["train_texts"]
            test_texts = cached["test_texts"]
            train_labels = cached["train_labels"]
            test_labels = cached["test_labels"]
            target_names = cached["target_names"].tolist()
        else:
            log.info("Downloading and caching raw newsgroups data...")
            # Load both train and test sets
            train_newsgroups = fetch_20newsgroups(
                subset="train",
                categories=categories,
                remove=tuple(remove),
                shuffle=True,
                random_state=random_seed,
                download_if_missing=True,
                data_home=str(cache_dir),
            )

            test_newsgroups = fetch_20newsgroups(
                subset="test",
                categories=categories,
                remove=tuple(remove),
                shuffle=False,  # Keep test set deterministic
                download_if_missing=True,
                data_home=str(cache_dir),
            )

            # Cache raw text data
            np.savez(
                cache_file,
                train_texts=np.array(train_newsgroups.data, dtype=object),
                test_texts=np.array(test_newsgroups.data, dtype=object),
                train_labels=train_newsgroups.target,
                test_labels=test_newsgroups.target,
                target_names=np.array(train_newsgroups.target_names),
            )

            train_texts = np.array(train_newsgroups.data)
            test_texts = np.array(test_newsgroups.data)
            train_labels = train_newsgroups.target
            test_labels = test_newsgroups.target
            target_names = train_newsgroups.target_names

        # Apply preprocessing based on parameters
        if use_count_vectorizer:
            log.info("Creating count features...")
            vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words="english",
                lowercase=True,
                strip_accents="ascii",
            )
        else:
            log.info("Creating TF-IDF features...")
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words="english",
                lowercase=True,
                strip_accents="ascii",
            )

        # Fit on training data, transform both sets
        train_features = vectorizer.fit_transform(train_texts)
        test_features = vectorizer.transform(test_texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Convert to dense JAX arrays
        train_dense = train_features.toarray().astype(np.float32)
        test_dense = test_features.toarray().astype(np.float32)
        train_labels = train_labels.astype(np.int32)
        test_labels = test_labels.astype(np.int32)

        feature_type = "count" if use_count_vectorizer else "TF-IDF"
        # Use official train/test split
        train_data = jnp.array(train_dense)
        test_data = jnp.array(test_dense)
        train_labels_final = jnp.array(train_labels)
        test_labels_final = jnp.array(test_labels)

        return cls(
            cache_dir=cache_dir,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            n_top_words=n_top_words,
            _train_data=train_data,
            _test_data=test_data,
            _train_labels=train_labels_final,
            _test_labels=test_labels_final,
            _feature_names=feature_names,
            _target_names=list(target_names),
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
        # Compact square for histogram visualization
        return (3, 3)

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        # Space for detailed cluster overview (words + stats)
        return (8, 12)

    @property
    @override
    def n_classes(self) -> int:
        return len(self._target_names)

    @override
    def paint_observable(self, observable: Array, axes: Axes):
        """Compact visualization of a single TF-IDF prototype - for use in hierarchies/merges."""
        axes.set_axis_off()

        # Get top features for distribution shape
        top_scores = observable[jnp.argsort(observable)[-50:]]  # Top 50 features

        # Create compact histogram showing TF-IDF distribution
        hist, bins = np.histogram(top_scores, bins=8, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]

        # Simple gray bars for clean look
        bars = axes.bar(
            bin_centers,
            hist,
            width=width * 0.8,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )

        axes.set_xticks([])
        axes.set_yticks([])

        # Minimal border for structure
        for spine in axes.spines.values():
            spine.set_linewidth(0.3)
            spine.set_color("gray")

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Detailed cluster overview showing topic words and statistics."""

        # Turn off the main axes frame
        axes.set_axis_off()

        # Get subplot specification and figure
        subplot_spec = axes.get_subplotspec()
        fig = axes.get_figure()

        if subplot_spec is None:
            raise ValueError("paint_cluster requires a subplot")

        assert isinstance(fig, Figure)

        # Create a vertical layout: topic words on top, statistics below
        gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=subplot_spec, height_ratios=[3, 1], hspace=0.4
        )

        # Create axes for topic words and statistics
        words_ax = fig.add_subplot(gs[0, 0])
        stats_ax = fig.add_subplot(gs[1, 0])

        # === TOP PANEL: Top Topic Words ===

        # Get top words for this cluster
        n_words = min(15, self.n_top_words)
        top_indices = jnp.argsort(prototype)[-n_words:]
        top_scores = prototype[top_indices]
        top_words = [self._feature_names[int(i)] for i in top_indices]

        # Create horizontal bar plot
        y_pos = np.arange(len(top_words))
        bars = words_ax.barh(
            y_pos, top_scores, color="darkgreen", alpha=0.8, height=0.7
        )
        words_ax.set_yticks(y_pos)
        words_ax.set_yticklabels(top_words, fontsize=11, fontweight="bold")
        words_ax.set_xlabel("TF-IDF Score", fontsize=11)
        words_ax.set_title(
            f"Cluster {cluster_id} Topic Words ({members.shape[0]} docs)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        words_ax.invert_yaxis()
        words_ax.grid(True, alpha=0.3, axis="x")

        # === BOTTOM PANEL: Cluster Statistics ===

        # Compute cluster statistics
        n_members = members.shape[0]
        sparsity = jnp.mean(members == 0)
        doc_lengths = jnp.sum(members > 0, axis=1)
        mean_length = jnp.mean(doc_lengths)
        std_length = jnp.std(doc_lengths)

        # Comprehensive statistics
        stats_text = f"""Cluster Statistics:
• Documents: {n_members:,}
• Avg active words/doc: {mean_length:.1f} ± {std_length:.1f}
• Sparsity: {sparsity:.1%}
• Top TF-IDF score: {top_scores[-1]:.4f}
• Prototype L2 norm: {jnp.linalg.norm(prototype):.3f}"""

        stats_ax.text(
            0.02,
            0.98,
            stats_text,
            transform=stats_ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
        )
        stats_ax.set_xlim(0, 1)
        stats_ax.set_ylim(0, 1)
        stats_ax.axis("off")

    def create_cluster_overview_heatmap(
        self, prototypes: Array, member_counts: Array, n_top_words: int = 20
    ) -> Figure:
        """Create a scalable heatmap overview of all clusters.

        Args:
            prototypes: Array of cluster prototypes (n_clusters, n_features)
            member_counts: Number of members in each cluster
            n_top_words: Number of top words to show

        Returns:
            Figure with heatmap overview
        """
        n_clusters = prototypes.shape[0]

        # Get global top words across all clusters
        cluster_maxes = jnp.max(prototypes, axis=0)
        global_top_indices = jnp.argsort(cluster_maxes)[-n_top_words:]
        global_top_words = [self._feature_names[int(i)] for i in global_top_indices]

        # Extract scores for these words across all clusters
        heatmap_data = prototypes[:, global_top_indices]

        # Create figure
        fig, (ax_main, ax_sizes) = plt.subplots(
            1,
            2,
            figsize=(16, max(8, n_clusters * 0.3)),
            gridspec_kw={"width_ratios": [4, 1]},
        )

        # Main heatmap
        im = ax_main.imshow(
            heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest"
        )

        # Set labels
        ax_main.set_xticks(range(len(global_top_words)))
        ax_main.set_xticklabels(global_top_words, rotation=45, ha="right", fontsize=10)
        ax_main.set_yticks(range(n_clusters))
        ax_main.set_yticklabels([f"C{i}" for i in range(n_clusters)], fontsize=8)
        ax_main.set_xlabel("Top Topic Words", fontsize=12)
        ax_main.set_ylabel("Clusters", fontsize=12)
        ax_main.set_title(
            f"Cluster-Word Heatmap ({n_clusters} clusters)", fontsize=14, pad=20
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
        cbar.set_label("TF-IDF Score", fontsize=10)

        # Cluster sizes bar chart
        y_pos = np.arange(n_clusters)
        ax_sizes.barh(y_pos, member_counts, alpha=0.7, color="steelblue")
        ax_sizes.set_yticks(y_pos)
        ax_sizes.set_yticklabels([f"C{i}" for i in range(n_clusters)], fontsize=8)
        ax_sizes.set_xlabel("Members", fontsize=10)
        ax_sizes.set_title("Cluster Sizes", fontsize=12)
        ax_sizes.grid(True, alpha=0.3, axis="x")

        # Add member count labels
        for i, count in enumerate(member_counts):
            ax_sizes.text(
                count + max(member_counts) * 0.01,
                i,
                str(int(count)),
                va="center",
                fontsize=8,
                alpha=0.8,
            )

        plt.tight_layout()
        return fig

    def create_cluster_wordcloud_grid(
        self,
        prototypes: Array,
        member_counts: Array,
        max_clusters: int = 50,
        words_per_cluster: int = 10,
    ) -> Figure:
        """Create a grid of small word lists for cluster overview.

        Args:
            prototypes: Array of cluster prototypes
            member_counts: Number of members in each cluster
            max_clusters: Maximum clusters to show
            words_per_cluster: Words per cluster to display

        Returns:
            Figure with word grid overview
        """
        n_clusters = min(prototypes.shape[0], max_clusters)

        # Calculate grid dimensions
        cols = min(8, n_clusters)
        rows = (n_clusters + cols - 1) // cols

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 2.5, rows * 2),
            subplot_kw={"xticks": [], "yticks": []},
        )
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            f"Cluster Topic Words Overview ({n_clusters} clusters)", fontsize=16, y=0.98
        )

        for i in range(n_clusters):
            row, col = i // cols, i % cols
            ax = axes[row, col]

            # Get top words for this cluster
            top_indices = jnp.argsort(prototypes[i])[-words_per_cluster:]
            top_words = [self._feature_names[int(idx)] for idx in top_indices]
            top_scores = prototypes[i, top_indices]

            # Create text display
            text_lines = []
            for word, score in zip(top_words, top_scores):
                text_lines.append(f"{word}")

            # Display words
            text_content = "\n".join(
                reversed(text_lines)
            )  # Reverse for better visual order
            ax.text(
                0.05,
                0.95,
                text_content,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
            )

            # Add cluster info
            ax.text(
                0.95,
                0.05,
                f"C{i}\n{int(member_counts[i])} docs",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5),
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        # Hide unused subplots
        for i in range(n_clusters, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis("off")

        plt.tight_layout()
        return fig

    def create_cluster_similarity_matrix(self, prototypes: Array) -> Figure:
        """Create cluster similarity matrix for understanding relationships.

        Args:
            prototypes: Array of cluster prototypes

        Returns:
            Figure with similarity matrix
        """
        n_clusters = prototypes.shape[0]

        # Compute cosine similarity between cluster prototypes
        norms = jnp.linalg.norm(prototypes, axis=1, keepdims=True)
        normalized_prototypes = prototypes / (norms + 1e-8)
        similarity_matrix = jnp.dot(normalized_prototypes, normalized_prototypes.T)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot heatmap
        im = ax.imshow(similarity_matrix, cmap="RdYlBu_r", vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        ax.set_xticklabels([f"C{i}" for i in range(n_clusters)], fontsize=8)
        ax.set_yticklabels([f"C{i}" for i in range(n_clusters)], fontsize=8)
        ax.set_xlabel("Clusters", fontsize=12)
        ax.set_ylabel("Clusters", fontsize=12)
        ax.set_title("Cluster Similarity Matrix (Cosine)", fontsize=14, pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine Similarity", fontsize=10)

        plt.tight_layout()
        return fig

    def create_cluster_overview_dashboard(
        self, prototypes: Array, member_counts: Array
    ) -> Figure:
        """Create a comprehensive dashboard for hundreds of clusters - no text reading required."""

        n_clusters = prototypes.shape[0]

        # Create figure with multiple visualization panels
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], width_ratios=[3, 2, 2, 2])

        # === PANEL 1: Cluster Heatmap (largest) ===
        ax_heatmap = fig.add_subplot(gs[0, :2])

        # Get global top words
        cluster_maxes = jnp.max(prototypes, axis=0)
        global_top_indices = jnp.argsort(cluster_maxes)[-30:]
        global_top_words = [self._feature_names[int(i)] for i in global_top_indices]
        heatmap_data = prototypes[:, global_top_indices]

        im = ax_heatmap.imshow(
            heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest"
        )
        ax_heatmap.set_xticks(range(len(global_top_words)))
        ax_heatmap.set_xticklabels(
            global_top_words, rotation=45, ha="right", fontsize=9
        )
        ax_heatmap.set_ylabel("Clusters", fontsize=12)
        ax_heatmap.set_title(
            f"Cluster-Word Heatmap ({n_clusters} clusters)", fontsize=14
        )

        # === PANEL 2: Cluster Size Distribution ===
        ax_sizes = fig.add_subplot(gs[0, 2])

        # Histogram of cluster sizes
        ax_sizes.hist(
            member_counts, bins=20, alpha=0.7, color="steelblue", edgecolor="black"
        )
        ax_sizes.set_xlabel("Documents per Cluster")
        ax_sizes.set_ylabel("Number of Clusters")
        ax_sizes.set_title("Cluster Size Distribution")
        ax_sizes.grid(True, alpha=0.3)

        # === PANEL 3: Topic Diversity Scatter ===
        ax_diversity = fig.add_subplot(gs[0, 3])

        # Compute cluster diversity metrics
        cluster_entropies = []
        cluster_max_scores = []

        for i in range(n_clusters):
            # Entropy as diversity measure
            probs = prototypes[i] / jnp.sum(prototypes[i])
            probs = probs[probs > 0]  # Remove zeros
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))
            cluster_entropies.append(float(entropy))
            cluster_max_scores.append(float(jnp.max(prototypes[i])))

        # Scatter plot: entropy vs max score, size = cluster size
        scatter = ax_diversity.scatter(
            cluster_entropies,
            cluster_max_scores,
            s=member_counts / 10,
            alpha=0.6,
            c=range(n_clusters),
            cmap="tab20",
        )
        ax_diversity.set_xlabel("Topic Diversity (Entropy)")
        ax_diversity.set_ylabel("Max TF-IDF Score")
        ax_diversity.set_title("Cluster Characteristics")
        ax_diversity.grid(True, alpha=0.3)

        # === PANEL 4: Cluster Similarity Matrix (subset) ===
        ax_similarity = fig.add_subplot(gs[1, :2])

        # Show similarity for up to 50 clusters
        n_show = min(50, n_clusters)
        subset_prototypes = prototypes[:n_show]

        # Compute cosine similarity
        norms = jnp.linalg.norm(subset_prototypes, axis=1, keepdims=True)
        normalized = subset_prototypes / (norms + 1e-8)
        similarity = jnp.dot(normalized, normalized.T)

        im_sim = ax_similarity.imshow(similarity, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax_similarity.set_title(f"Cluster Similarity (first {n_show} clusters)")
        ax_similarity.set_xlabel("Clusters")
        ax_similarity.set_ylabel("Clusters")

        # === PANEL 5: Top Words Cloud (visual) ===
        ax_wordcloud = fig.add_subplot(gs[1, 2:])

        # Get overall word importance across all clusters
        word_importance = jnp.sum(prototypes, axis=0)
        top_word_indices = jnp.argsort(word_importance)[-50:]
        top_words = [self._feature_names[int(i)] for i in top_word_indices]
        top_scores = word_importance[top_word_indices]

        # Create word cloud effect with font sizes
        y_positions = np.random.uniform(0, 1, len(top_words))
        x_positions = np.random.uniform(0, 1, len(top_words))

        for i, (word, score, x, y) in enumerate(
            zip(top_words, top_scores, x_positions, y_positions)
        ):
            fontsize = 8 + (score / jnp.max(top_scores)) * 12
            ax_wordcloud.text(
                x,
                y,
                word,
                fontsize=fontsize,
                ha="center",
                va="center",
                alpha=0.7,
                rotation=np.random.uniform(-30, 30),
            )

        ax_wordcloud.set_xlim(0, 1)
        ax_wordcloud.set_ylim(0, 1)
        ax_wordcloud.set_title("Most Important Words")
        ax_wordcloud.axis("off")

        # === PANEL 6: Summary Statistics ===
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis("off")

        # Compute summary statistics
        total_docs = jnp.sum(member_counts)
        avg_cluster_size = jnp.mean(member_counts)
        std_cluster_size = jnp.std(member_counts)
        min_cluster_size = jnp.min(member_counts)
        max_cluster_size = jnp.max(member_counts)

        avg_sparsity = jnp.mean(
            [jnp.mean(prototypes[i] == 0) for i in range(n_clusters)]
        )

        summary_text = f"""
CLUSTERING SUMMARY ({n_clusters} clusters, {total_docs:,} total documents)

Cluster Sizes: μ={avg_cluster_size:.1f} σ={std_cluster_size:.1f} | Range: {min_cluster_size}-{max_cluster_size}
Avg Sparsity: {avg_sparsity:.1%} | Vocabulary: {prototypes.shape[1]:,} features
        """

        ax_stats.text(
            0.02,
            0.5,
            summary_text,
            transform=ax_stats.transAxes,
            fontsize=12,
            fontfamily="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
        )

        plt.tight_layout()
        return fig
