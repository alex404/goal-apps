"""
CIFAR10 dataset plugin for HMoG clustering.
Save this as: plugins/datasets/cifar10.py
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import override
from urllib.error import URLError

import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torchvision import datasets, transforms

from apps.interface import ClusteringDataset, ClusteringDatasetConfig

# CIFAR10 is 32x32 RGB images
N_ROWS = 32
N_COLS = 32
N_CHANNELS = 3


@dataclass
class CIFAR10Config(ClusteringDatasetConfig):
    """Configuration for CIFAR10 dataset.

    Parameters:
        greyscale: Whether to convert RGB images to grayscale
        normalize: Whether to normalize to zero mean, unit variance
    """

    _target_: str = "plugins.datasets.cifar10.CIFAR10Dataset.load"
    greyscale: bool = True  # Convert to grayscale by default for HMoG
    normalize: bool = False  # Normalize for numerical stability


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="cifar10", node=CIFAR10Config)


@dataclass(frozen=True)
class CIFAR10Dataset(ClusteringDataset):
    """CIFAR10 natural image dataset."""

    cache_dir: Path
    greyscale: bool
    normalize: bool
    _train_images: Array
    _train_labels: Array
    _test_images: Array
    _test_labels: Array

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        greyscale: bool = True,
        normalize: bool = True,
    ) -> "CIFAR10Dataset":
        """Load CIFAR10 dataset.

        Args:
            cache_dir: Directory for caching downloaded data
            greyscale: Whether to convert to grayscale
            normalize: Whether to normalize data

        Returns:
            Loaded CIFAR10 dataset
        """

        def transform_tensor(img):
            x = img.numpy()

            if greyscale:
                # Standard RGB to grayscale conversion
                x = np.expand_dims(
                    x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114,
                    axis=0,
                )

            return x.reshape(-1).astype(np.float32)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Scales to [0, 1]
                transforms.Lambda(transform_tensor),
            ]
        )

        try:
            # Load datasets
            train_dataset = datasets.CIFAR10(
                root=str(cache_dir), train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root=str(cache_dir), train=False, download=True, transform=transform
            )
        except (URLError, RuntimeError) as e:
            cifar_dir = cache_dir / "cifar-10-batches-py"
            raise RuntimeError(
                f"""Failed to download CIFAR10 dataset. If you're on an HPC system,
manually download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
and extract to: {cifar_dir}

Original error: {e!s}"""
            ) from e

        # Convert to arrays
        train_data = [x for x, _ in train_dataset]
        train_images = np.stack(train_data)
        train_labels = np.array([y for _, y in train_dataset])

        test_data = [x for x, _ in test_dataset]
        test_images = np.stack(test_data)
        test_labels = np.array([y for _, y in test_dataset])

        # Convert to JAX arrays
        train_images = jnp.array(train_images)
        test_images = jnp.array(test_images)
        train_labels = jnp.array(train_labels)
        test_labels = jnp.array(test_labels)

        if normalize:
            # Normalize using training statistics
            mean = jnp.mean(train_images, axis=0, keepdims=True)
            std = jnp.std(train_images, axis=0, keepdims=True)
            std = jnp.maximum(std, 1e-6)  # Prevent division by zero

            train_images = (train_images - mean) / std
            test_images = (test_images - mean) / std

            # Clip to reasonable range
            train_images = jnp.clip(train_images, -3.0, 3.0)
            test_images = jnp.clip(test_images, -3.0, 3.0)

        return cls(
            cache_dir=cache_dir,
            greyscale=greyscale,
            normalize=normalize,
            _train_images=train_images,
            _train_labels=train_labels,
            _test_images=test_images,
            _test_labels=test_labels,
        )

    @property
    @override
    def train_data(self) -> Array:
        return self._train_images

    @property
    @override
    def test_data(self) -> Array:
        return self._test_images

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
        if self.greyscale:
            return N_ROWS * N_COLS
        return N_ROWS * N_COLS * N_CHANNELS

    @property
    @override
    def n_classes(self) -> int:
        return (
            10  # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        )

    @property
    @override
    def observable_shape(self) -> tuple[int, int]:
        return (N_ROWS, N_COLS)

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        # Make room for prototype + members display
        return (N_ROWS, math.ceil(N_COLS * 1.5))

    @override
    def paint_observable(self, observable: Array, axes: Axes):
        """Visualize a single CIFAR10 image."""
        if self.greyscale:
            img = observable.reshape(N_ROWS, N_COLS)
            # Denormalize for display if normalized
            if self.normalize:
                # Map to [0, 1] for display
                img = img - img.min()
                img = img / (img.max() + 1e-8)
            axes.imshow(img, cmap="gray", interpolation="nearest")
        else:
            img = observable.reshape(N_CHANNELS, N_ROWS, N_COLS).transpose(1, 2, 0)
            if self.normalize:
                # Map each channel to [0, 1]
                for c in range(N_CHANNELS):
                    channel = img[:, :, c]
                    img[:, :, c] = (channel - channel.min()) / (
                        channel.max() - channel.min() + 1e-8
                    )
            axes.imshow(img, interpolation="nearest")
        axes.axis("off")

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Visualize a CIFAR10 cluster prototype and selected members."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        # Turn off the main axes frame
        axes.set_axis_off()

        # Get subplot specification and figure
        subplot_spec = axes.get_subplotspec()
        fig = axes.get_figure()

        if subplot_spec is None:
            raise ValueError("paint_cluster requires a subplot")

        assert isinstance(fig, Figure)

        # Create a grid layout: prototype on left, members grid on right
        gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, width_ratios=[1, 3], wspace=0.2
        )

        # Create axes for prototype
        proto_ax = fig.add_subplot(gs[0, 0])

        # Plot prototype
        self.paint_observable(prototype, proto_ax)
        proto_ax.set_title(
            f"Cluster {cluster_id}\nSize: {members.shape[0]}", fontsize=10
        )

        # Determine grid size for members (up to 16 members in a 4x4 grid)
        if members.shape[0] > 0:
            n_members = min(16, members.shape[0])
            grid_size = int(np.ceil(np.sqrt(n_members)))
            members_gs = GridSpecFromSubplotSpec(
                grid_size, grid_size, subplot_spec=gs[0, 1], wspace=0.1, hspace=0.1
            )

            # Plot selected members in grid
            for i in range(n_members):
                member_ax = fig.add_subplot(members_gs[i // grid_size, i % grid_size])
                self.paint_observable(members[i], member_ax)
                member_ax.axis("off")
