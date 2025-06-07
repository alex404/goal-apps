"""SVHN dataset implementation."""

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

# SVHN is 32x32 RGB images
N_ROWS = 32
N_COLS = 32
N_CHANNELS = 3


@dataclass
class SVHNConfig(ClusteringDatasetConfig):
    """Configuration for SVHN dataset.

    Parameters:
        greyscale: Whether to convert RGB images to grayscale
        crop_margin: Pixels to crop from each side to focus on the digit
    """

    _target_: str = "plugins.datasets.svhn.SVHNDataset.load"
    greyscale: bool = False  # Option to convert to grayscale
    crop_margin: int = 0  # Crop pixels from each side to focus on digit


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="svhn", node=SVHNConfig)


@dataclass(frozen=True)
class SVHNDataset(ClusteringDataset):
    """SVHN (Street View House Numbers) dataset."""

    cache_dir: Path
    greyscale: bool
    crop_margin: int
    _train_images: Array
    _train_labels: Array
    _test_images: Array
    _test_labels: Array

    @classmethod
    def load(
        cls, cache_dir: Path, greyscale: bool = True, crop_margin: int = 4
    ) -> "SVHNDataset":
        """Load SVHN dataset.

        Args:
            cache_dir: Directory for caching downloaded data
            greyscale: Whether to convert RGB images to grayscale
            crop_margin: Pixels to crop from each side to focus on the digit

        Returns:
            Loaded SVHN dataset
        """

        def transform_tensor(img):
            # Convert PyTorch tensor to numpy
            x = img.numpy()

            # Crop the image to focus on the digit
            if crop_margin > 0:
                m = crop_margin
                height, width = N_ROWS, N_COLS
                x = x[:, m : height - m, m : width - m]

            # Convert to grayscale if requested
            if greyscale:
                # Use standard RGB to grayscale conversion weights
                x = np.expand_dims(
                    x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114,
                    axis=0,
                )

            # Flatten and normalize to [0, 1]
            return x.reshape(-1).astype(np.float32)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Already normalizes to [0, 1]
                transforms.Lambda(transform_tensor),
            ]
        )

        try:
            # SVHN dataset uses 'train' and 'test' splits
            train_dataset = datasets.SVHN(
                root=str(cache_dir), split="train", download=True, transform=transform
            )
            test_dataset = datasets.SVHN(
                root=str(cache_dir), split="test", download=True, transform=transform
            )
        except (URLError, RuntimeError) as e:
            svhn_dir = cache_dir / "SVHN" / "raw"
            raise RuntimeError(
                f"""Failed to download SVHN dataset. If you're running on an HPC environment,
consider manually copying the dataset to:
{svhn_dir}

Required files:
- train_32x32.mat
- test_32x32.mat

If you have the dataset locally, you can copy it using:
scp -r /path/to/local/SVHN username@hpc:{svhn_dir}

Original error: {e!s}"""
            ) from e

        # Calculate flattened dimension based on whether grayscale conversion is applied
        flat_dim = (N_ROWS - 2 * crop_margin) * (N_COLS - 2 * crop_margin)
        if not greyscale:
            flat_dim *= N_CHANNELS

        # Convert to JAX arrays
        train_data = [x for x, _ in train_dataset]
        train_images = jnp.array(np.stack(train_data))
        train_labels = jnp.array(np.array([y for _, y in train_dataset]))

        test_data = [x for x, _ in test_dataset]
        test_images = jnp.array(np.stack(test_data))
        test_labels = jnp.array(np.array([y for _, y in test_dataset]))

        # Create and return the immutable instance
        return cls(
            cache_dir=cache_dir,
            greyscale=greyscale,
            crop_margin=crop_margin,
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
    def train_labels(self) -> Array:
        return self._train_labels

    @property
    def test_labels(self) -> Array:
        return self._test_labels

    @property
    @override
    def data_dim(self) -> int:
        height, width = self.observable_shape
        channels = 1 if self.greyscale else N_CHANNELS
        return height * width * channels

    @property
    @override
    def observable_shape(self) -> tuple[int, int]:
        height = N_ROWS - 2 * self.crop_margin
        width = N_COLS - 2 * self.crop_margin
        return (height, width)

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        height, width = self.observable_shape
        return (height, math.ceil(width * 1.5))

    @property
    @override
    def n_classes(self) -> int:
        return 10  # Digits 0-9

    @override
    def paint_observable(self, observable: Array, axes: Axes):
        height, width = self.observable_shape
        if self.greyscale:
            img = observable.reshape(height, width)
            axes.imshow(img, cmap="gray", interpolation="nearest")
        else:
            img = observable.reshape(N_CHANNELS, height, width).transpose(1, 2, 0)
            axes.imshow(img, interpolation="nearest")
        axes.axis("off")

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Visualize an SVHN digit prototype and selected members.

        Args:
            cluster_id: The cluster index
            prototype: The prototype for the cluster
            members: The members of the cluster
            axes: Matplotlib axes to draw on
        """
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        # Turn off the main axes frame
        axes.set_axis_off()

        # Get subplot specification and figure
        subplot_spec = axes.get_subplotspec()
        fig = axes.get_figure()

        if subplot_spec is None:
            raise ValueError("paint_prototype requires a subplot")

        assert isinstance(fig, Figure)

        # Create a grid layout: prototype on left, members grid on right
        gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec, width_ratios=[1, 3], wspace=0.2
        )

        # Create axes for prototype and members grid
        proto_ax = fig.add_subplot(gs[0, 0])

        # Plot prototype
        height, width = self.observable_shape
        if self.greyscale:
            prototype_img = prototype.reshape(height, width)
            proto_ax.imshow(prototype_img, cmap="gray", interpolation="nearest")
        else:
            prototype_img = prototype.reshape(N_CHANNELS, height, width).transpose(
                1, 2, 0
            )
            proto_ax.imshow(prototype_img, interpolation="nearest")

        proto_ax.set_title(f"Cluster {cluster_id}\nSize: {members.shape[0]}")
        proto_ax.axis("off")

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
                if self.greyscale:
                    member_img = members[i].reshape(height, width)
                    member_ax.imshow(member_img, cmap="gray", interpolation="nearest")
                else:
                    member_img = (
                        members[i].reshape(N_CHANNELS, height, width).transpose(1, 2, 0)
                    )
                    member_ax.imshow(member_img, interpolation="nearest")
                member_ax.axis("off")
