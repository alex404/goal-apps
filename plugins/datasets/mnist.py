"""MNIST dataset implementation."""

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
from numpy.typing import NDArray
from torchvision import datasets, transforms

from apps.interface import ClusteringDataset, ClusteringDatasetConfig

N_ROWS = 28
N_COLS = 28


@dataclass
class MNISTConfig(ClusteringDatasetConfig):
    """Configuration for MNIST dataset.

    Parameters:
        None

    """

    _target_: str = "plugins.datasets.mnist.MNISTDataset.load"


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="mnist", node=MNISTConfig)


@dataclass(frozen=True)
class MNISTDataset(ClusteringDataset):
    """MNIST handwritten digits dataset."""

    cache_dir: Path
    _train_images: Array
    _train_labels: Array
    _test_images: Array
    _test_labels: Array

    @classmethod
    def load(cls, cache_dir: Path) -> "MNISTDataset":
        """Load MNIST dataset.

        Args:
            cache_dir: Directory for caching downloaded data

        Returns:
            Loaded MNIST dataset
        """

        def transform_tensor(x: NDArray[np.uint8]) -> NDArray[np.float32]:
            return x.reshape(-1).astype(np.float32)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(transform_tensor)]
        )

        try:
            train_dataset = datasets.MNIST(
                root=str(cache_dir), train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root=str(cache_dir), train=False, download=True, transform=transform
            )
        except (URLError, RuntimeError) as e:
            mnist_dir = cache_dir / "MNIST" / "raw"
            raise RuntimeError(
                f"""Failed to download MNIST dataset. If you're running on an HPC environment,
consider manually copying the dataset to:
{mnist_dir}

Required files:
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

If you have the dataset locally, you can copy it using:
scp -r /path/to/local/MNIST username@hpc:{mnist_dir}

Original error: {e!s}"""
            ) from e

        train_images = jnp.array(train_dataset.data.numpy()).reshape(-1, 784) / 255.0
        train_labels = jnp.array(train_dataset.targets.numpy())
        test_images = jnp.array(test_dataset.data.numpy()).reshape(-1, 784) / 255.0
        test_labels = jnp.array(test_dataset.targets.numpy())

        # Create the immutable instance with all fields initialized
        return cls(
            cache_dir=cache_dir,
            _train_images=train_images,
            _train_labels=train_labels,
            _test_images=test_images,
            _test_labels=test_labels,
        )

    @property
    @override
    def observable_shape(self) -> tuple[int, int]:
        return (N_ROWS, N_COLS)

    @property
    @override
    def cluster_shape(self) -> tuple[int, int]:
        return (N_ROWS, math.ceil(N_COLS * 1.5))

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
        return N_ROWS * N_COLS

    @property
    def n_classes(self) -> int:
        return 10  # Digits 0-9

    @override
    def paint_observable(self, observable: Array, axes: Axes):
        img = observable.reshape(N_ROWS, N_COLS)
        axes.imshow(img, cmap="gray", interpolation="nearest")
        axes.axis("off")

    @override
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ) -> None:
        """Visualize an MNIST digit prototype and selected members.

        Args:
            prototype_artifact: Artifact containing prototype and member digits
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
        prototype_img = prototype.reshape(N_ROWS, N_COLS)
        proto_ax.imshow(prototype_img, cmap="gray", interpolation="nearest")
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
                member_img = members[i].reshape(N_ROWS, N_COLS)
                member_ax.imshow(member_img, cmap="gray", interpolation="nearest")
                member_ax.axis("off")
