"""MNIST dataset implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from numpy.typing import NDArray
from torchvision import datasets, transforms

from apps.clustering.core.config import DatasetConfig
from apps.clustering.core.datasets import SupervisedDataset


@dataclass
class MNISTConfig(DatasetConfig):
    """Configuration for MNIST dataset.

    Parameters:
        None

    """

    _target_: str = "plugins.datasets.mnist.MNISTDataset"


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="mnist", node=MNISTConfig)


class MNISTDataset(SupervisedDataset):
    """MNIST handwritten digits dataset."""

    _train_images: Array
    _train_labels: Array
    _test_images: Array
    _test_labels: Array

    def __init__(self, cache_dir: Path) -> None:
        """Load MNIST dataset.

        Args:
            cache_dir: Directory for caching downloaded data

        Returns:
            Loaded MNIST dataset
        """

        super().__init__(cache_dir)

        def transform_tensor(x: NDArray[np.uint8]) -> NDArray[np.float32]:
            return x.reshape(-1).astype(np.float32)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(transform_tensor)]
        )

        train_dataset = datasets.MNIST(
            root=str(cache_dir), train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=str(cache_dir), train=False, download=True, transform=transform
        )

        train_images = jnp.array(train_dataset.data.numpy()).reshape(-1, 784) / 255.0
        train_labels = jnp.array(train_dataset.targets.numpy())
        test_images = jnp.array(test_dataset.data.numpy()).reshape(-1, 784) / 255.0
        test_labels = jnp.array(test_dataset.targets.numpy())

        self._train_images = train_images
        self._train_labels = train_labels
        self._test_images = test_images
        self._test_labels = test_labels

    @property
    @override
    def train_images(self) -> Array:
        return self._train_images

    @property
    @override
    def test_images(self) -> Array:
        return self._test_images

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
        return 784  # 28x28 images

    @property
    @override
    def n_classes(self) -> int:
        return 10  # Digits 0-9

    @override
    def visualize_observable(
        self, observable: Array, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Visualize MNIST digit as grayscale image.

        Args:
            observable: Flattened image array of shape (784,)
            ax: Optional axes to plot on
            **kwargs: Additional arguments passed to imshow

        Returns:
            The matplotlib axes containing the visualization
        """
        if ax is None:
            _, ax = plt.subplots()

        img = observable.reshape(28, 28)
        ax.imshow(img, cmap="gray", **kwargs)
        ax.axis("off")
        return ax
