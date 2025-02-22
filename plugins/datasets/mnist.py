"""MNIST dataset implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import override
from urllib.error import URLError

import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.axes import Axes
from numpy.typing import NDArray
from torchvision import datasets, transforms

from apps.configs import ClusteringDatasetConfig
from apps.plugins import ClusteringDataset, ObservableArtifact


@dataclass
class MNISTConfig(ClusteringDatasetConfig):
    """Configuration for MNIST dataset.

    Parameters:
        None

    """

    _target_: str = "plugins.datasets.mnist.MNISTDataset"


# Register config
cs = ConfigStore.instance()
cs.store(group="dataset", name="mnist", node=MNISTConfig)


class MNISTDataset(ClusteringDataset):
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

        self.cache_dir: Path = cache_dir

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

        self._train_images = train_images
        self._train_labels = train_labels
        self._test_images = test_images
        self._test_labels = test_labels

    @property
    @override
    def train_data(self) -> Array:
        return self._train_images

    @property
    @override
    def test_data(self) -> Array:
        return self._test_images

    @property
    def train_labels(self) -> Array:
        return self._train_labels

    @property
    def test_labels(self) -> Array:
        return self._test_labels

    @property
    @override
    def data_dim(self) -> int:
        return 784  # 28x28 images

    @property
    def n_classes(self) -> int:
        return 10  # Digits 0-9

    @override
    def observable_artifact(self, observable: Array) -> ObservableArtifact:
        return ObservableArtifact(obs=observable, shape=(28, 28))

    @override
    @staticmethod
    def paint_observable(observable: ObservableArtifact, axes: Axes):
        """Visualize a single MNIST digit.

        Args:
            obs: Flattened image array of shape (784,)

        Returns:
            Tuple of:
            - Image dimensions (height, width)
            - Figure containing the visualization
        """

        # Create figure
        obs = observable.obs
        shp_rws, shp_cls = observable.shape

        # Display image
        img = obs.reshape(shp_rws, shp_cls)
        axes.imshow(img, cmap="gray", interpolation="nearest")
        axes.axis("off")
