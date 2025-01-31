# src/apps/cli.py
import typer
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from plugins import register_plugins

from apps.clustering.core.config import ClusteringConfig

main = typer.Typer()
clustering_app = typer.Typer()
main.add_typer(clustering_app, name="clustering")


def compose_config(overrides: list[str]) -> DictConfig:
    """Compose configuration using Hydra."""
    # Register base config
    cs = ConfigStore.instance()
    cs.store(name="config", node=ClusteringConfig)

    register_plugins()

    with initialize():
        return compose(config_name="config", overrides=overrides)


train_overrides = typer.Argument(
    default=None, help="Configuration overrides (e.g., dataset=mnist model=hmog)"
)


@clustering_app.command()
def train(overrides: list[str] = train_overrides):
    """Train a clustering model."""
    cfg = compose_config(overrides)
    print(f"Training with config: {cfg}")
    from apps.clustering.train import train

    train(cfg)


if __name__ == "__main__":
    main()
