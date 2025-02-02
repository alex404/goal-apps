# src/apps/cli.py
from pathlib import Path

import typer
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from plugins import register_plugins

from apps.clustering.core.config import ClusteringConfig
from apps.experiments import ExperimentConfig, ExperimentHandler

main = typer.Typer()
clustering_app = typer.Typer()
main.add_typer(clustering_app, name="clustering")
package_root = Path(__file__).parents[2]


def compose_config(
    exp_conf: type[ExperimentConfig], overrides: list[str]
) -> DictConfig:
    """Compose configuration using Hydra."""
    # Register base config
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=exp_conf)

    register_plugins()

    with initialize_config_dir(
        version_base="1.3", config_dir=str(package_root / "config")
    ):
        return compose(config_name="config", overrides=overrides)


train_overrides = typer.Argument(
    default=None, help="Configuration overrides (e.g., dataset=mnist model=hmog)"
)


@clustering_app.command()
def train(overrides: list[str] = train_overrides):
    """Train a clustering model."""
    cfg = compose_config(ClusteringConfig, overrides)
    print(f"Training with config: {cfg}")
    from apps.clustering.train import train

    train(cfg)


analyze_overrides = typer.Argument(
    default=None, help="Configuration overrides (e.g., dataset=mnist model=hmog)"
)


# In cli.py
@clustering_app.command()
def analyze(overrides: list[str] = analyze_overrides):
    """Analyze results from a trained clustering model."""
    from apps.clustering.analyze import analyze

    cfg = compose_config(ClusteringConfig, overrides)

    analyze(ExperimentHandler(cfg.experiment))


if __name__ == "__main__":
    main()
