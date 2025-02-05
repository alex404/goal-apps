"""Shared utilities for GOAL examples."""

import logging
from pathlib import Path

import hydra
import jax
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ..configs import RunConfig
from ..plugins import Dataset, Model
from ..util import print_config_tree
from .handler import RunHandler
from .logger import JaxLogger, setup_logging

### Preamble ###

logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)

log = logging.getLogger(__name__)

### Initialization ###


def _initialize_jax(device: str = "cpu", disable_jit: bool = False) -> None:
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)


def initialize_run(
    run_type: type[RunConfig],
    overrides: list[str],
) -> tuple[RunHandler, Dataset, Model[Dataset], JaxLogger]:
    """Initialize a new run with hydra config and wandb logging."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=run_type)
    proot = Path(__file__).parents[3]

    # Initialize hydra config
    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(proot / "config")
    ):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    print_config_tree(OmegaConf.to_container(cfg, resolve=True))

    # Initialize JAX
    _initialize_jax(device=cfg.device, disable_jit=not cfg.jit)

    # Initialize run handler
    handler = RunHandler(name=cfg.run_name, project_root=proot)

    # Save config to run directory
    OmegaConf.save(cfg, handler.run_dir / "config.yaml")

    setup_logging(handler.run_dir)

    log.info(f"Run name: {cfg.run_name}")
    log.info(f"with JIT: {cfg.jit}")

    dataset: Dataset = hydra.utils.instantiate(cfg.dataset, cache_dir=handler.cache_dir)

    # Instantiate model
    model: Model[Dataset] = hydra.utils.instantiate(
        cfg.model, data_dim=dataset.data_dim
    )

    logger: JaxLogger = hydra.utils.instantiate(
        cfg.logger,
        run_name=cfg.run_name,
        run_dir=handler.run_dir,
    )

    # will return logger as well
    return handler, dataset, model, logger
