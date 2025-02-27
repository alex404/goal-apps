### Imports ###

import logging
import re
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

log = logging.getLogger(__name__)

### Really Util ###


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name)
    return name.lower()


### Sweep Management ###


def create_sweep_config(overrides: list[str], base_sweep: str | None) -> dict[str, Any]:
    parameters = {}

    if base_sweep is not None:
        proot = Path(__file__).parents[2]
        sweep_dir = str(proot / "config" / "sweeps")
        sweep_path = Path(sweep_dir) / f"{base_sweep}.yaml"

        base_config = OmegaConf.load(sweep_path)
        base_params = OmegaConf.to_container(base_config)
        parameters.update(base_params)
        log.info(f"Loaded base parameters from {sweep_path}")

    for override in overrides:
        if "=" not in override:
            continue

        param, value = override.split("=", 1)
        if "," in value:
            try:
                # Try to parse as list of numbers first
                values = [float(x) if "." in x else int(x) for x in value.split(",")]
                parameters[param] = {"values": values}
            except ValueError:
                # If number parsing fails, treat as list of strings
                values = [x.strip() for x in value.split(",")]
                if len(values) > 1:
                    parameters[param] = {"values": values}
                else:
                    parameters[param] = {"value": value}
        else:
            # Single value
            parameters[param] = {"value": value}

    return {
        "program": "${program}",  # Let wandb handle the program path
        "method": "grid",
        "parameters": parameters,
        "command": [
            "${env}",
            "${interpreter}",
            "-m",  # Use python module mode
            "apps.cli",  # Direct module reference
            "train",
            "${args_no_hyphens}",
        ],
    }


### Pretty Print Configs ###


def print_config_tree(data: dict[str, Any] | list[Any] | Any) -> Tree:
    """Create a rich tree from a dictionary."""
    tree = Tree("[bold]config[/bold]")
    _build_tree(tree, data)

    # Print it inside a panel for extra clarity
    rprint(Panel(tree, title="Hydra Config Overview", border_style="green"))
    return tree


def print_sweep_tree(data: dict[str, Any] | list[Any] | Any) -> Tree:
    """Create a rich tree from a dictionary."""
    tree = Tree("[bold]sweep[/bold]")
    _build_tree(tree, data)

    # Print it inside a panel for extra clarity
    rprint(Panel(tree, title="Sweep Config Overview", border_style="green"))
    return tree


def _build_tree(tree: Tree, data: dict[str, Any] | list[Any] | Any) -> None:
    """Recursively build a compact tree from a dictionary or list."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):  # Only nest if necessary
                branch = tree.add(f"[bold]{key}[/bold]")
                _build_tree(branch, value)
            else:
                tree.add(f"[bold]{key}[/bold]: [cyan]{value}[/cyan]")  # Inline values
    elif isinstance(data, list):
        for item in data:
            tree.add(f"[list] [cyan]{item}[/cyan]")  # Show list items inline
    else:
        tree.add(
            f"[cyan]{data}[/cyan]"
        )  # For direct values (shouldn't be hit at top-level)


### Plugin Inspection ###


def get_store_groups() -> dict[str, list[str]]:
    """Get available configs from ConfigStore by group."""
    groups: dict[str, list[str]] = {}

    cs = ConfigStore.instance()

    for name, node in cs.repo.items():
        # Handle the case where node is a dict
        if isinstance(node, dict):
            for config_name, config_node in node.items():  # pyright: ignore[reportUnknownVariableType]
                if hasattr(config_node, "group") and config_node.group:
                    group = config_node.group  # pyright: ignore[reportUnknownVariableType]
                    if group not in groups:
                        groups[group] = []
                    # Remove .yaml extension if present
                    clean_name = config_name.replace(".yaml", "")  # pyright: ignore[reportUnknownVariableType]
                    groups[group].append(clean_name)

    return groups


def format_config_table(name: str, params: dict[str, Any]) -> tuple[str | None, Table]:
    """Format configuration parameters as a rich table.

    Args:
        name: Name of the plugin/config
        params: Dictionary of parameters

    Returns:
        tuple of (target implementation path, formatted table)
    """
    from rich.table import Table

    # Create table of parameters
    table = Table(title=f"{name.upper()} Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Value", style="yellow")

    # Extract target
    target = params.get("_target_")

    # Add parameter rows
    for param_name, value in params.items():
        if param_name != "_target_":
            param_type: str = type(value).__name__ if value is not None else "Required"
            table.add_row(
                str(param_name),
                param_type,
                str(value) if value is not None else "Required",
            )

    return target, table
