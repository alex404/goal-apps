### Imports ###

import logging
from typing import Any

from hydra.core.config_store import ConfigStore
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

log = logging.getLogger(__name__)

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

    for _, node in cs.repo.items():
        # Handle the case where node is a dict
        if isinstance(node, dict):
            for config_name, config_node in node.items():
                if hasattr(config_node, "group") and config_node.group:
                    group = config_node.group
                    if group not in groups:
                        groups[group] = []
                    # Remove .yaml extension if present
                    clean_name = config_name.replace(".yaml", "")
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
