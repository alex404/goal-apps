### Imports ###

import logging
from typing import Any

from hydra.core.config_store import ConfigStore
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

log = logging.getLogger(__name__)

### Sweep Management ###


def _parse_args_to_parameters(args: list[str]) -> dict[str, Any]:
    """Parse CLI arguments into a wandb parameters dictionary."""
    parameters: dict[str, Any] = {}

    for arg in args:
        if "=" not in arg:
            continue

        param, value = arg.split("=", 1)
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

    return parameters


def create_sweep_config(overrides: list[str]) -> dict[str, Any]:
    """Create wandb sweep config from override strings."""
    # Parse all overrides into a single dictionary
    parameters = {}

    # Then parse CLI overrides (these will overwrite any overlapping parameters)
    cli_parameters = _parse_args_to_parameters(overrides)
    parameters.update(cli_parameters)

    return {
        "program": "${program}",
        "method": "grid",
        "parameters": parameters,
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            "apps.cli",
            "train",
            "${args_no_hyphens}",
        ],
    }


def sample_sweep_args(sweep_config: dict[str, Any]) -> list[str]:
    """Sample command line arguments from a wandb sweep config."""
    args: list[str] = []

    # Extract parameters
    parameters = sweep_config.get("parameters", {})

    # Sample one value from each parameter
    for param_name, param_config in parameters.items():
        if param_name == "use_wandb":
            # Don't use wandb for validation
            continue

        if "values" in param_config:
            # If it's a list of values, take the first one
            args.append(f"{param_name}={param_config['values'][0]}")
        elif "value" in param_config:
            # If it's a single value, use it
            args.append(f"{param_name}={param_config['value']}")
        elif "min" in param_config and "max" in param_config:
            # If it's a range, take the min value
            args.append(f"{param_name}={param_config['min']}")

    return args


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
