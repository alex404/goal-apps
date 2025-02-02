### Imports ###

from typing import Any

from hydra.core.config_store import ConfigStore
from rich.table import Table

### Preamable ###

cs = ConfigStore.instance()


def get_store_groups() -> dict[str, list[str]]:
    """Get available configs from ConfigStore by group."""
    groups: dict[str, list[str]] = {}

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
    table.add_column("Default", style="yellow")

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
