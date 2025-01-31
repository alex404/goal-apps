from importlib import import_module
from pathlib import Path


def register_plugins() -> None:
    """Import and register all plugins."""
    plugin_types = ["datasets", "models"]

    for plugin_type in plugin_types:
        plugin_dir = Path(__file__).parent / plugin_type
        for file in plugin_dir.glob("*.py"):
            if file.stem != "__init__":
                import_module(f"plugins.{plugin_type}.{file.stem}")
