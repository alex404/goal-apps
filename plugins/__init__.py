from importlib import import_module
from pathlib import Path


def register_plugins() -> None:
    """Import and register all plugins."""
    plugin_types = ["datasets", "models"]

    for plugin_type in plugin_types:
        plugin_dir = Path(__file__).parent / plugin_type
        # Look for both .py files and directories with __init__.py
        for item in plugin_dir.iterdir():
            # Skip __init__.py files
            if item.name == "__init__.py":
                continue
            # Handle .py files
            if item.is_file() and item.suffix == ".py":
                import_module(f"plugins.{plugin_type}.{item.stem}")
            # Handle directories that contain __init__.py (plugin packages)
            elif item.is_dir() and (item / "__init__.py").exists():
                import_module(f"plugins.{plugin_type}.{item.name}")
