### Imports ###

from __future__ import annotations

import logging
import math
from pathlib import Path
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


### Terminal Visualizations ###

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a unicode sparkline."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    spread = hi - lo
    if spread == 0:
        idx = len(_SPARK_CHARS) // 2
        return _SPARK_CHARS[idx] * min(len(values), width)

    # Bin values if more than width
    if len(values) > width:
        bin_size = len(values) / width
        binned = []
        for i in range(width):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            binned.append(sum(values[start:end]) / max(1, end - start))
        values = binned

    chars = []
    for v in values:
        idx = int((v - lo) / spread * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _histogram_bar(values: list[float], n_bins: int = 20) -> str:
    """Render a horizontal histogram as unicode blocks."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"[dim](all {lo:.4g})[/dim]"
    bins = [0] * n_bins
    for v in values:
        idx = int((v - lo) / (hi - lo) * (n_bins - 1))
        idx = max(0, min(idx, n_bins - 1))
        bins[idx] += 1
    max_count = max(bins)
    if max_count == 0:
        return ""
    chars = []
    for count in bins:
        level = int(count / max_count * (len(_SPARK_CHARS) - 1))
        chars.append(_SPARK_CHARS[level])
    return "".join(chars)


def _fmt_val(v: float | int | str) -> str:
    """Format a parameter value compactly."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 0.01 or abs(v) >= 1000:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def print_objective_sparkline(trials: list[Any], direction: str) -> None:
    """Print objective value progression as a sparkline over trial number."""
    if not trials:
        return
    # Sort by trial number
    sorted_trials = sorted(trials, key=lambda t: t.number)
    values = [t.value for t in sorted_trials if t.value is not None]
    if not values:
        return

    # Running best
    best_so_far = []
    current_best = values[0]
    maximize = direction == "MAXIMIZE"
    for v in values:
        if maximize:
            current_best = max(current_best, v)
        else:
            current_best = min(current_best, v)
        best_so_far.append(current_best)

    rprint(f"\n[bold]Objective progression[/bold] (n={len(values)} completed)")
    rprint(f"  All trials:  {_sparkline(values, 60)}  [{_fmt_val(min(values))}, {_fmt_val(max(values))}]")
    rprint(f"  Running best: {_sparkline(best_so_far, 60)}  {_fmt_val(best_so_far[0])} → {_fmt_val(best_so_far[-1])}")


def print_parameter_distributions(trials: list[Any], best_params: dict[str, Any]) -> None:
    """Print per-parameter histograms and statistics for completed trials."""
    if not trials:
        return

    # Collect all param values across trials
    param_values: dict[str, list[float | int | str]] = {}
    for t in trials:
        for k, v in t.params.items():
            param_values.setdefault(k, []).append(v)

    table = Table(title="Parameter Distributions")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Range", style="dim", no_wrap=True)
    table.add_column("Best", style="green bold", no_wrap=True)
    table.add_column("Distribution", no_wrap=True)

    for param, values in sorted(param_values.items()):
        best_val = best_params.get(param)
        best_str = _fmt_val(best_val) if best_val is not None else "?"

        # Check if categorical (strings or small set of discrete values)
        if any(isinstance(v, str) for v in values):
            # Categorical - show counts
            counts: dict[str, int] = {}
            for v in values:
                counts[str(v)] = counts.get(str(v), 0) + 1
            parts = [f"{k}:{c}" for k, c in sorted(counts.items(), key=lambda x: -x[1])]
            dist_str = "  ".join(parts)
            range_str = f"{len(counts)} categories"
        else:
            numeric = [float(v) for v in values]
            lo, hi = min(numeric), max(numeric)
            # Detect log-scale: if range spans >2 orders of magnitude
            if lo > 0 and hi / lo > 100:
                range_str = f"{_fmt_val(lo)} … {_fmt_val(hi)} [dim](log)[/dim]"
                numeric_for_hist = [math.log10(v) for v in numeric]
            else:
                range_str = f"{_fmt_val(lo)} … {_fmt_val(hi)}"
                numeric_for_hist = numeric
            dist_str = _histogram_bar(numeric_for_hist)

        table.add_row(param, range_str, best_str, dist_str)

    rprint()
    rprint(table)


def print_param_objective_scatter(
    trials: list[Any], _direction: str, top_n: int = 3
) -> None:
    """For each parameter, show a scatter of value vs objective using braille-ish text.

    Shows the top_n most "interesting" parameters (highest variance in objective
    across parameter bins).
    """
    if len(trials) < 4:
        return

    # Collect numeric params and their objectives
    param_data: dict[str, list[tuple[float, float]]] = {}
    for t in trials:
        if t.value is None:
            continue
        for k, v in t.params.items():
            if isinstance(v, (int, float)):
                param_data.setdefault(k, []).append((float(v), float(t.value)))

    if not param_data:
        return

    # Rank parameters by "interestingness" - variance of mean objective per bin
    param_scores: dict[str, float] = {}
    for param, pairs in param_data.items():
        if len(pairs) < 4:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        lo, hi = min(xs), max(xs)
        if lo == hi:
            param_scores[param] = 0.0
            continue
        n_bins = min(8, len(pairs) // 2)
        bin_means = []
        for i in range(n_bins):
            bin_lo = lo + (hi - lo) * i / n_bins
            bin_hi = lo + (hi - lo) * (i + 1) / n_bins
            bin_ys = [y for x, y in pairs if bin_lo <= x <= bin_hi]
            if bin_ys:
                bin_means.append(sum(bin_ys) / len(bin_ys))
        if len(bin_means) >= 2:
            mean = sum(bin_means) / len(bin_means)
            param_scores[param] = sum((m - mean) ** 2 for m in bin_means) / len(
                bin_means
            )
        else:
            param_scores[param] = 0.0

    ranked = sorted(param_scores, key=lambda p: param_scores[p], reverse=True)[:top_n]
    if not ranked:
        return

    rprint(f"\n[bold]Parameter vs Objective[/bold] (top {len(ranked)} by variance)")

    height = 8
    width = 40
    for param in ranked:
        pairs = param_data[param]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)

        if x_lo == x_hi or y_lo == y_hi:
            continue

        # Build grid
        grid = [[" "] * width for _ in range(height)]
        for x, y in pairs:
            col = int((x - x_lo) / (x_hi - x_lo) * (width - 1))
            row = int((y - y_lo) / (y_hi - y_lo) * (height - 1))
            col = max(0, min(col, width - 1))
            row = max(0, min(row, height - 1))
            # Invert row so high values are at top
            grid[height - 1 - row][col] = "●"

        rprint(f"\n  [cyan]{param}[/cyan]  x:[dim]{_fmt_val(x_lo)}…{_fmt_val(x_hi)}[/dim]  obj:[dim]{_fmt_val(y_lo)}…{_fmt_val(y_hi)}[/dim]")
        rprint(f"  {_fmt_val(y_hi):>8s} ┤", end="")
        rprint(grid[0] and "".join(grid[0]) or "")
        for row_idx in range(1, height - 1):
            rprint(f"  {'':>8s} │{''.join(grid[row_idx])}")
        rprint(f"  {_fmt_val(y_lo):>8s} ┤{''.join(grid[height - 1])}")
        rprint(f"  {'':>8s}  └{'─' * width}")


### Matplotlib Parameter Histogram Plot ###


def plot_param_histograms(
    trials: list[Any],
    direction: str,
    pct: float = 10.0,
    output: Path | None = None,
) -> Path:
    """Generate parameter histogram figure comparing all trials vs top/bottom pct%.

    For each searched parameter, draws two overlaid histograms:
      - grey: all completed trials (coverage)
      - coloured: top pct% by objective (good region)

    Log-scale params (sampled with log=True) use log-binned x-axis.
    Categorical params use a bar chart.

    Args:
        trials: Completed Optuna trials.
        direction: "MAXIMIZE" or "MINIMIZE".
        pct: Percentile threshold — top pct% (maximize) or bottom pct% (minimize).
        output: Save path. Defaults to optuna-params.png in cwd.

    Returns:
        Path where the figure was saved.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import optuna.distributions as od

    if not trials:
        raise ValueError("No completed trials to plot.")

    maximize = direction == "MAXIMIZE"
    values = [t.value for t in trials if t.value is not None]
    threshold = np.percentile(values, 100 - pct if maximize else pct)
    good_set = set(
        t.number
        for t in trials
        if t.value is not None and (t.value >= threshold if maximize else t.value <= threshold)
    )

    # Collect param values for all vs good
    param_all: dict[str, list[Any]] = {}
    param_good: dict[str, list[Any]] = {}
    for t in trials:
        for k, v in t.params.items():
            param_all.setdefault(k, []).append(v)
            if t.number in good_set:
                param_good.setdefault(k, []).append(v)

    params = sorted(param_all.keys())
    n = len(params)
    if n == 0:
        raise ValueError("No parameters found in trials.")

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flat)

    label_dir = "top" if maximize else "bottom"
    color_good = "#e05252"
    color_all = "#aaaaaa"
    n_bins = 20

    # Collect distributions: first trial that has each param wins
    param_dist: dict[str, Any] = {}
    for t in trials:
        for k, d in t.distributions.items():
            if k not in param_dist:
                param_dist[k] = d

    for ax, param in zip(axes_flat, params):
        all_vals = param_all[param]
        good_vals = param_good.get(param, [])

        dist = param_dist.get(param)

        # Determine axis type from the sampling distribution
        is_categorical = isinstance(dist, od.CategoricalDistribution)
        use_log = (
            isinstance(dist, (od.FloatDistribution, od.IntDistribution))
            and getattr(dist, "log", False)
        )

        if is_categorical:
            categories = sorted(set(str(v) for v in all_vals))
            all_counts = [sum(1 for v in all_vals if str(v) == c) for c in categories]
            good_counts = [sum(1 for v in good_vals if str(v) == c) for c in categories]
            x = np.arange(len(categories))
            ax.bar(x, all_counts, color=color_all, label="all", zorder=2)
            ax.bar(x, good_counts, color=color_good, alpha=0.85, label=f"{label_dir} {pct:.0f}%", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
        else:
            numeric_all = np.array([float(v) for v in all_vals])
            numeric_good = np.array([float(v) for v in good_vals]) if good_vals else np.array([])
            lo, hi = numeric_all.min(), numeric_all.max()

            if use_log and lo > 0:
                bins = np.logspace(math.log10(lo), math.log10(hi), n_bins + 1)
                ax.set_xscale("log")
            else:
                bins = np.linspace(lo, hi, n_bins + 1)

            ax.hist(numeric_all, bins=bins, color=color_all, label="all", zorder=2)
            if len(numeric_good) > 0:
                ax.hist(numeric_good, bins=bins, color=color_good, alpha=0.85,
                        label=f"{label_dir} {pct:.0f}%", zorder=3)

        ax.set_title(param, fontsize=9, fontweight="bold")
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    n_good = len(good_set)
    fig.suptitle(
        f"Parameter distributions — {len(trials)} trials, "
        f"{label_dir} {pct:.0f}% highlighted (n={n_good}, "
        f"objective ≥{threshold:.4g})" if maximize else
        f"Parameter distributions — {len(trials)} trials, "
        f"{label_dir} {pct:.0f}% highlighted (n={n_good}, "
        f"objective ≤{threshold:.4g})",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    if output is None:
        output = Path("optuna-params.png")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output
