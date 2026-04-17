### Imports ###

from __future__ import annotations

import logging
import math
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
    """Format a parameter value compactly, stripping redundant zeros from exponents."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 0.01 or abs(v) >= 1000:
            s = f"{v:.2e}"
            # 1.00e-05 -> 1e-5, 2.50e+02 -> 2.5e2
            mantissa, exp = s.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exp_sign = "-" if exp.startswith("-") else ""
            exp_digits = str(int(exp[1:]))  # strip leading zeros
            return f"{mantissa}e{exp_sign}{exp_digits}"
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
    rprint(
        f"  All trials:  {_sparkline(values, 60)}  [{_fmt_val(min(values))}, {_fmt_val(max(values))}]"
    )
    rprint(
        f"  Running best: {_sparkline(best_so_far, 60)}  {_fmt_val(best_so_far[0])} → {_fmt_val(best_so_far[-1])}"
    )


def print_parameter_distributions(
    trials: list[Any], best_params: dict[str, Any]
) -> None:
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

        rprint(
            f"\n  [cyan]{param}[/cyan]  x:[dim]{_fmt_val(x_lo)}…{_fmt_val(x_hi)}[/dim]  obj:[dim]{_fmt_val(y_lo)}…{_fmt_val(y_hi)}[/dim]"
        )
        rprint(f"  {_fmt_val(y_hi):>8s} ┤", end="")
        rprint((grid[0] and "".join(grid[0])) or "")
        for row_idx in range(1, height - 1):
            rprint(f"  {'':>8s} │{''.join(grid[row_idx])}")
        rprint(f"  {_fmt_val(y_lo):>8s} ┤{''.join(grid[height - 1])}")
        rprint(f"  {'':>8s}  └{'─' * width}")


### Terminal Parameter Distribution Plot ###


def _good_set(trials: list[Any], direction: str, pct: float) -> set[int]:
    """Return trial numbers in the top/bottom pct% by objective."""
    values = [t.value for t in trials if t.value is not None]
    if not values:
        return set()
    maximize = direction == "MAXIMIZE"
    threshold = float(
        sorted(values, reverse=maximize)[max(0, int(len(values) * pct / 100) - 1)]
    )
    return {
        t.number
        for t in trials
        if t.value is not None
        and (t.value >= threshold if maximize else t.value <= threshold)
    }


def _param_dists(trials: list[Any]) -> dict[str, Any]:
    """Collect Optuna distribution objects, one per parameter."""
    dists: dict[str, Any] = {}
    for t in trials:
        for k, d in t.distributions.items():
            if k not in dists:
                dists[k] = d
    return dists


def _range_str(dist: Any) -> str:
    """Format the configured search range from an Optuna distribution."""
    import optuna.distributions as od

    if isinstance(dist, od.CategoricalDistribution):
        return f"{len(dist.choices)} categories"
    lo, hi = dist.low, dist.high
    use_log = getattr(dist, "log", False)
    scale = "log" if use_log else "linear"
    return f"[{_fmt_val(lo)}, {_fmt_val(hi)}] [dim]({scale})[/dim]"


def _hist_bar(
    values: list[float], lo: float, hi: float, use_log: bool, n_bins: int = 20
) -> str:
    """Render a histogram bar using the configured search range as fixed bin boundaries."""
    if not values or lo == hi:
        return "[dim](all same)[/dim]" if values else "[dim](none)[/dim]"
    bins = [0] * n_bins
    for v in values:
        if use_log and lo > 0:
            frac = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            frac = (v - lo) / (hi - lo)
        idx = min(n_bins - 1, max(0, int(frac * n_bins)))
        bins[idx] += 1
    max_count = max(bins)
    if max_count == 0:
        return ""
    return "".join(
        _SPARK_CHARS[int(c / max_count * (len(_SPARK_CHARS) - 1))] for c in bins
    )


def _strip_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def _human_round(value: float) -> str:
    """Round to 1 significant digit; use scientific notation for |exponent| > 2."""
    if value == 0:
        return "0"
    exp = math.floor(math.log10(abs(value)))
    coeff = round(value / 10**exp)
    rounded = coeff * 10**exp
    if abs(exp) > 2:
        return f"{coeff}e{exp}"
    if exp >= 0:
        return str(int(rounded))
    return f"{rounded:.{-exp}f}"


def print_trial_summary(
    completed: list[Any],
    direction: str,
    pct: float = 10.0,
    top_n: int = 10,
) -> None:
    """Print best trial params, human-rounded params, and top N in a compact layout."""
    import statistics

    import optuna.distributions as od
    from rich.columns import Columns

    maximize = direction == "MAXIMIZE"
    best = (
        max(completed, key=lambda t: t.value or 0)
        if maximize
        else min(completed, key=lambda t: t.value or 0)
    )
    ranked = sorted(completed, key=lambda t: t.value or 0, reverse=maximize)

    good_nums = _good_set(completed, direction, pct)
    good_trials = [t for t in completed if t.number in good_nums]
    dists = _param_dists(completed)
    label_dir = "top" if maximize else "bottom"

    # --- Left: param table (best + human best) ---
    param_table = Table(
        title=f"Best trial: t{best.number} ({best.value:.6f})",
        show_header=True,
        header_style="bold",
        title_style="bold",
    )
    param_table.add_column("Parameter", style="cyan", no_wrap=True)
    param_table.add_column(f"t{best.number}", no_wrap=True)
    param_table.add_column(
        f"Human ({label_dir} {pct:.4g}%, n={len(good_trials)})",
        style="green",
        no_wrap=True,
    )

    for param in sorted(dists.keys()):
        display = _strip_model_prefix(param)
        best_val = best.params.get(param)
        # Human-rounded value
        good_values = [t.params[param] for t in good_trials if param in t.params]
        if isinstance(dists.get(param), od.CategoricalDistribution):
            human_val = (
                str(max(set(good_values), key=good_values.count)) if good_values else ""
            )
        elif good_values:
            human_val = _human_round(statistics.median(good_values))
        else:
            human_val = ""
        param_table.add_row(display, str(best_val), human_val)

    # --- Right: top N list ---
    top_table = Table(
        title=f"Top {min(top_n, len(ranked))} trials",
        show_header=True,
        header_style="bold",
        title_style="bold",
    )
    top_table.add_column("Trial", style="cyan", no_wrap=True)
    top_table.add_column("Value", no_wrap=True)
    for t in ranked[:top_n]:
        top_table.add_row(f"t{t.number}", f"{t.value:.6f}")

    rprint()
    rprint(Columns([param_table, top_table], padding=(0, 2)))


def print_param_distributions_split(
    completed: list[Any],
    direction: str,
    pct: float = 10.0,
    n_bins: int = 20,
    all_trials: list[Any] | None = None,
) -> None:
    """Print parameter distributions showing three regimes: sampled, stable, top-pct%.

    - Sampled: all trials including pruned (shows what was tried)
    - Stable: completed trials only (shows what converged numerically)
    - Top pct%: top performers among completed (shows what worked well)

    Bin boundaries are derived from each parameter's configured search range.
    """
    import optuna.distributions as od

    if not completed:
        rprint("[red]No completed trials.[/red]")
        return

    coverage_trials = all_trials if all_trials is not None else completed

    maximize = direction == "MAXIMIZE"
    good = _good_set(completed, direction, pct)
    dists = _param_dists(coverage_trials)

    param_sampled: dict[str, list[Any]] = {}
    param_stable: dict[str, list[Any]] = {}
    param_good: dict[str, list[Any]] = {}
    for t in coverage_trials:
        for k, v in t.params.items():
            param_sampled.setdefault(k, []).append(v)
    for t in completed:
        for k, v in t.params.items():
            param_stable.setdefault(k, []).append(v)
            if t.number in good:
                param_good.setdefault(k, []).append(v)

    label_dir = "top" if maximize else "bottom"
    n_good = len(good)
    obj_vals = [t.value for t in completed if t.value is not None]
    threshold_idx = max(0, int(len(obj_vals) * pct / 100) - 1)
    threshold = sorted(obj_vals, reverse=maximize)[threshold_idx]
    threshold_sym = "≥" if maximize else "≤"

    title = (
        f"Parameter Distributions  —  {len(coverage_trials)} sampled, {len(completed)} completed  |  "
        f"{label_dir} {pct:.4g}%: n={n_good}, objective {threshold_sym}{threshold:.4g}"
    )

    # --- Continuous histogram table ---
    cont_params = [
        p
        for p in sorted(param_sampled.keys())
        if not (
            dists.get(p) is None or isinstance(dists.get(p), od.CategoricalDistribution)
        )
    ]
    if cont_params:
        hist_table = Table(title=title, show_header=True, header_style="bold")
        hist_table.add_column("Model Parameter", style="cyan", no_wrap=True)
        hist_table.add_column("Range", style="dim", no_wrap=True)
        hist_table.add_column(f"Sampled (n={len(coverage_trials)})", no_wrap=True)
        hist_table.add_column(
            f"Completed (n={len(completed)})", style="yellow", no_wrap=True
        )
        hist_table.add_column(
            f"{label_dir.capitalize()} {pct:.4g}% (n={n_good})",
            style="red",
            no_wrap=True,
        )

        for param in cont_params:
            dist = dists[param]
            lo, hi = float(dist.low), float(dist.high)
            use_log = bool(getattr(dist, "log", False))
            sampled_bar = _hist_bar(
                [float(v) for v in param_sampled[param]], lo, hi, use_log, n_bins
            )
            stable_bar = _hist_bar(
                [float(v) for v in param_stable.get(param, [])], lo, hi, use_log, n_bins
            )
            good_bar = _hist_bar(
                [float(v) for v in param_good.get(param, [])], lo, hi, use_log, n_bins
            )
            hist_table.add_row(
                _strip_model_prefix(param),
                _range_str(dist),
                sampled_bar,
                stable_bar,
                good_bar,
            )

        rprint()
        rprint(hist_table)

    # --- Categorical counts table ---
    cat_params = [
        p
        for p in sorted(param_sampled.keys())
        if dists.get(p) is None or isinstance(dists.get(p), od.CategoricalDistribution)
    ]
    if cat_params:
        cat_table = Table(
            title="Categorical Parameters", show_header=True, header_style="bold"
        )
        cat_table.add_column("Model Parameter", style="cyan", no_wrap=True)
        cat_table.add_column("Category", no_wrap=True)
        cat_table.add_column(
            f"Sampled (n={len(coverage_trials)})", justify="right", no_wrap=True
        )
        cat_table.add_column(
            f"Completed (n={len(completed)})",
            style="yellow",
            justify="right",
            no_wrap=True,
        )
        cat_table.add_column(
            f"{label_dir.capitalize()} {pct:.4g}% (n={n_good})",
            style="red",
            justify="right",
            no_wrap=True,
        )

        for param in cat_params:
            dist = dists.get(param)
            choices = (
                [str(c) for c in dist.choices]
                if dist is not None
                else sorted(set(str(v) for v in param_sampled[param]))
            )
            sampled_counts: dict[str, int] = {c: 0 for c in choices}
            stable_counts: dict[str, int] = {c: 0 for c in choices}
            good_counts: dict[str, int] = {c: 0 for c in choices}
            for v in param_sampled[param]:
                sampled_counts[str(v)] = sampled_counts.get(str(v), 0) + 1
            for v in param_stable.get(param, []):
                stable_counts[str(v)] = stable_counts.get(str(v), 0) + 1
            for v in param_good.get(param, []):
                good_counts[str(v)] = good_counts.get(str(v), 0) + 1

            for i, choice in enumerate(choices):
                param_label = _strip_model_prefix(param) if i == 0 else ""
                cat_table.add_row(
                    param_label,
                    choice,
                    str(sampled_counts.get(choice, 0)),
                    str(stable_counts.get(choice, 0)),
                    str(good_counts.get(choice, 0)),
                )

        rprint()
        rprint(cat_table)


_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"


def _render_grid(
    px: str,
    py: str,
    dists: dict[str, Any],
    coverage_trials: list[Any],
    completed: list[Any],
    good: set[int],
    grid_w: int,
    grid_h: int,
    converged_trials: list[Any] | None = None,
) -> tuple[list[str], int]:
    """Render a pairwise coverage grid. Returns (lines, visible_width).

    Lines may contain ANSI color codes; visible_width is the display width
    for correct side-by-side padding.

    Encoding:
      space       — not sampled
      · (dim)     — sampled but all diverged (no performance signal)
      ░▒▓█ (grey) — stable fraction (early-stopped only, no completed)
      ░▒▓█ yellow — has completed runs, none in top pct%
      ░▒▓█ red    — at least one in top pct%
    Opacity (char density) always reflects fraction of stable (converged) trials.
    """
    _converged = converged_trials if converged_trials is not None else completed

    dx, dy = dists[px], dists[py]
    lox, hix = float(dx.low), float(dx.high)
    loy, hiy = float(dy.low), float(dy.high)
    log_x = bool(getattr(dx, "log", False))
    log_y = bool(getattr(dy, "log", False))

    def bin_val(v: float, lo: float, hi: float, use_log: bool, n: int) -> int:
        if use_log and lo > 0:
            frac = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            frac = (v - lo) / (hi - lo) if hi > lo else 0.5
        return min(n - 1, max(0, int(frac * n)))

    sampled_grid = [[0] * grid_w for _ in range(grid_h)]
    converged_grid = [[0] * grid_w for _ in range(grid_h)]
    completed_grid = [[0] * grid_w for _ in range(grid_h)]
    good_grid = [[0] * grid_w for _ in range(grid_h)]

    for t in coverage_trials:
        if px not in t.params or py not in t.params:
            continue
        col = bin_val(float(t.params[px]), lox, hix, log_x, grid_w)
        row = bin_val(float(t.params[py]), loy, hiy, log_y, grid_h)
        sampled_grid[row][col] += 1

    for t in _converged:
        if px not in t.params or py not in t.params:
            continue
        col = bin_val(float(t.params[px]), lox, hix, log_x, grid_w)
        row = bin_val(float(t.params[py]), loy, hiy, log_y, grid_h)
        converged_grid[row][col] += 1

    for t in completed:
        if px not in t.params or py not in t.params:
            continue
        col = bin_val(float(t.params[px]), lox, hix, log_x, grid_w)
        row = bin_val(float(t.params[py]), loy, hiy, log_y, grid_h)
        completed_grid[row][col] += 1
        if t.number in good:
            good_grid[row][col] += 1

    lbl = 6  # y-axis label width

    def _abbrev(name: str, max_len: int = 18) -> str:
        parts = name.split(".")
        s = parts[-1]
        return s[:max_len] if len(s) > max_len else s

    visible_w = 2 + lbl + 2 + grid_w  # indent + label + "│ " + grid

    def _pad(s: str, visible_len: int) -> str:
        """Pad s so its visible width equals visible_w."""
        return s + " " * max(0, visible_w - visible_len)

    title = f"{_abbrev(px)} × {_abbrev(py)}"
    lines: list[str] = []
    lines.append(_pad(f"  {title:<{visible_w - 2}}", visible_w))
    lines.append(_pad(f"  {_fmt_val(hiy):>{lbl}s} ╮", 2 + lbl + 2))
    _DENSITY_CHARS = "░▒▓█"
    for row in range(grid_h - 1, -1, -1):
        line = ""
        for col in range(grid_w):
            n_sampled = sampled_grid[row][col]
            n_converged = converged_grid[row][col]
            n_completed = completed_grid[row][col]
            n_good = good_grid[row][col]
            if n_sampled == 0:
                line += " "
            elif n_converged == 0:
                line += f"{_ANSI_DIM}·{_ANSI_RESET}"
            else:
                conv_frac = n_converged / n_sampled
                char = _DENSITY_CHARS[min(3, int(conv_frac * 4))]
                if n_completed == 0:
                    # only early-stopped — greyscale, no color
                    line += char
                elif n_good > 0:
                    # has top performers — red
                    line += f"{_ANSI_RED}{char}{_ANSI_RESET}"
                else:
                    # completed but not top pct% — yellow
                    line += f"{_ANSI_YELLOW}{char}{_ANSI_RESET}"
        lines.append(f"  {'':>{lbl}s} │{line}")  # already visible_w
    lines.append(_pad(f"  {_fmt_val(loy):>{lbl}s} ╯", 2 + lbl + 2))
    lines.append(f"  {'':>{lbl}s}  {'':─<{grid_w}}")  # already visible_w
    lines.append(
        f"  {'':>{lbl}s}  {_fmt_val(lox)!s:<{grid_w // 2}}{_fmt_val(hix):>{grid_w // 2}}"
    )  # already visible_w
    return lines, visible_w


def print_param_pair_coverage(
    completed: list[Any],
    direction: str,
    pct: float = 10.0,
    n_params: int = 4,
    grid_w: int = 18,
    grid_h: int = 8,
    all_trials: list[Any] | None = None,
    converged_trials: list[Any] | None = None,
) -> None:
    """Print pairwise coverage grids for the top n_params most structured params.

    4 params → C(4,2)=6 pairs printed in a 3-column layout.
    Coverage includes all_trials (completed + pruned) if provided;
    good fraction is from completed only.
    """
    import optuna.distributions as od

    coverage_trials = all_trials if all_trials is not None else completed

    if len(coverage_trials) < 4:
        return

    good = _good_set(completed, direction, pct)
    dists = _param_dists(coverage_trials)

    numeric_params = [
        p
        for p, d in dists.items()
        if isinstance(d, (od.FloatDistribution, od.IntDistribution))
    ]
    if len(numeric_params) < 2:
        return

    # Select params by variance of mean objective across bins — picks params
    # where the objective landscape has interesting structure.
    param_scores: dict[str, float] = {}
    n_bins = 8
    for param in numeric_params:
        pairs = [
            (float(t.params[param]), t.value)
            for t in completed
            if param in t.params and t.value is not None
        ]
        if len(pairs) < 4:
            continue
        dist = dists[param]
        lo, hi = float(dist.low), float(dist.high)
        use_log = bool(getattr(dist, "log", False))
        bin_sums = [0.0] * n_bins
        bin_counts = [0] * n_bins
        for v, obj in pairs:
            frac = (
                (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
                if (use_log and lo > 0)
                else (v - lo) / (hi - lo)
                if hi > lo
                else 0.5
            )
            idx = min(n_bins - 1, max(0, int(frac * n_bins)))
            bin_sums[idx] += obj
            bin_counts[idx] += 1
        means = [
            bin_sums[i] / bin_counts[i] for i in range(n_bins) if bin_counts[i] > 0
        ]
        if len(means) >= 2:
            mu = sum(means) / len(means)
            param_scores[param] = sum((m - mu) ** 2 for m in means) / len(means)

    if not param_scores:
        return

    ranked = sorted(param_scores, key=lambda p: param_scores[p], reverse=True)
    top_params = ranked[:n_params]

    if len(top_params) < 2:
        return

    label_dir = "top" if direction == "MAXIMIZE" else "bottom"
    rprint(
        f"\n[bold]Pairwise coverage[/bold] — top {len(top_params)} params by objective variance across completed trials"
    )
    print(
        f"  {_ANSI_DIM}space=unsampled  ·=all diverged  ░▒▓█=stable fraction  "
        + f"{_ANSI_RESET}{_ANSI_YELLOW}█{_ANSI_RESET}{_ANSI_DIM}=completed  "
        + f"{_ANSI_RESET}{_ANSI_RED}█{_ANSI_RESET}{_ANSI_DIM}={label_dir} {pct:.4g}%{_ANSI_RESET}"
    )

    pairs = [(px, py) for i, px in enumerate(top_params) for py in top_params[i + 1 :]]

    for i in range(0, len(pairs), 3):
        row_pairs = pairs[i : i + 3]
        rendered = [
            _render_grid(
                px,
                py,
                dists,
                coverage_trials,
                completed,
                good,
                grid_w,
                grid_h,
                converged_trials=converged_trials,
            )
            for px, py in row_pairs
        ]
        gap = 2
        visible_w = rendered[0][1]
        empty_col = " " * (visible_w + gap)
        n_lines = max(len(lines) for lines, _ in rendered)
        rprint("")
        for line_i in range(n_lines):
            row = ""
            for lines, _ in rendered:
                # Each line is pre-padded to visible_w; add gap for column spacing
                cell = lines[line_i] if line_i < len(lines) else empty_col
                row += cell + " " * gap
            print(row.rstrip())
