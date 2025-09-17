#!/usr/bin/env python3
"""
Paper plots: comparative Δdex error distributions across datasets.

This script loads the per-dataset error dictionaries saved by the benchmark
(compare_errors -> scripts/pp/<dataset>/all_log_errors.npz), and creates a 2x2 grid
with one subplot per dataset, each showing the same comparative plot as
plot_error_distribution_comparative(..., mode="deltadex").

Usage:
    python paper_plots.py --root scripts/pp \
        [--output plots/paper/error_dist_deltadex_by_dataset.png] \
        [--cols 2]

Notes:
- Each dataset directory must contain an NPZ file named 'all_log_errors.npz'.
  For robustness, we also try 'all_errors_log.npz' as a fallback.
- The NPZ file typically contains a single object array 'arr_0' which is a
  dict mapping surrogate_name -> numpy array of Δdex errors with shape [N, T, Q].
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Reuse palette from project for consistent styling, but keep a safe fallback
try:
    from codes.benchmark.bench_plots import get_custom_palette
except Exception:  # pragma: no cover - fallback if import fails

    def get_custom_palette(n: int):
        return plt.cm.viridis(np.linspace(0, 0.95, n))


def _format_dataset_title(name: str) -> str:
    """Map dataset folder names to display titles with proper capitalization."""
    mapping = {
        "primordial": "Primordial",
        "primordial_parametric": "Primordial Parametric",
    }
    if name in mapping:
        return mapping[name]
    # Fallback: replace underscores with space and title-case
    return name.replace("_", " ").title()


def _load_errors_npz(path_npz: str) -> Dict[str, np.ndarray]:
    """
    Load a dict[str, np.ndarray] from an NPZ file created by np.savez.

    Supports the common patterns:
    - arr_0 (object array) holding a Python dict
    - Named arrays per surrogate (if saved with kwargs)
    - A single key 'log_errors' holding the dict
    """
    if not os.path.exists(path_npz):
        raise FileNotFoundError(path_npz)

    data = np.load(path_npz, allow_pickle=True)
    try:
        # Preferred: saved as a single object array containing the dict
        if "arr_0" in data.files and isinstance(data["arr_0"], np.ndarray):
            obj = data["arr_0"]
            # Could be 0-d object array with dict
            if obj.dtype == object:
                d = obj.item()
                if isinstance(d, dict):
                    return d
        # Alternative: explicit key name
        if "log_errors" in data.files:
            d = data["log_errors"].item()
            if isinstance(d, dict):
                return d
        # Fallback: construct dict from per-surrogate arrays
        out: Dict[str, np.ndarray] = {}
        for k in data.files:
            arr = data[k]
            # Only accept ND arrays
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                out[k] = arr
        if out:
            return out
    finally:
        data.close()

    raise ValueError(f"Could not interpret NPZ structure in {path_npz}")


def load_dataset_errors(root: str, dataset: str) -> Dict[str, np.ndarray]:
    """Try both file names and return the errors dict for a dataset."""
    cand1 = os.path.join(root, dataset, "all_log_errors.npz")
    cand2 = os.path.join(root, dataset, "all_errors_log.npz")  # user-mentioned alt
    last_err: Exception | None = None
    for p in (cand1, cand2):
        try:
            return _load_errors_npz(p)
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError(
        f"No error file found for dataset '{dataset}'. Tried: {cand1}, {cand2}. Last error: {last_err}"
    )


def compute_global_range(
    datasets_errors: Dict[str, Dict[str, np.ndarray]],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> Tuple[float, float]:
    """
    Compute global x-range in log10 space across all datasets and surrogates,
    following the same logic as plot_error_distribution_comparative.
    """
    log_arrays: List[np.ndarray] = []
    for ds, err_dict in datasets_errors.items():
        for _, arr in err_dict.items():
            flat = arr.astype(float).ravel()
            # Filter finite and strictly positive (avoid log10(0) and NaN)
            mask = np.isfinite(flat) & (flat > 0)
            if not np.any(mask):
                continue
            log_arrays.append(np.log10(flat[mask]))
    if not log_arrays:
        # Default safe range if everything is empty
        return -8.0, 0.0

    mins = [np.percentile(x, low_pct) for x in log_arrays if x.size > 0]
    maxs = [np.percentile(x, high_pct) for x in log_arrays if x.size > 0]
    global_min = float(np.min(mins))
    global_max = float(np.max(maxs))
    # Expand to nice boundaries
    x_min = float(np.floor(global_min))
    x_max = float(np.ceil(global_max))
    return x_min, x_max


def build_color_map(datasets_errors: Dict[str, Dict[str, np.ndarray]]):
    """Build a consistent surrogate->color map across all datasets using viridis.

    Ensures deterministic ordering by sorting surrogate names alphabetically.
    Returns a dict preserving this order and the ordered list of names.
    """
    name_set = set()
    for err_dict in datasets_errors.values():
        name_set.update(list(err_dict.keys()))
    names = sorted(name_set)
    # Permute to specific legend ordering
    if len(names) == 4:
        names = [names[i] for i in [3, 0, 2, 1]]
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(names)))
    # Dict preserves insertion order, matching `names` sequence
    color_map = {name: colors[i] for i, name in enumerate(names)}
    return color_map, names


def reorder_legend_entries_rowwise(
    handles: List, labels: List[str], max_ncols: int
) -> Tuple[List, List[str], int]:
    """
    Reorder legend entries so they display row-wise when Matplotlib fills columns first.

    Provide `handles`/`labels` in the desired row-wise reading order; this
    function returns a permutation that, when passed to Matplotlib legend with
    ncol=legend_ncol, yields that row-wise order.

    Returns (final_handles, final_labels, legend_ncol).
    """
    N = len(handles)
    if N == 0:
        return [], [], 1

    legend_ncol = max(1, min(max_ncols, N))
    rows = int(np.ceil(N / legend_ncol))

    final_handles: List = []
    final_labels: List[str] = []
    # Convert a row-wise ordered list to the column-first order expected by Matplotlib
    # Example (N=6, ncol=2): input [a,b,c,d,e,f] -> output [a,c,e,b,d,f]
    for c in range(legend_ncol):
        for r in range(rows):
            idx = r * legend_ncol + c
            if idx < N:
                final_handles.append(handles[idx])
                final_labels.append(labels[idx])

    return final_handles, final_labels, legend_ncol


def plot_grid_deltadex(
    datasets: List[str],
    datasets_errors: Dict[str, Dict[str, np.ndarray]],
    x_log_min: float,
    x_log_max: float,
    color_map: Dict[str, Tuple[float, float, float, float]],
    dpi: int = 300,
    n_cols: int = 2,
):
    """
    Render a 2x2 grid: one subplot per dataset, reproducing the comparative
    error distribution plot for Δdex with consistent axes and colors.
    """
    # Prepare x bin edges in log10 space and transform to linear for plotting
    x_vals = np.linspace(x_log_min, x_log_max + 0.1, 100)

    n = max(1, len(datasets))
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        sharex=True,  # sharey=True
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        err_dict = datasets_errors.get(dataset, {})
        if not err_dict:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha="center", va="center")
            ax.set_axis_off()
            continue

        # For legend ordering, use color_map order
        for model_name, color in color_map.items():
            if model_name not in err_dict:
                continue
            arr = err_dict[model_name]
            flat = arr.astype(float).ravel()
            mask = np.isfinite(flat) & (flat > 0)
            if not np.any(mask):
                continue
            vals = flat[mask]
            logs = np.log10(vals)

            hist, bin_edges = np.histogram(logs, bins=x_vals, density=True)
            smoothed = gaussian_filter1d(hist, sigma=2)

            ax.plot(10 ** bin_edges[:-1], smoothed, label=model_name, color=color)

            # Mean and median markers (on linear scale)
            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            ax.axvline(
                x=mean_val, color=color, linestyle="--", linewidth=1.0, alpha=0.9
            )
            ax.axvline(
                x=median_val, color=color, linestyle="-.", linewidth=1.0, alpha=0.9
            )

        ax.set_xscale("log")
        ax.set_xlim(left=1e-4, right=10)
        # Y label only on first column
        if (idx % n_cols) == 0:
            ax.set_ylabel("Smoothed Histogram Count")
        ax.set_ylim(0, None)
        ax.set_title(_format_dataset_title(dataset))

    # Common X label and legend
    for ax in axes[-n_cols:]:
        ax.set_xlabel(r"Log-MAE ($\Delta dex$)")

    # Build a single legend using first axis handles for present models
    handles, labels = [], []
    for model_name, color in color_map.items():
        # Proxy lines for legend
        line = plt.Line2D([0], [0], color=color, label=model_name)
        handles.append(line)
        labels.append(model_name)
    # Mean/median style proxies
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="Mean"))
    labels.append("Mean")
    handles.append(plt.Line2D([0], [0], color="black", linestyle="-.", label="Median"))
    labels.append("Median")

    final_handles, final_labels, legend_ncol = reorder_legend_entries_rowwise(
        handles, labels, max_ncols=2
    )

    # Place legend below plots, arranged row-wise
    fig.legend(
        final_handles,
        final_labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.05),
        fontsize="small",
        frameon=True,
        ncol=legend_ncol,
    )

    # No overall title; leave space at the bottom for legend
    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.98])

    out_path = "scripts/pp/error_dist_deltadex_by_dataset.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grid_deltadex_percentiles(
    datasets: List[str],
    datasets_errors: Dict[str, Dict[str, np.ndarray]],
    timesteps: np.ndarray,
    color_map: Dict[str, Tuple[float, float, float, float]],
    dpi: int = 300,
    n_cols: int = 2,
):
    """
    Create a grid of subplots (one per dataset) showing Δdex error percentiles over time.

    For each dataset:
      - Draw neutral grey one-sided percentile bands (50, 90, 99) aggregated across surrogates.
      - Overlay each surrogate's mean and median Δdex over time using the shared viridis color map.
    """
    # Prepare layout
    n = max(1, len(datasets))
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=False, sharey=True
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Legend proxies for mean and 99th percentile styles
    mean_proxy = plt.Line2D([0], [0], color="black", linestyle="-", label="Mean")
    p99_proxy = plt.Line2D(
        [0], [0], color="black", linestyle="--", label="99th Percentile"
    )

    surrogate_proxies = []
    surrogate_labels = []
    for name, color in color_map.items():
        surrogate_proxies.append(plt.Line2D([0], [0], color=color, label=name))
        surrogate_labels.append(name)

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        err_dict = datasets_errors.get(dataset, {})
        if not err_dict:
            ax.text(0.5, 0.5, f"No data for {dataset}", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Assume same T across surrogates within a dataset
        any_arr = next(iter(err_dict.values()))
        T = any_arr.shape[1]

        # Aggregate across surrogates for percentile bands
        pooled = []
        for arr in err_dict.values():
            if arr.shape[1] != T:
                continue
            pooled.append(arr)

        # Plot mean and 99th percentile per surrogate
        for model_name, color in color_map.items():
            if model_name not in err_dict:
                continue
            arr = err_dict[model_name]
            if arr.shape[1] != T:
                continue
            mean_ts = np.mean(arr, axis=(0, 2))
            p99_ts = np.percentile(arr, 99, axis=(0, 2))
            ax.plot(timesteps, mean_ts, color=color, linestyle="-", linewidth=1.2)
            ax.plot(timesteps, p99_ts, color=color, linestyle="--", linewidth=1.0)

        ax.set_xscale("log")
        ax.set_xlim(left=max(1, timesteps[0]), right=timesteps[-1])
        # Y label only on first column
        if (idx % n_cols) == 0:
            ax.set_ylabel(r"$\Delta dex$")
        ax.set_ylim(2 * 1e-2, 20)
        ax.set_title(_format_dataset_title(dataset))
        ax.set_yscale("log")
        ax.grid(False)

    # Label bottom row x-axis
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Time (y)")

    # Turn off tick labels on upper rows if multiple rows
    if n_rows > 1:
        for ax in axes[:-n_cols]:
            ax.set_xticklabels([])

    # Build combined legend below plots: surrogates + mean/99th style proxies
    handles = surrogate_proxies + [mean_proxy, p99_proxy]
    labels = surrogate_labels + ["Mean", "99th Percentile"]

    handles, labels, legend_ncol = reorder_legend_entries_rowwise(handles, labels, 2)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.04),
        fontsize="small",
        frameon=True,
        ncol=legend_ncol,
    )

    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.98])

    out_path = "scripts/pp/error_percentiles_deltadex_by_dataset.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_catastrophic_recall(
    root: str, dataset: str, percentile: int
) -> np.ndarray | None:
    """Load catastrophic recall matrix saved as npz for a dataset.

    Expected path: scripts/pp/<dataset>/catastrophic_recall_<percentile>.npz
    Returns array of shape [S, F] (surrogates x flag fractions), or None if missing.
    """
    path = os.path.join(root, dataset, f"catastrophic_recall_{percentile}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    try:
        if "arr_0" in data.files:
            arr = data["arr_0"]
        else:
            # Fallback to the first entry
            arr = data[data.files[0]]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr
    finally:
        data.close()
    return None


def plot_grid_catastrophic_detection(
    datasets: List[str],
    datasets_errors: Dict[str, Dict[str, np.ndarray]],
    color_map: Dict[str, Tuple[float, float, float, float]],
    root: str,
    recall_percentile: int = 99,
    flag_fractions: Tuple[float, ...] = (
        0.0,
        0.025,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ),
    dpi: int = 300,
    n_cols: int = 2,
):
    """Create a grid of catastrophic error detection curves across datasets.

    Plots recall (%) vs fraction flagged (%) for each surrogate using precomputed
    recall matrices (90 or 99). Uses consistent color mapping and a single legend below.
    """
    # Layout
    n = max(1, len(datasets))
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=False, sharey=True
    )
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Precompute legend proxies for surrogates in desired order
    leg_handles: List = []
    leg_labels: List[str] = []
    for name, color in color_map.items():
        leg_handles.append(plt.Line2D([0], [0], color=color, marker="o", label=name))
        leg_labels.append(name)

    for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        err_dict = datasets_errors.get(dataset, {})
        recall_mat = _load_catastrophic_recall(root, dataset, recall_percentile)
        if not err_dict or recall_mat is None:
            ax.text(0.5, 0.5, f"No recall data for {dataset}", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Surrogate names order used when recall was saved
        saved_names = list(err_dict.keys())
        F = recall_mat.shape[1]
        # Build X values. Prefer the canonical fractions if lengths match; fallback to linear spacing
        if F == len(flag_fractions):
            xs = np.array(flag_fractions, dtype=float) * 100.0
        else:
            xs = np.linspace(0.0, 100.0 * max(flag_fractions), F)

        # Plot curves in our global color order, mapping into the saved row index
        for name, color in color_map.items():
            if name not in saved_names:
                continue
            row = saved_names.index(name)
            ys = recall_mat[row, : len(xs)] * 100.0
            ax.plot(xs, ys, marker="o", color=color, linewidth=1.2)

        ax.set_ylabel("Catastrophic error recall (%)" if (idx % n_cols) == 0 else "")
        ax.set_xlim(0, float(xs.max()))
        ax.set_ylim(0, 100)
        if (idx % n_cols) == 0:
            ax.set_ylabel("Catastrophic error recall (%)")
        ax.grid(True, alpha=0.3)
        ax.set_title(_format_dataset_title(dataset))

    # Label bottom row x-axis
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Flagged fraction (%)")

    # Turn off tick labels on upper rows if multiple rows
    if n_rows > 1:
        for ax in axes[:-n_cols]:
            ax.set_xticklabels([])

    # Figure-level legend
    handles, labels, legend_ncol = reorder_legend_entries_rowwise(
        leg_handles, leg_labels, 2
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.05),
        fontsize="small",
        frameon=True,
        ncol=legend_ncol,
    )

    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.98])

    out_path = f"scripts/pp/catastrophic_detection_{recall_percentile}_by_dataset.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="2x2 comparative Δdex error distributions across datasets"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="scripts/pp",
        help="Root directory containing per-dataset subdirectories",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of columns in the subplot grid",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")

    args = parser.parse_args()

    # Auto-discover datasets: list subdirectories under root that contain an NPZ
    if not os.path.isdir(args.root):
        raise NotADirectoryError(args.root)

    all_subdirs = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )
    datasets: List[str] = []
    datasets_errors: Dict[str, Dict[str, np.ndarray]] = {}
    for ds in all_subdirs:
        # Only include if an NPZ exists
        npz1 = os.path.join(args.root, ds, "all_log_errors.npz")
        npz2 = os.path.join(args.root, ds, "all_errors_log.npz")
        if not (os.path.exists(npz1) or os.path.exists(npz2)):
            continue
        try:
            datasets_errors[ds] = load_dataset_errors(args.root, ds)
            datasets.append(ds)
        except Exception:
            # Skip subdirs with unreadable NPZs
            continue
    timesteps_path = os.path.join(args.root, "timesteps.npz")
    timesteps = np.load(timesteps_path)["arr_0"]

    if not datasets:
        raise RuntimeError(
            f"No datasets with error NPZs found under root '{args.root}'."
        )

    # Global x-range and consistent colors across all surrogates
    x_min, x_max = compute_global_range(datasets_errors)
    color_map, _ = build_color_map(datasets_errors)

    plot_grid_deltadex(
        datasets=datasets,
        datasets_errors=datasets_errors,
        x_log_min=x_min,
        x_log_max=x_max,
        color_map=color_map,
        dpi=args.dpi,
        n_cols=args.cols,
    )

    # Percentiles-over-time grid (deltadex mode)
    plot_grid_deltadex_percentiles(
        datasets=datasets,
        datasets_errors=datasets_errors,
        timesteps=timesteps,
        color_map=color_map,
        dpi=args.dpi,
        n_cols=args.cols,
    )

    # Catastrophic detection grid
    plot_grid_catastrophic_detection(
        datasets=datasets,
        datasets_errors=datasets_errors,
        color_map=color_map,
        root=args.root,
        recall_percentile=99,
        dpi=args.dpi,
        n_cols=args.cols,
    )

    plot_grid_catastrophic_detection(
        datasets=datasets,
        datasets_errors=datasets_errors,
        color_map=color_map,
        root=args.root,
        recall_percentile=90,
        dpi=args.dpi,
        n_cols=args.cols,
    )


if __name__ == "__main__":
    main()
