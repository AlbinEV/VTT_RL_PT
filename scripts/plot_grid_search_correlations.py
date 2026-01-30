#!/usr/bin/env python3
"""
Plot correlation analysis for grid search results.

Usage:
    python scripts/plot_grid_search_correlations.py /path/to/results.json
    python scripts/plot_grid_search_correlations.py /path/to/grid_search_dir
"""
import _path_setup

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PARAM_KEYS = [
    "w_force_tracking",
    "force_error_scale",
    "w_adaptive_kpz",
    "optimal_kpz_center",
    "w_kpz_change_penalty",
    "w_contact_bonus",
]

METRIC_KEYS = [
    "force_err_mean",
    "fz_osc_rms",
    "score",
    "contact_steps",
    "fz_mean",
    "fz_std",
]


def _is_finite(value):
    return value is not None and not (isinstance(value, float) and math.isinf(value)) and np.isfinite(value)


def load_results(path: Path):
    if path.is_dir():
        path = path / "results.json"
    with path.open() as f:
        data = json.load(f)
    return data, path


def build_rows(results):
    rows = []
    for run in results:
        if not run.get("success", False):
            continue
        row = {}
        params = run.get("params", {})
        for key in PARAM_KEYS:
            row[key] = params.get(key)
        for key in METRIC_KEYS:
            row[key] = run.get(key)
        rows.append(row)
    return rows


def as_array(rows, keys):
    arr = []
    for row in rows:
        arr.append([row.get(k) for k in keys])
    return np.array(arr, dtype=float)


def corr_matrix(data, keys):
    n = len(keys)
    matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            a = data[:, i]
            b = data[:, j]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                continue
            matrix[i, j] = np.corrcoef(a[mask], b[mask])[0, 1]
    return matrix


def plot_heatmap(matrix, keys, output_file):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_title("Correlation Heatmap (Params + Metrics)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_param_metric_scatter(rows, output_file):
    params = PARAM_KEYS
    metrics = ["force_err_mean", "fz_osc_rms", "score"]

    fig, axes = plt.subplots(len(metrics), len(params), figsize=(18, 9), sharey="row")
    for i, metric in enumerate(metrics):
        for j, param in enumerate(params):
            ax = axes[i, j]
            x = np.array([r.get(param) for r in rows], dtype=float)
            y = np.array([r.get(metric) for r in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=12, alpha=0.6)
            if i == 0:
                ax.set_title(param)
            if j == 0:
                ax.set_ylabel(metric)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def print_top_correlations(matrix, keys, top_k=8):
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            val = matrix[i, j]
            if np.isfinite(val):
                pairs.append((abs(val), val, keys[i], keys[j]))
    pairs.sort(reverse=True)
    print("\nTop correlations (absolute value):")
    for rank, (_, val, a, b) in enumerate(pairs[:top_k], 1):
        print(f"{rank:>2}. {a} vs {b}: {val:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot grid search correlations")
    parser.add_argument("path", type=str, help="Path to results.json or grid_search directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    results, results_path = load_results(Path(args.path))
    rows = build_rows(results)
    if not rows:
        print("No successful runs found.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = PARAM_KEYS + METRIC_KEYS
    data = as_array(rows, keys)
    matrix = corr_matrix(data, keys)

    heatmap_file = output_dir / "correlation_heatmap.png"
    scatter_file = output_dir / "param_metric_scatter.png"

    plot_heatmap(matrix, keys, heatmap_file)
    plot_param_metric_scatter(rows, scatter_file)
    print(f"Saved heatmap to {heatmap_file}")
    print(f"Saved scatter grid to {scatter_file}")
    print_top_correlations(matrix, keys)


if __name__ == "__main__":
    main()
