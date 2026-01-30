#!/usr/bin/env python3
"""
Plot OSC optimization results from trials data (CSV/JSON/log).

Outputs Plotly HTML files:
  - kp_z vs zeta_z colored by score
  - score by trial (with best-so-far)
  - parameter vs score scatter
  - distributions for kp_z, zeta_z, score
"""
import _path_setup

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


OPTIONAL_FLOAT_FIELDS = [
    "fz_error_mean",
    "fz_error_std",
    "fz_mean",
    "fz_min",
    "fz_max",
    "fz_stability",
    "completion_rate",
    "avg_max_wpt",
    "avg_reward",
    "avg_steps",
    "avg_contact_steps",
    "trial_time_s",
    "total_time_s",
]


def load_trials_csv(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                trial = int(raw["trial"])
                kp_z = float(raw["kp_z"])
                zeta_z = float(raw["zeta_z"])
                score = float(raw["score"])
            except Exception:
                continue

            row = {
                "trial": trial,
                "kp_z": kp_z,
                "zeta_z": zeta_z,
                "score": score,
            }

            for key in OPTIONAL_FLOAT_FIELDS:
                if key in raw and raw[key] != "":
                    try:
                        row[key] = float(raw[key])
                    except Exception:
                        row[key] = None

            rows.append(row)

    rows.sort(key=lambda r: r["trial"])
    return rows


def load_trials_json(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text())
    results = data.get("results", [])
    rows = []
    for idx, raw in enumerate(results, start=1):
        try:
            kp_z = float(raw["kp_z"])
            zeta_z = float(raw["zeta_z"])
            score = float(raw["score"])
        except Exception:
            continue
        row = {
            "trial": idx,
            "kp_z": kp_z,
            "zeta_z": zeta_z,
            "score": score,
        }
        if "force_error_mean" in raw:
            row["fz_error_mean"] = raw.get("force_error_mean")
        if "completion_rate" in raw:
            row["completion_rate"] = raw.get("completion_rate")
        rows.append(row)
    rows.sort(key=lambda r: r["trial"])
    return rows


def load_trials_log(log_path: Path) -> list[dict]:
    rows = []
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\d+)\s+(\d+)\s*%"
    )
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        trial = int(match.group(1))
        kp_z = float(match.group(2))
        zeta_z = float(match.group(3))
        score = float(match.group(4))
        fz_error = float(match.group(5))
        fz_mean = float(match.group(6))
        done_pct = float(match.group(8))
        rows.append(
            {
                "trial": trial,
                "kp_z": kp_z,
                "zeta_z": zeta_z,
                "score": score,
                "fz_error_mean": fz_error,
                "fz_mean": fz_mean,
                "completion_rate": done_pct / 100.0,
            }
        )
    rows.sort(key=lambda r: r["trial"])
    return rows


def load_trials(input_path: Path) -> list[dict]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return load_trials_csv(input_path)
    if suffix == ".json":
        return load_trials_json(input_path)
    if suffix == ".log":
        return load_trials_log(input_path)
    raise SystemExit(f"Unsupported input type: {input_path}")


def best_so_far(scores: list[float]) -> list[float]:
    best_vals = []
    current = float("inf")
    for score in scores:
        if score < current:
            current = score
        best_vals.append(current)
    return best_vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OSC optimization results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(_path_setup.DATA_ROOT / "osc_optimization" / "trials.csv"),
        help="Path to trials CSV/JSON/log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_path_setup.DATA_ROOT / "results_phd" / "osc_optimization" / "plots"),
        help="Directory for HTML plots",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Prefix for output filenames (default: input stem)",
    )
    args = parser.parse_args()

    rows = load_trials(args.input)
    if not rows:
        raise SystemExit(f"No valid rows found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label.strip() or args.input.stem
    label = label.replace(" ", "_")

    trials = [r["trial"] for r in rows]
    kp_z_vals = [r["kp_z"] for r in rows]
    zeta_vals = [r["zeta_z"] for r in rows]
    scores = [r["score"] for r in rows]

    # Best run
    best_idx = min(range(len(scores)), key=lambda i: scores[i])
    best_row = rows[best_idx]

    # 1) Kp_z vs zeta_z colored by score
    fig = go.Figure(
        data=go.Scatter(
            x=kp_z_vals,
            y=zeta_vals,
            mode="markers",
            marker=dict(
                size=9,
                color=scores,
                colorscale="Viridis",
                colorbar=dict(title="score"),
            ),
            text=[f"trial={r['trial']}" for r in rows],
            hovertemplate="kp_z=%{x:.1f}<br>zeta_z=%{y:.3f}<br>score=%{marker.color:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_row["kp_z"]],
            y=[best_row["zeta_z"]],
            mode="markers",
            marker=dict(size=14, color="#d62728", symbol="star"),
            name="best",
            hovertemplate="BEST<br>kp_z=%{x:.1f}<br>zeta_z=%{y:.3f}<br>score=%{text}<extra></extra>",
            text=[f"{best_row['score']:.3f}"],
        )
    )
    fig.update_layout(
        title="OSC Random Search: kp_z vs zeta_z (colored by score)",
        xaxis_title="kp_z",
        yaxis_title="zeta_z",
    )
    fig.write_html(args.output_dir / f"{label}_kpz_vs_zeta.html")

    # 2) Score by trial with best-so-far
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trials,
            y=scores,
            mode="markers",
            name="score",
            marker=dict(size=7, color="#1f77b4"),
            hovertemplate="trial=%{x}<br>score=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trials,
            y=best_so_far(scores),
            mode="lines",
            name="best_so_far",
            line=dict(color="#d62728", width=2),
        )
    )
    fig.update_layout(
        title="OSC Random Search: score by trial",
        xaxis_title="trial",
        yaxis_title="score (lower is better)",
    )
    fig.write_html(args.output_dir / f"{label}_score_by_trial.html")

    # 3) Parameter vs score scatter
    fig = make_subplots(rows=1, cols=2, subplot_titles=("kp_z vs score", "zeta_z vs score"))
    fig.add_trace(
        go.Scatter(
            x=kp_z_vals,
            y=scores,
            mode="markers",
            marker=dict(size=7, color=scores, colorscale="Viridis", showscale=False),
            hovertemplate="kp_z=%{x:.1f}<br>score=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=zeta_vals,
            y=scores,
            mode="markers",
            marker=dict(size=7, color=scores, colorscale="Viridis", showscale=False),
            hovertemplate="zeta_z=%{x:.3f}<br>score=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="kp_z", row=1, col=1)
    fig.update_xaxes(title_text="zeta_z", row=1, col=2)
    fig.update_yaxes(title_text="score", row=1, col=1)
    fig.update_yaxes(title_text="score", row=1, col=2)
    fig.update_layout(title="OSC Random Search: parameter vs score")
    fig.write_html(args.output_dir / f"{label}_param_vs_score.html")

    # 4) Distributions
    fig = make_subplots(rows=1, cols=3, subplot_titles=("kp_z", "zeta_z", "score"))
    fig.add_trace(go.Histogram(x=kp_z_vals, nbinsx=20, name="kp_z"), row=1, col=1)
    fig.add_trace(go.Histogram(x=zeta_vals, nbinsx=20, name="zeta_z"), row=1, col=2)
    fig.add_trace(go.Histogram(x=scores, nbinsx=20, name="score"), row=1, col=3)
    fig.update_layout(title="OSC Random Search: distributions", showlegend=False)
    fig.write_html(args.output_dir / f"{label}_distributions.html")


if __name__ == "__main__":
    main()
