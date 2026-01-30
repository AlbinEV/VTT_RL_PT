#!/usr/bin/env python3
"""Plot force trends and summarize friction sweep results."""
import _path_setup

import argparse
import csv
import glob
import json
import os
from datetime import datetime

import numpy as np

# matplotlib import can be slow; defer until needed

F_TOUCH = -2.0
WINDOW = 20
EPS = 1e-6
CONTACT_WPT_IDX = 3  # Waypoint index for contact (0-based)
LATERAL0_WPT_IDX = 4  # Waypoint index for lateral 0 (0-based)
LATERAL1_WPT_IDX = 5  # Waypoint index for lateral 1 (0-based)
LATERAL2_WPT_IDX = 6  # Waypoint index for lateral 2 (0-based)
SMOOTH_WINDOW = 9  # Moving average window for force_trends (odd recommended)
ELLIPSOID_STEPS = 16  # Max number of variability ellipsoids along the mean trajectory


def _safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def _safe_std(values):
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _smooth_series(series, window):
    if window <= 1 or series.size < window:
        return series
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(series, kernel, mode="same")


def _load_osc_pure(mu_dir):
    h5_files = sorted(glob.glob(os.path.join(mu_dir, "**", "*.h5"), recursive=True))
    episodes = []
    times = None
    contact_times = []
    lateral0_times = []
    lateral1_times = []
    lateral2_times = []
    for f in h5_files:
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError("h5py is required for OSC pure files") from e
        with h5py.File(f, "r") as h5:
            if "fz" not in h5:
                continue
            fz = np.array(h5["fz"], dtype=float)
            if "sim_time" in h5:
                t = np.array(h5["sim_time"], dtype=float)
            else:
                t = np.arange(len(fz), dtype=float) * 0.1
            wpt_idx = np.array(h5["wpt_idx"], dtype=int) if "wpt_idx" in h5 else None
        if fz.size == 0:
            continue
        episodes.append(fz)
        if times is None:
            times = t
        if wpt_idx is not None and wpt_idx.size:
            idx_contact = np.where(wpt_idx >= CONTACT_WPT_IDX)[0]
            if idx_contact.size:
                contact_times.append(float(t[int(idx_contact[0])]))
            idx_l0 = np.where(wpt_idx >= LATERAL0_WPT_IDX)[0]
            if idx_l0.size:
                lateral0_times.append(float(t[int(idx_l0[0])]))
            idx_l1 = np.where(wpt_idx >= LATERAL1_WPT_IDX)[0]
            if idx_l1.size:
                lateral1_times.append(float(t[int(idx_l1[0])]))
            idx_l2 = np.where(wpt_idx >= LATERAL2_WPT_IDX)[0]
            if idx_l2.size:
                lateral2_times.append(float(t[int(idx_l2[0])]))
    return episodes, times, contact_times, lateral0_times, lateral1_times, lateral2_times


def _load_kz(mu_dir):
    traj_files = sorted(glob.glob(os.path.join(mu_dir, "**", "trajectories.json"), recursive=True))
    if not traj_files:
        return [], None
    episodes = []
    times = None
    for path in traj_files:
        with open(path) as f:
            data = json.load(f)
        for ep in data.get("episodes", []):
            fz = np.array(ep.get("contact_forces", []), dtype=float)
            t = np.array(ep.get("timestamps", []), dtype=float)
            if fz.size == 0:
                continue
            episodes.append(fz)
            if times is None and t.size:
                times = t
    return episodes, times


def _contact_metrics(fz):
    if fz.size == 0:
        return None
    contact_idx = np.where(fz < F_TOUCH)[0]
    if contact_idx.size == 0:
        return None
    start = int(contact_idx[0])
    end = min(start + WINDOW, fz.size)
    window = fz[start:end]
    if window.size == 0:
        return None
    df = np.abs(np.diff(window))
    return {
        "contact_min_fz": float(window.min()),
        "contact_max_df": float(df.max() if df.size else 0.0),
        "contact_mean_df": float(df.mean() if df.size else 0.0),
        "contact_ratio": float((fz < F_TOUCH).mean()),
    }


def _summarize(episodes):
    if not episodes:
        return None
    min_len = min(len(ep) for ep in episodes)
    trimmed = [ep[:min_len] for ep in episodes]
    mean_series = np.mean(np.stack(trimmed, axis=0), axis=0)

    mean_fz_list = [float(np.mean(ep)) for ep in episodes]
    min_fz_list = [float(np.min(ep)) for ep in episodes]

    metrics = {
        "episodes": len(episodes),
        "steps": int(min_len),
        "mean_fz": _safe_mean(mean_fz_list),
        "mean_fz_std": _safe_std(mean_fz_list),
        "min_fz": _safe_mean(min_fz_list),
        "min_fz_std": _safe_std(min_fz_list),
    }

    contact_min = []
    contact_max_df = []
    contact_mean_df = []
    contact_ratio = []
    for ep in episodes:
        m = _contact_metrics(ep)
        if not m:
            continue
        contact_min.append(m["contact_min_fz"])
        contact_max_df.append(m["contact_max_df"])
        contact_mean_df.append(m["contact_mean_df"])
        contact_ratio.append(m["contact_ratio"])

    metrics.update({
        "contact_min_fz_mean": _safe_mean(contact_min),
        "contact_min_fz_std": _safe_std(contact_min),
        "contact_max_df_mean": _safe_mean(contact_max_df),
        "contact_max_df_std": _safe_std(contact_max_df),
        "contact_mean_df_mean": _safe_mean(contact_mean_df),
        "contact_mean_df_std": _safe_std(contact_mean_df),
        "contact_ratio_mean": _safe_mean(contact_ratio),
        "contact_ratio_std": _safe_std(contact_ratio),
    })

    return mean_series, metrics


def _collect_controller(base_dir, controller):
    controller_dir = os.path.join(base_dir, controller)
    if not os.path.isdir(controller_dir):
        return {}
    out = {}
    for mu_dir in sorted(glob.glob(os.path.join(controller_dir, "mu_*"))):
        mu = os.path.basename(mu_dir).replace("mu_", "")
        if controller == "osc_pure":
            episodes, times, contact_times, lateral0_times, lateral1_times, lateral2_times = _load_osc_pure(mu_dir)
        else:
            episodes, times = _load_kz(mu_dir)
            lateral2_times = None
        if not episodes:
            continue
        summary = _summarize(episodes)
        if not summary:
            continue
        mean_series, metrics = summary
        payload = {
            "mean_series": mean_series,
            "times": times,
            "metrics": metrics,
        }
        if controller == "osc_pure":
            if contact_times:
                payload["contact_time_mean"] = _safe_mean(contact_times)
                payload["contact_time_std"] = _safe_std(contact_times)
            if lateral0_times:
                payload["lateral0_time_mean"] = _safe_mean(lateral0_times)
                payload["lateral0_time_std"] = _safe_std(lateral0_times)
            if lateral1_times:
                payload["lateral1_time_mean"] = _safe_mean(lateral1_times)
                payload["lateral1_time_std"] = _safe_std(lateral1_times)
            if lateral2_times:
                payload["lateral2_time_mean"] = _safe_mean(lateral2_times)
                payload["lateral2_time_std"] = _safe_std(lateral2_times)
        out[mu] = payload
    return out


def _write_table(table_path, rows, fieldnames):
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(md_path, rows, fieldnames):
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            values = [str(row.get(k, "")) for k in fieldnames]
            f.write("| " + " | ".join(values) + " |\n")


def _make_recap(recap_path, controllers, rows):
    os.makedirs(os.path.dirname(recap_path), exist_ok=True)
    lines = []
    lines.append("Recap friction sweep")
    lines.append("====================")
    lines.append("")

    present = {c: sorted(controllers.get(c, {}).keys()) for c in ["osc_pure", "kz_only", "kz_dz"]}
    lines.append("Data availability:")
    lines.append(f"- osc_pure: {', '.join(present['osc_pure']) or 'none'}")
    lines.append(f"- kz_only: {', '.join(present['kz_only']) or 'none'}")
    lines.append(f"- kz_dz: {', '.join(present['kz_dz']) or 'none'}")
    lines.append("")

    # Simple textual trends by controller
    def _rows_for(ctrl):
        return [r for r in rows if r.get("controller") == ctrl]

    for ctrl in ["osc_pure", "kz_only", "kz_dz"]:
        rset = _rows_for(ctrl)
        if not rset:
            lines.append(f"{ctrl}: no data yet.")
            continue
        lines.append(f"{ctrl}:")
        # Sort by mu numeric
        def _mu_key(r):
            try:
                return float(r.get("mu", "nan"))
            except ValueError:
                return float("inf")
        rset = sorted(rset, key=_mu_key)
        for r in rset:
            mu = r.get("mu")
            lines.append(
                f"- mu={mu}: min_fz_mean={r.get('contact_min_fz_mean')}, max_df_mean={r.get('contact_max_df_mean')}, mean_fz={r.get('mean_fz')}"
            )
        lines.append("")

    with open(recap_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _collect_xy_trajectories(base_dir, controller, max_episodes=None):
    trajectories = []
    if controller == "osc_pure":
        h5_files = sorted(glob.glob(os.path.join(base_dir, controller, "**", "*.h5"), recursive=True))
        for f in h5_files:
            try:
                import h5py
            except ImportError as e:
                raise RuntimeError("h5py is required for OSC trajectory plots") from e
            with h5py.File(f, "r") as h5:
                if "ee_pos" not in h5:
                    continue
                pos = np.array(h5["ee_pos"], dtype=float)
            if pos.size == 0:
                continue
            trajectories.append(pos[:, :2])
            if max_episodes and len(trajectories) >= max_episodes:
                break
    else:
        traj_files = sorted(glob.glob(os.path.join(base_dir, controller, "**", "trajectories.json"), recursive=True))
        for path in traj_files:
            with open(path) as f:
                data = json.load(f)
            for ep in data.get("episodes", []):
                pos = np.array(ep.get("tcp_positions", []), dtype=float)
                if pos.size == 0:
                    continue
                trajectories.append(pos[:, :2])
                if max_episodes and len(trajectories) >= max_episodes:
                    break
            if max_episodes and len(trajectories) >= max_episodes:
                break
    return trajectories


def _collect_xyz_trajectories(base_dir, controller, max_episodes=None):
    trajectories = []
    if controller == "osc_pure":
        h5_files = sorted(glob.glob(os.path.join(base_dir, controller, "**", "*.h5"), recursive=True))
        for f in h5_files:
            try:
                import h5py
            except ImportError as e:
                raise RuntimeError("h5py is required for OSC trajectory plots") from e
            with h5py.File(f, "r") as h5:
                if "ee_pos" not in h5:
                    continue
                pos = np.array(h5["ee_pos"], dtype=float)
            if pos.size == 0:
                continue
            trajectories.append(pos[:, :3])
            if max_episodes and len(trajectories) >= max_episodes:
                break
    else:
        traj_files = sorted(glob.glob(os.path.join(base_dir, controller, "**", "trajectories.json"), recursive=True))
        for path in traj_files:
            with open(path) as f:
                data = json.load(f)
            for ep in data.get("episodes", []):
                pos = np.array(ep.get("tcp_positions", []), dtype=float)
                if pos.size == 0:
                    continue
                trajectories.append(pos[:, :3])
                if max_episodes and len(trajectories) >= max_episodes:
                    break
            if max_episodes and len(trajectories) >= max_episodes:
                break
    return trajectories


def _compute_mean_std_trajectory(trajectories):
    if not trajectories:
        return None, None
    min_len = min(len(t) for t in trajectories)
    if min_len <= 0:
        return None, None
    trimmed = [t[:min_len] for t in trajectories]
    stack = np.stack(trimmed, axis=0)
    mean = np.mean(stack, axis=0)
    if stack.shape[0] > 1:
        std = np.std(stack, axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


def _plot_mean_variability_3d(ax, mean, std, color):
    ax.plot(mean[:, 0], mean[:, 1], mean[:, 2], color=color, linewidth=2.0, label="mean")
    if std is None:
        return
    step = max(1, int(len(mean) / max(1, ELLIPSOID_STEPS)))
    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 8)
    for idx in range(0, len(mean), step):
        rx, ry, rz = std[idx]
        if rx <= 0 and ry <= 0 and rz <= 0:
            continue
        x = rx * np.outer(np.cos(u), np.sin(v)) + mean[idx, 0]
        y = ry * np.outer(np.sin(u), np.sin(v)) + mean[idx, 1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + mean[idx, 2]
        ax.plot_wireframe(x, y, z, color=color, alpha=0.18, linewidth=0.4)


def _save_figure(fig, path, transparent=False):
    fig.savefig(path, dpi=200, bbox_inches="tight", transparent=transparent)


def _compute_adaptability(controllers):
    metrics = [
        ("contact_max_df_mean", "lower", "oscillation"),
        ("contact_ratio_mean", "higher", "contact_ratio"),
        ("contact_min_fz_mean", "higher", "min_fz"),
    ]

    # Build OSC baseline per mu for relative improvements
    osc = controllers.get("osc_pure", {})
    osc_by_mu = {}
    for mu, payload in osc.items():
        osc_by_mu[mu] = payload["metrics"]

    rows = []
    for ctrl, data in controllers.items():
        if not data:
            continue
        if ctrl == "osc_pure":
            for _, _, short in metrics:
                rows.append({
                    "controller": ctrl,
                    "metric": short,
                    "mean_mu": float("nan"),
                    "std_mu": float("nan"),
                    "cv_mu": float("nan"),
                    "improvement_mean": 0.0,
                    "improvement_pos_frac": 0.0,
                    "mu_count": 0,
                })
            rows.append({
                "controller": ctrl,
                "metric": "overall",
                "improvement_mean": 0.0,
                "improvement_pos_frac": 0.0,
                "mu_count": 0,
            })
            continue

        overall_components = []
        for key, direction, short in metrics:
            improvements = []
            pos = 0
            count = 0
            for mu, payload in sorted(data.items(), key=lambda x: float(x[0])):
                osc_metrics = osc_by_mu.get(mu)
                if not osc_metrics:
                    continue
                osc_val = osc_metrics.get(key)
                val = payload["metrics"].get(key)
                if osc_val is None or val is None:
                    continue
                if not np.isfinite(osc_val) or not np.isfinite(val):
                    continue
                if direction == "lower":
                    imp = (osc_val - val) / (abs(osc_val) + EPS)
                else:
                    imp = (val - osc_val) / (abs(osc_val) + EPS)
                improvements.append(float(imp))
                count += 1
                if imp > 0:
                    pos += 1
            if improvements:
                mean_imp = float(np.mean(improvements))
                pos_frac = float(pos / max(count, 1))
                rows.append({
                    "controller": ctrl,
                    "metric": short,
                    "mean_mu": float("nan"),
                    "std_mu": float("nan"),
                    "cv_mu": float("nan"),
                    "improvement_mean": mean_imp,
                    "improvement_pos_frac": pos_frac,
                    "mu_count": count,
                })
                if short in ("oscillation", "contact_ratio"):
                    overall_components.append(mean_imp)
            else:
                rows.append({
                    "controller": ctrl,
                    "metric": short,
                    "mean_mu": float("nan"),
                    "std_mu": float("nan"),
                    "cv_mu": float("nan"),
                    "improvement_mean": float("nan"),
                    "improvement_pos_frac": float("nan"),
                    "mu_count": 0,
                })

        if overall_components:
            overall = float(np.mean(overall_components))
            rows.append({
                "controller": ctrl,
                "metric": "overall",
                "improvement_mean": overall,
                "improvement_pos_frac": float("nan"),
                "mu_count": len(overall_components),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot friction sweep results and summarize metrics")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=str(_path_setup.DATA_ROOT / "friction_tests"),
        help="Path to friction sweep run directory",
    )
    parser.add_argument(
        "--full_force_trends",
        action="store_true",
        help="Plot full force trends without phase trimming or markers",
    )
    parser.add_argument(
        "--plot_trajectories",
        action="store_true",
        help="Generate XY trajectory plots (one per controller) from existing data",
    )
    parser.add_argument(
        "--plot_trajectories_3d",
        action="store_true",
        help="Generate XYZ 3D trajectory plots (one per controller) from existing data",
    )
    parser.add_argument(
        "--plot_trajectories_3d_mean",
        action="store_true",
        help="Generate XYZ 3D mean-trajectory plots with variability ellipsoids",
    )
    parser.add_argument(
        "--traj_max",
        type=int,
        default=0,
        help="Optional limit on number of trajectories per controller (0 = all)",
    )
    parser.add_argument(
        "--traj_format",
        choices=["png", "svg"],
        default="png",
        help="Image format for trajectory plots",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save plots with transparent background",
    )
    args = parser.parse_args()

    base_dir = args.run_dir
    analysis_dir = os.path.join(base_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    controllers = {
        "osc_pure": _collect_controller(base_dir, "osc_pure"),
        "kz_only": _collect_controller(base_dir, "kz_only"),
        "kz_dz": _collect_controller(base_dir, "kz_dz"),
    }

    # Build rows for table
    rows = []
    for ctrl, data in controllers.items():
        for mu, payload in data.items():
            metrics = payload["metrics"].copy()
            metrics.update({"controller": ctrl, "mu": mu})
            rows.append(metrics)

    fieldnames = [
        "controller",
        "mu",
        "episodes",
        "steps",
        "mean_fz",
        "mean_fz_std",
        "min_fz",
        "min_fz_std",
        "contact_min_fz_mean",
        "contact_min_fz_std",
        "contact_max_df_mean",
        "contact_max_df_std",
        "contact_mean_df_mean",
        "contact_mean_df_std",
        "contact_ratio_mean",
        "contact_ratio_std",
    ]

    csv_path = os.path.join(analysis_dir, "summary_table.csv")
    md_path = os.path.join(analysis_dir, "summary_table.md")
    _write_table(csv_path, rows, fieldnames)
    _write_markdown(md_path, rows, fieldnames)

    recap_path = os.path.join(analysis_dir, "recap.txt")
    _make_recap(recap_path, controllers, rows)

    # Adaptability coefficient (robustness across mu)
    adapt_rows = _compute_adaptability(controllers)
    adapt_csv = os.path.join(analysis_dir, "adaptability.csv")
    adapt_md = os.path.join(analysis_dir, "adaptability.md")
    adapt_fields = [
        "controller",
        "metric",
        "mean_mu",
        "std_mu",
        "cv_mu",
        "improvement_mean",
        "improvement_pos_frac",
        "mu_count",
    ]
    _write_table(adapt_csv, adapt_rows, adapt_fields)
    _write_markdown(adapt_md, adapt_rows, adapt_fields)

    # Plot
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    ctrl_order = ["osc_pure", "kz_only", "kz_dz"]
    titles = ["OSC pure", "KZ-only", "KZ + DZ"]

    cut_time_by_mu = {}
    start_time_by_mu = {}
    lateral_markers_by_mu = {}
    if not args.full_force_trends:
        osc_data = controllers.get("osc_pure", {})
        for mu, payload in osc_data.items():
            start_time = payload.get("contact_time_mean")
            l0_time = payload.get("lateral0_time_mean")
            l1_time = payload.get("lateral1_time_mean")
            cut_time = payload.get("lateral2_time_mean")
            if start_time is not None and np.isfinite(start_time):
                start_time_by_mu[mu] = float(start_time)
            lateral_markers = []
            if l0_time is not None and np.isfinite(l0_time):
                lateral_markers.append(("L0", float(l0_time)))
            if l1_time is not None and np.isfinite(l1_time):
                lateral_markers.append(("L1", float(l1_time)))
            if cut_time is not None and np.isfinite(cut_time):
                lateral_markers.append(("L2", float(cut_time)))
            if lateral_markers:
                lateral_markers_by_mu[mu] = lateral_markers
            if cut_time is not None and np.isfinite(cut_time):
                cut_time_by_mu[mu] = float(cut_time)

    legend_handles = None
    legend_labels = None
    for ax, ctrl, title in zip(axes, ctrl_order, titles):
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Fz [N]")
        data = controllers.get(ctrl, {})
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.grid(True, alpha=0.3)
            continue
        for mu, payload in sorted(data.items(), key=lambda x: float(x[0])):
            series = payload["mean_series"]
            times = payload["times"]
            if times is None or len(times) < len(series):
                times = np.arange(len(series)) * 0.1
            else:
                times = times[: len(series)]
            if not args.full_force_trends:
                start_time = start_time_by_mu.get(mu)
                cut_time = cut_time_by_mu.get(mu)
                lateral_markers = lateral_markers_by_mu.get(mu, [])
                if start_time is not None:
                    idx = np.where(times >= start_time)[0]
                    if idx.size:
                        first = int(idx[0])
                        times = times[first:]
                        series = series[first:]
                if cut_time is not None:
                    idx = np.where(times <= cut_time)[0]
                    if idx.size:
                        last = int(idx[-1])
                        times = times[: last + 1]
                        series = series[: last + 1]
            if SMOOTH_WINDOW > 1:
                series = _smooth_series(series, SMOOTH_WINDOW)
            ax.plot(times, series, label=f"mu={mu}")
            if not args.full_force_trends:
                # Draw lateral markers (same for all controllers, derived from OSC timing)
                for label, t_marker in lateral_markers:
                    ax.axvline(t_marker, color="#000000", alpha=0.15, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle("Friction sweep: mean contact force over time")
    fig.tight_layout()

    plot_name = "force_trends_full.png" if args.full_force_trends else "force_trends.png"
    plot_path = os.path.join(analysis_dir, plot_name)
    fig.savefig(plot_path, dpi=200)

    # Standalone legend (same width as the 3-panel figure)
    if legend_handles and legend_labels:
        legend_fig = plt.figure(figsize=(18, 1.6))
        legend_fig.legend(
            legend_handles,
            legend_labels,
            loc="center",
            ncol=10,
            frameon=False,
        )
        legend_fig.gca().axis("off")
        legend_name = "force_trends_legend_full.png" if args.full_force_trends else "force_trends_legend.png"
        legend_path = os.path.join(analysis_dir, legend_name)
        legend_fig.savefig(legend_path, dpi=200, bbox_inches="tight")

    # Error-bar plot for summary metrics (mean ± std across episodes)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metrics_plot = [
        ("contact_min_fz_mean", "contact_min_fz_std", "Min Fz after contact [N]"),
        ("contact_max_df_mean", "contact_max_df_std", "Max |dF| after contact [N]"),
        ("mean_fz", "mean_fz_std", "Mean Fz [N]"),
    ]

    for ax, (mean_key, std_key, title) in zip(axes2, metrics_plot):
        ax.set_title(title)
        ax.set_xlabel("mu")
        ax.grid(True, alpha=0.3)
        for ctrl in ctrl_order:
            data = controllers.get(ctrl, {})
            mu_vals = []
            mean_vals = []
            std_vals = []
            for mu, payload in sorted(data.items(), key=lambda x: float(x[0])):
                metrics = payload["metrics"]
                mean_val = metrics.get(mean_key)
                std_val = metrics.get(std_key, 0.0)
                if mean_val is None or not np.isfinite(mean_val):
                    continue
                mu_vals.append(float(mu))
                mean_vals.append(float(mean_val))
                std_vals.append(0.0 if std_val is None or not np.isfinite(std_val) else float(std_val))
            if mu_vals:
                ax.errorbar(mu_vals, mean_vals, yerr=std_vals, fmt="-o", capsize=3, label=ctrl)
        ax.legend()

    fig2.suptitle("Friction sweep: summary metrics (mean ± std)")
    fig2.tight_layout()

    plot_path_err = os.path.join(analysis_dir, "force_metrics_errorbars.png")
    fig2.savefig(plot_path_err, dpi=200)

    # Robustness-focused plot: mean across mu with std bands
    colors = {
        "osc_pure": "#6c757d",
        "kz_only": "#0d6efd",
        "kz_dz": "#198754",
    }
    styles = {
        "osc_pure": "--",
        "kz_only": "-",
        "kz_dz": "-",
    }

    fig3, axes3 = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    font_inc = 4
    base_font = plt.rcParams.get("font.size", 10)
    title_fs = base_font + font_inc
    label_fs = base_font + font_inc
    tick_fs = base_font + font_inc
    legend_fs = base_font + font_inc
    legend_handles = None
    legend_labels = None
    for ax, (mean_key, std_key, title) in zip(axes3, metrics_plot):
        for ctrl in ctrl_order:
            data = controllers.get(ctrl, {})
            mu_vals = []
            mean_vals = []
            std_vals = []
            for mu, payload in sorted(data.items(), key=lambda x: float(x[0])):
                metrics = payload["metrics"]
                mean_val = metrics.get(mean_key)
                std_val = metrics.get(std_key, 0.0)
                if mean_val is None or not np.isfinite(mean_val):
                    continue
                mu_vals.append(float(mu))
                mean_vals.append(float(mean_val))
                std_vals.append(0.0 if std_val is None or not np.isfinite(std_val) else float(std_val))
            if not mu_vals:
                continue
            mu_vals = np.array(mu_vals)
            mean_vals = np.array(mean_vals)
            std_vals = np.array(std_vals)
            ax.plot(
                mu_vals,
                mean_vals,
                styles.get(ctrl, "-"),
                color=colors.get(ctrl, None),
                linewidth=2.2 if ctrl != "osc_pure" else 1.6,
                label=ctrl,
            )
            ax.fill_between(
                mu_vals,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=colors.get(ctrl, None),
                alpha=0.12 if ctrl != "osc_pure" else 0.08,
                linewidth=0,
            )
        ax.set_title(title, fontsize=title_fs)
        ax.set_ylabel("Force [N]", fontsize=label_fs)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_fs)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    axes3[-1].set_xlabel("mu (friction)", fontsize=label_fs)
    fig3.suptitle("Robustness vs friction: mean ± std across episodes", fontsize=title_fs)
    fig3.tight_layout(rect=(0, 0.06, 1, 1))
    if legend_handles and legend_labels:
        fig3.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=legend_fs,
            bbox_to_anchor=(0.5, 0.01),
        )
    plot_path_robust = os.path.join(analysis_dir, "robustness_over_mu.png")
    fig3.savefig(plot_path_robust, dpi=200)

    # Robustness summary: average std across mu (lower is better)
    fig4, axes4 = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, (mean_key, std_key, title) in zip(axes4, metrics_plot):
        labels = []
        vals = []
        for ctrl in ctrl_order:
            data = controllers.get(ctrl, {})
            std_vals = []
            for _, payload in sorted(data.items(), key=lambda x: float(x[0])):
                metrics = payload["metrics"]
                std_val = metrics.get(std_key, 0.0)
                if std_val is None or not np.isfinite(std_val):
                    continue
                std_vals.append(float(std_val))
            if not std_vals:
                continue
            labels.append(ctrl)
            vals.append(float(np.mean(std_vals)))
        if labels:
            ax.bar(labels, vals, color=[colors.get(l, "#333333") for l in labels])
        ax.set_title(f"Avg std across mu\n{title}")
        ax.set_ylabel("Std [N]")
        ax.grid(True, axis="y", alpha=0.3)
    fig4.suptitle("Robustness summary (lower variance = more robust)")
    fig4.tight_layout()
    plot_path_summary = os.path.join(analysis_dir, "robustness_summary.png")
    fig4.savefig(plot_path_summary, dpi=200)

    # Adaptability index plot (overall improvement vs OSC, higher = better)
    fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4))
    labels = []
    values = []
    for ctrl in ctrl_order:
        row = next((r for r in adapt_rows if r["controller"] == ctrl and r.get("metric") == "overall"), None)
        if not row:
            continue
        labels.append(ctrl)
        values.append(row.get("improvement_mean", float("nan")))
    if labels:
        ax5.bar(labels, values, color=[colors.get(l, "#333333") for l in labels])
        for i, v in enumerate(values):
            ax5.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax5.axhline(0.0, color="#333333", linewidth=1.0)
    ax5.set_ylabel("Avg relative improvement vs OSC")
    ax5.set_title("Adaptability across friction (higher = better)")
    ax5.grid(True, axis="y", alpha=0.3)
    fig5.tight_layout()
    plot_path_adapt = os.path.join(analysis_dir, "adaptability_index.png")
    fig5.savefig(plot_path_adapt, dpi=200)

    # Trajectory plots (XY) - one per controller
    if args.plot_trajectories:
        max_eps = args.traj_max if args.traj_max > 0 else None
        traj_data = {}
        for ctrl in ctrl_order:
            traj_data[ctrl] = _collect_xy_trajectories(base_dir, ctrl, max_episodes=max_eps)

        # Determine global limits for consistent axes
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")
        for trajs in traj_data.values():
            for t in trajs:
                x_min = min(x_min, float(np.min(t[:, 0])))
                x_max = max(x_max, float(np.max(t[:, 0])))
                y_min = min(y_min, float(np.min(t[:, 1])))
                y_max = max(y_max, float(np.max(t[:, 1])))
        if x_min < x_max and y_min < y_max:
            margin = 0.02
            for ctrl in ctrl_order:
                fig_t, ax_t = plt.subplots(figsize=(5, 5))
                color = colors.get(ctrl, "#333333")
                for t in traj_data.get(ctrl, []):
                    ax_t.plot(t[:, 0], t[:, 1], color=color, alpha=0.15, linewidth=0.8)
                ax_t.set_title(f"Trajectory XY — {ctrl}")
                ax_t.set_xlabel("X [m]")
                ax_t.set_ylabel("Y [m]")
                ax_t.set_xlim(x_min - margin, x_max + margin)
                ax_t.set_ylim(y_min - margin, y_max + margin)
                ax_t.set_aspect("equal", adjustable="box")
                ax_t.grid(True, alpha=0.3)
                traj_path = os.path.join(analysis_dir, f"trajectory_xy_{ctrl}.{args.traj_format}")
                fig_t.tight_layout()
                _save_figure(fig_t, traj_path, transparent=args.transparent)

    # 3D trajectory plots (XYZ) - one per controller
    if args.plot_trajectories_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        max_eps = args.traj_max if args.traj_max > 0 else None
        traj_data = {}
        for ctrl in ctrl_order:
            traj_data[ctrl] = _collect_xyz_trajectories(base_dir, ctrl, max_episodes=max_eps)

        # Determine global limits for consistent axes
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")
        z_min = float("inf")
        z_max = float("-inf")
        for trajs in traj_data.values():
            for t in trajs:
                x_min = min(x_min, float(np.min(t[:, 0])))
                x_max = max(x_max, float(np.max(t[:, 0])))
                y_min = min(y_min, float(np.min(t[:, 1])))
                y_max = max(y_max, float(np.max(t[:, 1])))
                z_min = min(z_min, float(np.min(t[:, 2])))
                z_max = max(z_max, float(np.max(t[:, 2])))
        if x_min < x_max and y_min < y_max and z_min < z_max:
            margin = 0.02
            for ctrl in ctrl_order:
                fig_t = plt.figure(figsize=(6, 6))
                ax_t = fig_t.add_subplot(111, projection="3d")
                color = colors.get(ctrl, "#333333")
                for t in traj_data.get(ctrl, []):
                    ax_t.plot(t[:, 0], t[:, 1], t[:, 2], color=color, alpha=0.15, linewidth=0.8)
                ax_t.set_title(f"Trajectory XYZ — {ctrl}")
                ax_t.set_xlabel("X [m]")
                ax_t.set_ylabel("Y [m]")
                ax_t.set_zlabel("Z [m]")
                ax_t.set_xlim(x_min - margin, x_max + margin)
                ax_t.set_ylim(y_min - margin, y_max + margin)
                ax_t.set_zlim(z_min - margin, z_max + margin)
                ax_t.view_init(elev=25, azim=-60)
                ax_t.grid(True, alpha=0.3)
                fig_t.tight_layout()
                traj_path = os.path.join(analysis_dir, f"trajectory_xyz_{ctrl}.{args.traj_format}")
                _save_figure(fig_t, traj_path, transparent=args.transparent)

    # 3D trajectory mean + variability plots (XYZ) - one per controller
    if args.plot_trajectories_3d_mean:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        max_eps = args.traj_max if args.traj_max > 0 else None
        traj_data = {}
        mean_std = {}
        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")
        z_min = float("inf")
        z_max = float("-inf")

        for ctrl in ctrl_order:
            traj_data[ctrl] = _collect_xyz_trajectories(base_dir, ctrl, max_episodes=max_eps)
            mean, std = _compute_mean_std_trajectory(traj_data[ctrl])
            if mean is None:
                continue
            mean_std[ctrl] = (mean, std)
            std_max = np.max(std, axis=0) if std is not None and std.size else np.zeros(3)
            x_min = min(x_min, float(np.min(mean[:, 0]) - std_max[0]))
            x_max = max(x_max, float(np.max(mean[:, 0]) + std_max[0]))
            y_min = min(y_min, float(np.min(mean[:, 1]) - std_max[1]))
            y_max = max(y_max, float(np.max(mean[:, 1]) + std_max[1]))
            z_min = min(z_min, float(np.min(mean[:, 2]) - std_max[2]))
            z_max = max(z_max, float(np.max(mean[:, 2]) + std_max[2]))

        if mean_std and x_min < x_max and y_min < y_max and z_min < z_max:
            margin = 0.02
            for ctrl in ctrl_order:
                if ctrl not in mean_std:
                    continue
                mean, std = mean_std[ctrl]
                fig_t = plt.figure(figsize=(6, 6))
                ax_t = fig_t.add_subplot(111, projection="3d")
                color = colors.get(ctrl, "#333333")
                _plot_mean_variability_3d(ax_t, mean, std, color)
                ax_t.set_title(f"Trajectory XYZ (mean + variability) — {ctrl}")
                ax_t.set_xlabel("X [m]")
                ax_t.set_ylabel("Y [m]")
                ax_t.set_zlabel("Z [m]")
                ax_t.set_xlim(x_min - margin, x_max + margin)
                ax_t.set_ylim(y_min - margin, y_max + margin)
                ax_t.set_zlim(z_min - margin, z_max + margin)
                ax_t.view_init(elev=25, azim=-60)
                ax_t.grid(True, alpha=0.3)
                fig_t.tight_layout()
                traj_path = os.path.join(analysis_dir, f"trajectory_xyz_meanvar_{ctrl}.{args.traj_format}")
                _save_figure(fig_t, traj_path, transparent=args.transparent)

    print("Saved:")
    print("-", plot_path)
    print("-", plot_path_err)
    print("-", plot_path_robust)
    print("-", plot_path_summary)
    print("-", plot_path_adapt)
    print("-", csv_path)
    print("-", md_path)
    print("-", recap_path)
    print("-", adapt_csv)
    print("-", adapt_md)


if __name__ == "__main__":
    main()
