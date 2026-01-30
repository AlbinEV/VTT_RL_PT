# Usage Guide (Fixed Mode)

This guide summarizes the main scripts and how to run them with Isaac Lab 2.0.1.

## Global Setup

Set a data root for outputs (logs, trajectories, plots):

```bash
export VTT_RL_DATA=/path/to/data_root
```

Default if not set: `VTT_RL_PT/data/`.

## Training

### Kp_z only (main training used in grid search)

```bash
./isaaclab.sh -p scripts/train_kpz_only.py --num_envs 16 --max_epochs 200
```

### Kp_z + damping (KZ+DZ)

```bash
./isaaclab.sh -p scripts/train_kz_dz.py --num_envs 16 --max_epochs 200
```

### Safe Kp_z (contact-safe variant)

```bash
./isaaclab.sh -p scripts/train_kpz_safe.py --num_envs 16 --max_epochs 200
```

## Evaluation / Play

### Play a checkpoint

```bash
./isaaclab.sh -p scripts/play.py --checkpoint /path/to/model.pth
```

### Play Kp_z only (auto-picks latest in logs)

```bash
./isaaclab.sh -p scripts/play_kz.py
```

### Evaluate KZ+DZ policy and extract trajectories

```bash
./isaaclab.sh -p scripts/extract_trajectories.py \
  --checkpoint /path/to/model.pth \
  --num_episodes 10 \
  --output ${VTT_RL_DATA}/trajectories
```

## Grid Search

```bash
python scripts/run_grid_search.py --num_runs 50 --epochs_per_run 100
```

Outputs are written under `${VTT_RL_DATA}/grid_search/`.

## Friction Sweep

Set checkpoint paths before running:

```bash
export KZ_CHECKPOINT=/path/to/kz_only.pth
export KZ_DZ_CHECKPOINT=/path/to/kz_dz.pth
./scripts/run_friction_sweep.sh
```

Outputs are written under `${VTT_RL_DATA}/results_phd/friction_tests/`.

## Analysis / Plotting

All plotting scripts read from `VTT_RL_DATA` by default:

- `scripts/plot_friction_results.py`
- `scripts/plot_grid_search_correlations.py`
- `scripts/plot_eval_results.py`
- `scripts/plot_osc_optimization_results.py`
- `scripts/plot_trajectory_html.py`

Example:

```bash
python scripts/plot_friction_results.py --run_dir ${VTT_RL_DATA}/friction_tests/run_YYYYMMDD_HHMMSS
```

## Notes

- Most scripts assume a CUDA device (`cuda:0`).
- The Fixed Mode task is registered as `Polish-Fixed-v0`.
- If you install the package (`pip install -e source/vtt_rl_pt`), scripts import `robo_pp_fixed` directly.
