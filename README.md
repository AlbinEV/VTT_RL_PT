# VTT_RL_PT: Procedural Trajectory for Robotic Polishing

Procedural Trajectory (PT) environment scaffold for Isaac Lab 2.0.1.

This is a public-ready standalone base. The PT code will be migrated here from the private repository.

## Whats Included

- `source/vtt_rl_pt/` - Isaac Lab extension-style package (placeholder)
- `scripts/` - Train/play utilities (placeholder)
- `examples/` - Demo trajectories and minimal configs (placeholder)
- `docs/` - Public documentation (placeholder)

## Quick Start

### 1. Prerequisites

- Isaac Lab 2.0.1 installed and working
- Python 3.10
- CUDA 11.8+ (for GPU acceleration)

### 2. Installation

Clone the repository and install the extension package in editable mode:

```bash
git clone <REPO_URL>
cd VTT_RL_PT
python -m pip install -e source/vtt_rl_pt
```

### 3. Train (placeholder)

```bash
cd <ISAAC_LAB_PATH>
./isaaclab.sh -p /path/to/VTT_RL_PT/scripts/train.py --num_envs <N> --max_iterations 500
```

### 4. Evaluate (placeholder)

```bash
./isaaclab.sh -p /path/to/VTT_RL_PT/scripts/play.py --checkpoint <PATH>
```

## Environment Details (TBD)

- Action space: TBD
- Observation space: TBD
- Reward: TBD
- Procedural trajectory generation: TBD

## File Structure

```
VTT_RL_PT/
├── docs/                 # Public documentation
├── examples/             # Demo scripts / trajectories
├── scripts/              # Train/play/utility scripts
├── source/               # Isaac Lab extension-style package
│   └── vtt_rl_pt/
│       ├── config/
│       │   └── extension.toml
│       ├── vtt_rl_pt/
│       │   └── __init__.py
│       └── setup.py
├── LICENSE
├── pyproject.toml
└── .gitignore
```

## Configuration (TBD)

Environment variables and config files will be documented here once the PT code is ported.

## Citation (TBD)

If you use this in research, add a citation here once the repository is public.

