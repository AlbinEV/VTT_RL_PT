"""Path setup helper for local scripts.

Adds the extension source folder to sys.path so robo_pp_fixed can be imported
without installing the package.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'source' / 'vtt_rl_pt'
DATA_ROOT = Path(os.environ.get('VTT_RL_DATA', REPO_ROOT / 'data'))

if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
