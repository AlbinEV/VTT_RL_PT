# utils/data_logger.py
#
# Stand-alone HDF5 logger for Isaac Lab / RL-Games episodes.
# ──────────────────────────────────────────────────────────
# 1. `DataLogger`  : low-level buffered writer (unchanged logic).
# 2. `EpisodeLogger`: thin wrapper that opens / closes one `DataLogger`
#    per episode so you never have to manage files inside the main loop.

from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
import yaml


# ════════════════════════════════════════════════════════════════════
# 1. Low-level writer
# ════════════════════════════════════════════════════════════════════
class DataLogger:
    """
    Buffered HDF5 writer.

    • The YAML file must contain a key ``log_vars`` with the list of fields to
      keep. Missing keys are written as zeros.
    • Datasets are created on the first flush so the code can infer dtypes
      and shapes automatically.
    """

    def __init__(self, path: str | Path, yaml_cfg: str, chunk_size: int = 1, attrs: Dict[str, Any] | None = None):
        """
        Args
        ----
        path       : Output file path (`.h5`).
        yaml_cfg   : YAML file with ``log_vars`` list.
        chunk_size : How many time-steps to keep in RAM before flushing.
        """
        self.path = Path(path)
        self.chunk_size = int(chunk_size)

        with open(yaml_cfg, "r") as f:
            self.var_list = yaml.safe_load(f)["log_vars"]

        self._buffer: list[Dict[str, Any]] = []
        self._datasets: dict[str, h5py.Dataset] = {}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")
        # Optional file-level attributes (metadata)
        if attrs:
            for k, v in attrs.items():
                try:
                    self._h5.attrs[k] = v
                except Exception:
                    # Fallback: store as string
                    try:
                        self._h5.attrs[k] = str(v)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    def log(self, data: dict):
        """Append one record to the in-memory buffer."""
        # Keep only variables of interest
        rec = {}
        for k in self.var_list:
            v = data.get(k)
            if v is None:
                rec[k] = 0.0
            else:
                try:
                    arr = np.asarray(v)
                    if arr.dtype == object:
                        arr = arr.astype(np.float32, copy=False)
                    rec[k] = arr
                except Exception:
                    # fallback scalar
                    try:
                        rec[k] = float(v)
                    except Exception:
                        rec[k] = 0.0
        self._buffer.append(rec)
        if len(self._buffer) >= self.chunk_size:
            self._flush()

    # ------------------------------------------------------------------
    def _flush(self):
        """Write buffered records to disk."""
        if not self._datasets:
            # First time → create datasets
            first = self._buffer[0]
            for k, v in first.items():
                arr = np.asarray(v)
                if arr.dtype == object:
                    arr = arr.astype(np.float32)
                shape = (0, *arr.shape)
                self._datasets[k] = self._h5.create_dataset(
                    k,
                    shape=shape,
                    maxshape=(None, *arr.shape),
                    dtype=arr.dtype,
                    chunks=(self.chunk_size, *arr.shape),
                    compression="gzip",
                )

        n_new = len(self._buffer)
        old_total = next(iter(self._datasets.values())).shape[0]
        new_total = old_total + n_new

        # Extend all datasets
        for ds in self._datasets.values():
            ds.resize((new_total, *ds.shape[1:]))

        # Write records
        for i, rec in enumerate(self._buffer, start=old_total):
            for k, ds in self._datasets.items():
                val = rec.get(k)
                arr = np.zeros(ds.shape[1:], dtype=ds.dtype) if val is None else np.asarray(val)
                if arr.dtype == object:
                    arr = arr.astype(ds.dtype)
                else:
                    arr = arr.astype(ds.dtype, copy=False)
                ds[i] = arr.reshape(ds.shape[1:])

        self._h5.flush()
        self._buffer.clear()

    # ------------------------------------------------------------------
    def flush(self):
        """Force flush of any remaining samples."""
        if self._buffer:
            self._flush()

    def close(self):
        """Flush (if needed) and close the HDF5 file."""
        self.flush()
        self._h5.close()


# ════════════════════════════════════════════════════════════════════
# 2. Episode-aware wrapper
# ════════════════════════════════════════════════════════════════════
class EpisodeLogger:
    """
    Rotate one HDF5 file per episode.

    Example
    -------
    ```python
    elog = EpisodeLogger("logs/run42", "log_vars.yaml")

    while True:
        elog.log(env.export_log_dict())
        if done:                      # episode finished
            elog.next_episode()
    ```
    """

    def __init__(
        self,
        run_dir: str | Path,
        yaml_cfg: str,
        chunk_size: int = 512,
        start_idx: int = 0,
        attrs: Dict[str, Any] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.yaml_cfg = yaml_cfg
        self.chunk_size = int(chunk_size)
        self.ep_idx = int(start_idx)
        self.attrs = attrs or {}

        self._logger = None
        self._open_new_logger()

    # ------------------------------------------------------------------
    def log(self, data: dict):
        """Forward log call to the underlying `DataLogger`."""
        if self._logger is not None:
            self._logger.log(data)

    def next_episode(self):
        """Close current file and open a new one, incrementing the index."""
        if self._logger is not None:
            self._logger.close()
        self.ep_idx += 1
        self._open_new_logger()

    def close(self):
        """Close the current logger (to be called at program exit)."""
        if self._logger is not None:
            self._logger.close()

    # ------------------------------------------------------------------
    def _open_new_logger(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"episode_{self.ep_idx:04d}_{ts}.h5"
        path = self.run_dir / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._logger = DataLogger(path, self.yaml_cfg, self.chunk_size, attrs=self.attrs)
        print(f"[EpisodeLogger] ➡️  created {path}")


# ----------------------------------------------------------------------
__all__ = ["DataLogger", "EpisodeLogger"]
