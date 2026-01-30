import yaml
from pathlib import Path

# Load configuration parameters for PolishEnv (YAML)
_cfg_path = Path(__file__).parent / "config_parameters.yaml"
with _cfg_path.open() as f:
    _params = yaml.safe_load(f)

# Expose contact thresholds as floats
F_TOUCH = float(_params["F_TOUCH"])
F_LOST = float(_params["F_LOST"])

__all__ = ["F_TOUCH", "F_LOST"]
