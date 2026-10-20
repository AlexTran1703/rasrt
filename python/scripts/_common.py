"""Shared helpers of the pipeline scripts: import path, result files."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # rasrt/python


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def write(path, obj, indent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent))


def keep(m, keys=("ok", "t_post_max_ms", "sram", "load_miss", "load_n", "raw_viol", "hash")):
    return {k: m[k] for k in keys}
