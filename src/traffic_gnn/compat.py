"""
Compatibility helpers for migrating from monolithic `GNN0.py`.
This file lets you keep using legacy baseline outputs while moving to modular code.
"""

import importlib.util
from pathlib import Path


def load_legacy_module(legacy_path: str):
    p = Path(legacy_path)
    if not p.exists():
        raise FileNotFoundError(f"Legacy file not found: {legacy_path}")
    spec = importlib.util.spec_from_file_location("gnn0_legacy", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
