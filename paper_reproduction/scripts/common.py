"""Paths and small helpers shared by the reproduction scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]          # paper_reproduction/
REPO = ROOT.parent                                    # unofficial-depth-as-time/
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
MEANFLOW_REPO = Path("/home/dsi/wajeeha/MeanFlow-PyTorch")

sys.path.insert(0, str(REPO / "src"))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(v):
        if isinstance(v, Tensor):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v

    path.write_text(json.dumps(conv(obj), indent=1))


def free_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
