"""Helpers shared by the training script: device detection, LR schedules,
checkpointing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Pick a sensible mixed-precision dtype, or None to disable autocast."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return None


def warmup_cosine_lr(step: int, *, warmup: int, max_steps: int, min_ratio: float = 0.1) -> float:
    """Returns a multiplier in [min_ratio, 1] following warmup -> cosine."""
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_optimizer_lr(opt: torch.optim.Optimizer, base_lrs: list[float], mult: float) -> None:
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * mult


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: Any,
    step: int,
    muon: torch.optim.Optimizer | None = None,
    adamw: torch.optim.Optimizer | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "config": config.to_dict() if hasattr(config, "to_dict") else config.__dict__,
        "step": step,
    }
    if muon is not None:
        state["muon"] = muon.state_dict()
    if adamw is not None:
        state["adamw"] = adamw.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: nn.Module, *, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    return state
