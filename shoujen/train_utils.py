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


def move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a batch tensor to device without triggering MPS async-copy corruption.

    PyTorch MPS has shown intermittent corruption for small integer/float batch
    tensors when they are copied while large model allocations are live. Keep
    CUDA non-blocking copies, but make MPS transfers contiguous, owned, and
    synchronized before the training step reads them.
    """
    if device.type == "mps":
        out = tensor.clone().contiguous().to(device)
        torch.mps.synchronize()
        return out
    return tensor.to(device, non_blocking=device.type == "cuda")


def warmup_cosine_lr(step: int, *, warmup: int, max_steps: int, min_ratio: float = 0.1) -> float:
    """Returns a multiplier in [min_ratio, 1] following warmup -> cosine."""
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def warmup_stable_decay_lr(
    step: int,
    *,
    warmup: int,
    max_steps: int,
    stable_steps: int = 0,
    min_ratio: float = 0.0,
) -> float:
    """Returns a warmup -> optional stable -> linear decay multiplier."""
    if step < warmup:
        return float(step + 1) / max(1, warmup)

    decay_start = warmup + max(0, stable_steps)
    if step < decay_start:
        return 1.0

    progress = (step - decay_start) / max(1, max_steps - decay_start)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * (1.0 - progress)


def set_optimizer_lr(opt: torch.optim.Optimizer, base_lrs: list[float], mult: float) -> None:
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * mult


def warmup_momentum(step: int, *, warmup: int, start: float, end: float) -> float:
    """Linear ramp from `start` to `end` over `warmup` steps, then `end`.

    Used for Muon's momentum: early in training the momentum buffer is mostly
    noise, so a lower initial momentum prevents amplifying that noise into the
    Newton-Schulz update direction.
    """
    if warmup <= 0 or step >= warmup:
        return end
    progress = step / warmup
    return start + (end - start) * progress


def set_optimizer_momentum(opt: torch.optim.Optimizer, momentum: float) -> None:
    for g in opt.param_groups:
        if "momentum" in g:
            g["momentum"] = momentum


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


_VOCAB_RESIZABLE_KEYS = {
    "model.embed_tokens.weight",
    "model.embed_tokens_per_layer.weight",
    "lm_head.weight",
}


def _load_state_dict_with_appended_vocab(
    model: nn.Module,
    checkpoint_model_state: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    current_model_state = model.state_dict()
    patched_model_state = dict(checkpoint_model_state)
    resized: list[dict[str, Any]] = []

    for name in _VOCAB_RESIZABLE_KEYS:
        saved = checkpoint_model_state.get(name)
        current = current_model_state.get(name)
        if saved is None or current is None or saved.shape == current.shape:
            continue
        if saved.ndim != current.ndim or saved.shape[1:] != current.shape[1:]:
            continue
        if saved.shape[0] > current.shape[0]:
            continue

        patched = current.detach().clone()
        patched[: saved.shape[0]].copy_(saved.to(device=patched.device, dtype=patched.dtype))
        patched_model_state[name] = patched
        resized.append(
            {
                "name": name,
                "checkpoint_shape": tuple(saved.shape),
                "model_shape": tuple(current.shape),
                "new_rows": int(current.shape[0] - saved.shape[0]),
            }
        )

    return resized, patched_model_state


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location="cpu",
    allow_appended_vocab: bool = False,
) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model_state = state["model"]
    if allow_appended_vocab:
        resized, model_state = _load_state_dict_with_appended_vocab(model, model_state)
        state["_shoujen_appended_vocab_keys"] = resized
    model.load_state_dict(model_state)
    return state


def filter_optimizer_state_for_param_shapes(
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict,
) -> tuple[dict, list[int]]:
    """Drop optimizer state entries whose tensor shapes no longer match params.

    This is used when resuming a checkpoint after appending tokenizer rows. The
    model weights can copy old rows into a larger embedding matrix, but AdamW's
    moment buffers for those matrices still have the old row count and must be
    reinitialized.
    """
    saved_groups = optimizer_state.get("param_groups", [])
    saved_param_ids = [pid for group in saved_groups for pid in group.get("params", [])]
    current_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(saved_param_ids) != len(current_params):
        return optimizer_state, []

    skipped: list[int] = []
    filtered_state = {}
    saved_state = optimizer_state.get("state", {})
    for saved_pid, param in zip(saved_param_ids, current_params):
        param_state = saved_state.get(saved_pid)
        if param_state is None:
            continue
        shape_mismatch = False
        for value in param_state.values():
            if torch.is_tensor(value) and value.ndim > 0 and tuple(value.shape) != tuple(param.shape):
                shape_mismatch = True
                break
        if shape_mismatch:
            skipped.append(saved_pid)
        else:
            filtered_state[saved_pid] = param_state

    filtered = dict(optimizer_state)
    filtered["state"] = filtered_state
    filtered["param_groups"] = [dict(group) for group in saved_groups]
    return filtered, skipped
