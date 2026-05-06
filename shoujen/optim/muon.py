"""Muon optimizer (single-process version) + helper to split params.

Muon = SGD-momentum on 2D hidden matrices, with the gradient passed through a
Newton-Schulz orthogonalization at every step. See:
    https://kellerjordan.github.io/posts/muon/

This implementation is the single-process variant: no distributed sharding, no
DDP gradient gather. Good enough for a toy single-machine MPS run.

`build_optimizers` returns (Muon, AdamW). Grouping is conservative:
    1. Preserve RWKV-v7 optimizer intent first: `w0` gets a 2x AdamW lr,
       time-mix LoRA/scalar params, norm and bias params use AdamW without
       weight decay.
    2. Move only dense-input linear layer matrices to Muon. Embeddings, PLE,
       LM head, norm, bias, per-layer gates and RWKV small/LoRA params stay on
       AdamW.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration to orthogonalize a 2D matrix `g`.

    Coefficients (3.4445, -4.7750, 2.0315) are the Muon defaults from the
    original blog post; they are tuned to converge a normalized matrix to its
    nearest orthogonal counterpart in ~5 steps.
    """
    assert g.ndim == 2, f"newton_schulz expects 2D matrix, got shape {tuple(g.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.float32)
    x = x / (x.norm() + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon requires 2D parameters; got shape {tuple(p.shape)}"
                    )
                g = p.grad
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.mul_(mu).add_(g)
                use = g.add(buf, alpha=mu) if nesterov else buf
                update = newton_schulz(use, steps=ns_steps)
                scale = 0.2 * max(update.shape[0], update.shape[1]) ** 0.5
                p.add_(update, alpha=-lr * scale)

        return loss


_MUON_DENSE_LINEAR_SUFFIXES = (
    ".token_mixer.receptance.weight",
    ".token_mixer.key.weight",
    ".token_mixer.value.weight",
    ".token_mixer.output.weight",
    ".token_mixer.q_proj.weight",
    ".token_mixer.k_proj.weight",
    ".token_mixer.v_proj.weight",
    ".token_mixer.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def _is_muon_param(name: str, param: nn.Parameter) -> bool:
    if param.ndim != 2:
        return False
    if not name.endswith(".weight"):
        return False
    if min(param.shape) < 16:
        return False
    return name.endswith(_MUON_DENSE_LINEAR_SUFFIXES)


def _is_rwkv_w0(name: str) -> bool:
    return ".token_mixer.w0" in name or ".tmix.w0" in name


def _is_adamw_no_decay(name: str, param: nn.Parameter) -> bool:
    if _is_rwkv_w0(name):
        return True
    if param.ndim < 2:
        return True
    if not name.endswith(".weight"):
        return True
    for kw in ("norm", "ln_x", ".bias", ".r_k", ".k_k", ".k_a"):
        if kw in name:
            return True
    return False


def build_optimizers(
    model: nn.Module,
    *,
    muon_lr: float = 3e-4,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    muon_wd: float = 0.0,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    adamw_wd: float = 0.1,
    adamw_foreach: bool | None = None,
) -> tuple[Muon, torch.optim.AdamW]:
    muon_params: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    adamw_rwkv_w0: list[nn.Parameter] = []
    seen: set[int] = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        if _is_muon_param(name, p):
            muon_params.append(p)
        elif _is_rwkv_w0(name):
            adamw_rwkv_w0.append(p)
        else:
            if _is_adamw_no_decay(name, p):
                adamw_no_decay.append(p)
            else:
                adamw_decay.append(p)

    muon = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        ns_steps=muon_ns_steps,
        weight_decay=muon_wd,
    )
    adamw_kwargs = {}
    if adamw_foreach is not None:
        adamw_kwargs["foreach"] = adamw_foreach
    adamw = torch.optim.AdamW(
        [
            {"params": adamw_decay, "weight_decay": adamw_wd},
            {"params": adamw_rwkv_w0, "weight_decay": 0.0, "lr": adamw_lr * 2.0, "my_lr_scale": 2.0},
            {"params": adamw_no_decay, "weight_decay": 0.0},
        ],
        lr=adamw_lr,
        betas=adamw_betas,
        eps=adamw_eps,
        **adamw_kwargs,
    )
    return muon, adamw
