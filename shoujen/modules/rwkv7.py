"""RWKV-7 x070 time-mixing block, implemented with transparent PyTorch ops.

The parameterization and initialization follow the official RWKV-v7 `x070`
training implementation as closely as practical in this toy codebase. The CUDA
kernel path in RWKV clamps and exponentiates `w` internally; here that is made
explicit before the recurrent loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shoujen.modules.rwkv7_mps import can_use_mps_wind_backstepping, mps_wind_backstepping


def _round_lora_dim(value: float) -> int:
    return max(32, int(round(value / 32.0) * 32))


def _ortho_init_(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    with torch.no_grad():
        shape = tensor.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(tensor, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(tensor[i], gain=gain * scale)
        else:
            nn.init.normal_(tensor, std=0.02)
    return tensor


def rwkv7_recurrent(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    neg_kk: torch.Tensor,
    kka: torch.Tensor,
    state: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrent RWKV-v7 wind-backstepping update.

    Mirrors the official CUDA / Metal-MPS shader: pre-decay state is projected
    by `neg_kk` (shader's `a`), then state is updated in a single fused step
    with decay, the `sa·kka` (shader's `b`) outer product and the `v·k` outer
    product. `r` plays the role of the shader's `q` for output projection.

    Args:
        r, k, v, w, neg_kk, kka: tensors of shape `(B, H, T, D)`.
        state: optional recurrent state `(B, H, D, D)` (axis order `[i, j]`).
        reset_mask: optional bool tensor `(B, T)` that zeros recurrent state
            before processing sequence starts inside a packed block.

    Returns:
        Output `(B, H, T, D)` and the final recurrent state.
    """
    bsz, n_head, seq_len, head_dim = r.shape
    device, dtype = r.device, r.dtype

    if state is None:
        state = torch.zeros(bsz, n_head, head_dim, head_dim, device=device, dtype=torch.float32)
    state = state.to(torch.float32)

    out = torch.empty(bsz, n_head, seq_len, head_dim, device=device, dtype=torch.float32)
    rf = r.float()
    kf = k.float()
    vf = v.float()
    wf = w.float()
    af = neg_kk.float()
    bf = kka.float()
    if reset_mask is not None:
        reset_mask = reset_mask.to(device=device, dtype=torch.bool)

    for t in range(seq_len):
        if reset_mask is not None:
            keep = (~reset_mask[:, t]).to(torch.float32).view(bsz, 1, 1, 1)
            state = state * keep

        rt = rf[:, :, t]
        kt = kf[:, :, t]
        vt = vf[:, :, t]
        wt = wf[:, :, t]
        at = af[:, :, t]
        bt = bf[:, :, t]

        sa = (state * at.unsqueeze(-2)).sum(-1, keepdim=True)
        state = (
            state * wt.unsqueeze(-2)
            + sa * bt.unsqueeze(-2)
            + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        )
        out[:, :, t] = (state * rt.unsqueeze(-2)).sum(-1)

    return out.to(dtype), state


class RWKV7TimeMix(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        if hidden_size % head_dim != 0:
            raise ValueError("RWKV7 expects hidden_size divisible by head_dim")

        self.n_head = hidden_size // head_dim
        self.head_dim = head_dim

        n_layer = max(1, config.num_hidden_layers)
        ratio_0_to_1 = layer_idx / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (layer_idx / n_layer)
        channel_pos = torch.arange(hidden_size, dtype=torch.float32).view(1, 1, hidden_size) / hidden_size

        self.x_r = nn.Parameter(1.0 - torch.pow(channel_pos, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(channel_pos, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(channel_pos, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(channel_pos, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(channel_pos, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(channel_pos, 0.2 * ratio_1_to_almost0))

        idx = torch.arange(hidden_size, dtype=torch.float32)
        linear = idx / max(1, hidden_size - 1) - 0.5
        zigzag = ((idx % head_dim) - ((head_dim - 1) / 2.0)) / ((head_dim - 1) / 2.0)
        zigzag = zigzag * zigzag.abs()
        www = -6.0 + 6.0 * torch.pow(idx / max(1, hidden_size - 1), 1.0 + ratio_0_to_1**0.3)

        d_decay = config.rwkv_decay_lora or _round_lora_dim(2.5 * hidden_size**0.5)
        self.w1 = nn.Parameter(torch.zeros(hidden_size, d_decay))
        self.w2 = nn.Parameter(_ortho_init_(torch.zeros(d_decay, hidden_size), 0.1))
        self.w0 = nn.Parameter((www + 0.5 + zigzag * 2.5).view(1, 1, hidden_size))

        d_aaa = config.rwkv_aaa_lora or _round_lora_dim(2.5 * hidden_size**0.5)
        self.a1 = nn.Parameter(torch.zeros(hidden_size, d_aaa))
        self.a2 = nn.Parameter(_ortho_init_(torch.zeros(d_aaa, hidden_size), 0.1))
        self.a0 = nn.Parameter((torch.zeros(hidden_size) - 0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, hidden_size))

        d_value = config.rwkv_v_first_lora or _round_lora_dim(1.7 * hidden_size**0.5)
        self.v1 = nn.Parameter(torch.zeros(hidden_size, d_value))
        self.v2 = nn.Parameter(_ortho_init_(torch.zeros(d_value, hidden_size), 0.1))
        self.v0 = nn.Parameter((torch.zeros(hidden_size) + 0.73 - linear * 0.4).view(1, 1, hidden_size))

        d_gate = config.rwkv_gate_lora or _round_lora_dim(5.0 * hidden_size**0.5)
        self.g1 = nn.Parameter(torch.zeros(hidden_size, d_gate))
        self.g2 = nn.Parameter(_ortho_init_(torch.zeros(d_gate, hidden_size), 0.1))

        self.k_k = nn.Parameter((torch.zeros(hidden_size) + 0.71 - linear * 0.1).view(1, 1, hidden_size))
        self.k_a = nn.Parameter(torch.zeros(1, 1, hidden_size) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(self.n_head, head_dim) - 0.04)

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, hidden_size, eps=64e-5)

        self.reset_rwkv_parameters()

    def reset_rwkv_parameters(self) -> None:
        hidden_size = self.receptance.in_features
        bound = hidden_size**-0.5
        nn.init.uniform_(self.receptance.weight, -0.5 * bound, 0.5 * bound)
        nn.init.uniform_(self.key.weight, -0.05 * bound, 0.05 * bound)
        nn.init.uniform_(self.value.weight, -0.5 * bound, 0.5 * bound)
        nn.init.zeros_(self.output.weight)
        for module in (self.receptance, self.key, self.value, self.output):
            module._skip_shoujen_init = True
        self.ln_x._skip_shoujen_init = True

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None = None,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        sequence_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, hidden_size = x.shape
        n_head, head_dim = self.n_head, self.head_dim

        if state is None:
            recurrent_state = None
            prev_token = x.new_zeros(bsz, 1, hidden_size)
        else:
            recurrent_state, prev_token = state
            if prev_token is None:
                prev_token = x.new_zeros(bsz, 1, hidden_size)

        x_prev = torch.cat([prev_token.to(x), x[:, :-1]], dim=1)
        if sequence_start_mask is not None:
            sequence_start_mask = sequence_start_mask.to(device=x.device, dtype=torch.bool)
            x_prev = x_prev.masked_fill(sequence_start_mask[:, :, None], 0)
        xx = x_prev - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w_raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        # Official RWKV-v7 x070 soft-clamp to (-inf, -0.5). The CUDA/Metal
        # wind-backstepping kernels apply exp(-exp(w_log)) internally.
        w_log = (-F.softplus(-w_raw.float()) - 0.5).to(x.dtype)
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_idx == 0 or v_first is None:
            v_first_out = v
        else:
            v_mix = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v = v + (v_first.to(v) - v) * v_mix
            v_first_out = v_first

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(bsz, seq_len, n_head, head_dim), dim=-1, p=2.0, eps=1e-12)
        k = k * (1.0 + (a - 1.0) * self.k_a)

        r_seq = r.view(bsz, seq_len, n_head, head_dim).contiguous()
        k_seq = k.view(bsz, seq_len, n_head, head_dim).contiguous()
        v_seq = v.view(bsz, seq_len, n_head, head_dim).contiguous()
        w_seq = w_log.view(bsz, seq_len, n_head, head_dim).contiguous()
        neg_kk_seq = (-kk).contiguous()
        kka_seq = (kk.reshape(bsz, seq_len, hidden_size) * a).view(bsz, seq_len, n_head, head_dim).contiguous()

        mps_pad_len = 0
        if recurrent_state is None:
            # Autocast leaves the elementwise RWKV tensors in mixed dtypes
            # because they interact with fp32 parameters. The Metal shader uses
            # one scalar_t for all six inputs, while still accumulating state in
            # fp32 internally, so normalize only the shader boundary.
            mps_dtype = r_seq.dtype
            mps_args = tuple(
                tensor if tensor.dtype == mps_dtype else tensor.to(dtype=mps_dtype)
                for tensor in (
                    w_seq,
                    r_seq,
                    k_seq,
                    v_seq,
                    neg_kk_seq,
                    kka_seq,
                )
            )
            mps_pad_len = (-seq_len) % 16
            if mps_pad_len:
                mps_args = tuple(F.pad(tensor, (0, 0, 0, 0, 0, mps_pad_len)) for tensor in mps_args)
        else:
            mps_args = ()
        has_internal_resets = False
        if sequence_start_mask is not None and seq_len > 1:
            has_internal_resets = bool(sequence_start_mask[:, 1:].any().detach().cpu().item())
        if (
            mps_args
            and not has_internal_resets
            and can_use_mps_wind_backstepping(*mps_args, chunk_len=16)
        ):
            out_seq = mps_wind_backstepping(
                *mps_args,
                chunk_len=16,
            )
            if mps_pad_len:
                out_seq = out_seq[:, :seq_len]
            new_recurrent_state = None
        else:
            rh = r_seq.transpose(1, 2).contiguous()
            kh = k_seq.transpose(1, 2).contiguous()
            vh = v_seq.transpose(1, 2).contiguous()
            wh = torch.exp(-torch.exp(w_seq.float())).to(x.dtype).transpose(1, 2).contiguous()
            neg_kk = neg_kk_seq.transpose(1, 2).contiguous()
            kka = kka_seq.transpose(1, 2).contiguous()
            out_h, new_recurrent_state = rwkv7_recurrent(
                rh,
                kh,
                vh,
                wh,
                neg_kk,
                kka,
                state=recurrent_state,
                reset_mask=sequence_start_mask,
            )
            out_seq = out_h.transpose(1, 2).contiguous()

        rk_bonus = (r_seq * k_seq * self.r_k.view(1, 1, n_head, head_dim)).sum(dim=-1, keepdim=True) * v_seq
        out_seq = out_seq + rk_bonus

        out = out_seq.reshape(bsz, seq_len, hidden_size)
        out = self.ln_x(out.view(bsz * seq_len, hidden_size)).view(bsz, seq_len, hidden_size)
        out = self.output(out * g)

        new_prev_token = x[:, -1:, :].detach()
        return out, v_first_out, (new_recurrent_state, new_prev_token)
