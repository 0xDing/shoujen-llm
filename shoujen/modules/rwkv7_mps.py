"""Metal implementation of the RWKV-7 wind-backstepping op.

This mirrors the official CUDA bf16/fp32 wind-backstepping op from:
https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp/cuda

The first version is intentionally narrow:
  * MPS only
  * float32 / float16 / bfloat16 tensors
  * contiguous inputs shaped [B, T, H, C]
  * T divisible by chunk_len

Other cases should fall back to the transparent PyTorch implementation.
"""

from __future__ import annotations

from functools import lru_cache
import os

import torch


def can_use_mps_wind_backstepping(
    *tensors: torch.Tensor,
    chunk_len: int,
) -> bool:
    if os.environ.get("SHOUJEN_RWKV7_MPS", "1") in {"0", "false", "False", "off", "OFF"}:
        return False
    if not torch.backends.mps.is_available():
        return False
    if not hasattr(torch.mps, "compile_shader"):
        return False
    if not tensors:
        return False
    first = tensors[0]
    valid_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    if first.device.type != "mps" or first.dtype not in valid_dtypes or first.ndim != 4:
        return False
    if first.shape[1] % chunk_len != 0:
        return False
    if first.shape[-1] <= 0 or first.shape[-1] > 256:
        return False
    return all(
        t.device.type == "mps"
        and t.dtype == first.dtype
        and t.shape == first.shape
        and t.is_contiguous()
        for t in tensors
    )


@lru_cache(maxsize=16)
def _compile_wind_backstepping_shader(head_dim: int, chunk_len: int, metal_dtype: str):
    source = f"""
#include <metal_stdlib>
using namespace metal;

constant uint C = {head_dim};
constant uint CHUNK_LEN = {chunk_len};
using scalar_t = {metal_dtype};

kernel void rwkv7_forward(
    device const scalar_t* w_ [[buffer(0)]],
    device const scalar_t* q_ [[buffer(1)]],
    device const scalar_t* k_ [[buffer(2)]],
    device const scalar_t* v_ [[buffer(3)]],
    device const scalar_t* a_ [[buffer(4)]],
    device const scalar_t* b_ [[buffer(5)]],
    device scalar_t* y_ [[buffer(6)]],
    device float* s_ [[buffer(7)]],
    device float* sa_ [[buffer(8)]],
    constant int& T_const [[buffer(9)]],
    constant int& H_const [[buffer(10)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threadgroup_position_in_grid]]
) {{
    const uint i = tid.x;
    const uint hh = tpg.y;
    const uint bb = tpg.z;
    const uint T = uint(T_const);
    const uint H = uint(H_const);

    float state[{head_dim}];
    for (uint j = 0; j < C; ++j) {{
        state[j] = 0.0f;
    }}

    threadgroup float q[{head_dim}];
    threadgroup float k[{head_dim}];
    threadgroup float w[{head_dim}];
    threadgroup float a[{head_dim}];
    threadgroup float b[{head_dim}];

    for (uint t = 0; t < T; ++t) {{
        const uint ind = ((bb * T + t) * H + hh) * C + i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        q[i] = float(q_[ind]);
        w[i] = metal::exp(-metal::exp(float(w_[ind])));
        k[i] = float(k_[ind]);
        a[i] = float(a_[ind]);
        b[i] = float(b_[ind]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sa = 0.0f;
        for (uint j = 0; j < C; ++j) {{
            sa += a[j] * state[j];
        }}
        sa_[ind] = sa;

        const float vi = float(v_[ind]);
        float yi = 0.0f;
        for (uint j = 0; j < C; ++j) {{
            state[j] = state[j] * w[j] + sa * b[j] + k[j] * vi;
            yi += state[j] * q[j];
        }}
        y_[ind] = scalar_t(yi);

        if (((t + 1) % CHUNK_LEN) == 0) {{
            const uint base = ((bb * H + hh) * (T / CHUNK_LEN) + (t / CHUNK_LEN)) * C * C + i;
            for (uint j = 0; j < C; ++j) {{
                s_[base + j * C] = state[j];
            }}
        }}
    }}
}}

kernel void rwkv7_backward(
    device const scalar_t* w_ [[buffer(0)]],
    device const scalar_t* q_ [[buffer(1)]],
    device const scalar_t* k_ [[buffer(2)]],
    device const scalar_t* v_ [[buffer(3)]],
    device const scalar_t* a_ [[buffer(4)]],
    device const scalar_t* b_ [[buffer(5)]],
    device const scalar_t* dy_ [[buffer(6)]],
    device const float* s_ [[buffer(7)]],
    device const float* sa_ [[buffer(8)]],
    device scalar_t* dw_ [[buffer(9)]],
    device scalar_t* dq_ [[buffer(10)]],
    device scalar_t* dk_ [[buffer(11)]],
    device scalar_t* dv_ [[buffer(12)]],
    device scalar_t* da_ [[buffer(13)]],
    device scalar_t* db_ [[buffer(14)]],
    constant int& T_const [[buffer(15)]],
    constant int& H_const [[buffer(16)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threadgroup_position_in_grid]]
) {{
    const uint i = tid.x;
    const uint hh = tpg.y;
    const uint bb = tpg.z;
    const uint T = uint(T_const);
    const uint H = uint(H_const);

    float stateT[{head_dim}];
    float dstate[{head_dim}];
    float dstateT[{head_dim}];
    for (uint j = 0; j < C; ++j) {{
        stateT[j] = 0.0f;
        dstate[j] = 0.0f;
        dstateT[j] = 0.0f;
    }}

    threadgroup float w[{head_dim}];
    threadgroup float q[{head_dim}];
    threadgroup float k[{head_dim}];
    threadgroup float v[{head_dim}];
    threadgroup float a[{head_dim}];
    threadgroup float b[{head_dim}];
    threadgroup float dy[{head_dim}];
    threadgroup float sa[{head_dim}];
    threadgroup float dSb_shared[{head_dim}];

    for (int tt = int(T) - 1; tt >= 0; --tt) {{
        const uint t = uint(tt);
        const uint ind = ((bb * T + t) * H + hh) * C + i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float qi = float(q_[ind]);
        const float wi_fac = -metal::exp(float(w_[ind]));
        const float wi = metal::exp(wi_fac);
        const float ki = float(k_[ind]);
        const float ai = float(a_[ind]);
        const float bi = float(b_[ind]);
        const float dyi = float(dy_[ind]);

        q[i] = qi;
        w[i] = wi;
        k[i] = ki;
        a[i] = ai;
        b[i] = bi;
        v[i] = float(v_[ind]);
        dy[i] = dyi;
        sa[i] = sa_[ind];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (((t + 1) % CHUNK_LEN) == 0) {{
            const uint base = ((bb * H + hh) * (T / CHUNK_LEN) + (t / CHUNK_LEN)) * C * C + i * C;
            for (uint j = 0; j < C; ++j) {{
                stateT[j] = s_[base + j];
            }}
        }}

        float dqi = 0.0f;
        for (uint j = 0; j < C; ++j) {{
            dqi += stateT[j] * dy[j];
        }}
        dq_[ind] = scalar_t(dqi);

        const float iwi = 1.0f / wi;
        for (uint j = 0; j < C; ++j) {{
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }}

        float dwi = 0.0f;
        float dki = 0.0f;
        float dvi = 0.0f;
        float dbi = 0.0f;
        float dSb = 0.0f;
        for (uint j = 0; j < C; ++j) {{
            dwi += dstateT[j] * stateT[j];
            dki += dstateT[j] * v[j];
            dvi += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            dbi += dstateT[j] * sa[j];
        }}
        dw_[ind] = scalar_t(dwi * wi * wi_fac);
        dk_[ind] = scalar_t(dki);
        dv_[ind] = scalar_t(dvi);
        db_[ind] = scalar_t(dbi);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        dSb_shared[i] = dSb;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float dai = 0.0f;
        for (uint j = 0; j < C; ++j) {{
            dai += stateT[j] * dSb_shared[j];
        }}
        da_[ind] = scalar_t(dai);

        for (uint j = 0; j < C; ++j) {{
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }}
    }}
}}
"""
    return torch.mps.compile_shader(source)


def _metal_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float16:
        return "half"
    if dtype == torch.bfloat16:
        return "bfloat"
    raise TypeError(f"Unsupported MPS wind-backstepping dtype: {dtype}")


class MPSWindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        chunk_len: int = 16,
    ) -> torch.Tensor:
        if not can_use_mps_wind_backstepping(w, q, k, v, a, b, chunk_len=chunk_len):
            raise RuntimeError("MPS wind-backstepping requires contiguous fp32 MPS tensors [B,T,H,C]")

        B, T, H, C = w.shape
        lib = _compile_wind_backstepping_shader(C, chunk_len, _metal_dtype(w.dtype))
        y = torch.empty_like(v)
        s = torch.empty((B, H, T // chunk_len, C, C), device=w.device, dtype=torch.float32)
        sa = torch.empty_like(w)
        lib.rwkv7_forward(
            w,
            q,
            k,
            v,
            a,
            b,
            y,
            s,
            sa,
            T,
            H,
            threads=(C, H, B),
            group_size=(C, 1, 1),
        )
        ctx.save_for_backward(w, q, k, v, a, b, s, sa)
        ctx.chunk_len = chunk_len
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        w, q, k, v, a, b, s, sa = ctx.saved_tensors
        B, T, H, C = w.shape
        lib = _compile_wind_backstepping_shader(C, ctx.chunk_len, _metal_dtype(w.dtype))
        dy = dy.contiguous()
        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        lib.rwkv7_backward(
            w,
            q,
            k,
            v,
            a,
            b,
            dy,
            s,
            sa,
            dw,
            dq,
            dk,
            dv,
            da,
            db,
            T,
            H,
            threads=(C, H, B),
            group_size=(C, 1, 1),
        )
        return dw, dq, dk, dv, da, db, None


def mps_wind_backstepping(
    w: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    chunk_len: int = 16,
) -> torch.Tensor:
    return MPSWindBackstepping.apply(w, q, k, v, a, b, chunk_len)
