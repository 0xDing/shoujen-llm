"""CUDA implementation of the RWKV-7 wind-backstepping op.

This is a runtime-compiled wrapper adapted from the Apache-2.0 reference
RWKV-v7 CUDA kernel:
https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp/cuda

The CUDA path is intentionally narrow and falls back to the transparent
PyTorch recurrence for unsupported cases:
  * CUDA tensors only
  * float32 / float16 / bfloat16 tensors
  * contiguous inputs shaped [B, T, H, C]
  * T divisible by chunk_len after caller-side padding
  * zero initial recurrent state
"""

from __future__ import annotations

from functools import lru_cache
import os

import torch


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void cuda_forward(
    int B,
    int T,
    int H,
    void* w,
    void* q,
    void* k,
    void* v,
    void* a,
    void* b,
    void* y,
    float* s,
    float* sa
);

void cuda_backward(
    int B,
    int T,
    int H,
    void* w,
    void* q,
    void* k,
    void* v,
    void* a,
    void* b,
    void* dy,
    float* s,
    float* sa,
    void* dw,
    void* dq,
    void* dk,
    void* dv,
    void* da,
    void* db
);

void forward(
    torch::Tensor w,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y,
    torch::Tensor s,
    torch::Tensor sa
) {
    const c10::cuda::CUDAGuard device_guard(w.device());
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(
        B,
        T,
        H,
        w.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        y.data_ptr(),
        static_cast<float*>(s.data_ptr()),
        static_cast<float*>(sa.data_ptr())
    );
}

void backward(
    torch::Tensor w,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor dy,
    torch::Tensor s,
    torch::Tensor sa,
    torch::Tensor dw,
    torch::Tensor dq,
    torch::Tensor dk,
    torch::Tensor dv,
    torch::Tensor da,
    torch::Tensor db
) {
    const c10::cuda::CUDAGuard device_guard(w.device());
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(
        B,
        T,
        H,
        w.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        dy.data_ptr(),
        static_cast<float*>(s.data_ptr()),
        static_cast<float*>(sa.data_ptr()),
        dw.data_ptr(),
        dq.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        da.data_ptr(),
        db.data_ptr()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "RWKV7 wind-backstepping forward");
    m.def("backward", &backward, "RWKV7 wind-backstepping backward");
}
"""


_CUDA_SOURCE_TEMPLATE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <assert.h>

{scalar_type}

__device__ inline float to_float(const scalar_t& u) {{
    {to_float}
}}

__device__ inline scalar_t to_scalar(const float& u) {{
    {to_scalar}
}}

typedef scalar_t* __restrict__ F_;

__global__ void forward_kernel(
    int T,
    int H,
    F_ w_,
    F_ q_,
    F_ k_,
    F_ v_,
    F_ a_,
    F_ b_,
    scalar_t* y_,
    float* s_,
    float* sa_
) {{
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float state[C] = {{0}};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {{
        int ind = bb * T * H * C + t * H * C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0.0f;
#pragma unroll
        for (int j = 0; j < C; j++) {{
            sa += a[j] * state[j];
        }}
        sa_[ind] = sa;

        float vi = to_float(v_[ind]);
        float yi = 0.0f;
#pragma unroll
        for (int j = 0; j < C; j++) {{
            float& state_ij = state[j];
            state_ij = state_ij * w[j] + sa * b[j] + k[j] * vi;
            yi += state_ij * q[j];
        }}
        y_[ind] = to_scalar(yi);

        if ((t + 1) % _CHUNK_LEN_ == 0) {{
            int base = (bb * H + hh) * (T / _CHUNK_LEN_) * C * C
                + (t / _CHUNK_LEN_) * C * C
                + i;
#pragma unroll
            for (int j = 0; j < C; j++) {{
                s_[base + j * C] = state[j];
            }}
        }}
    }}
}}

__global__ void backward_kernel(
    int T,
    int H,
    F_ w_,
    F_ q_,
    F_ k_,
    F_ v_,
    F_ a_,
    F_ b_,
    F_ dy_,
    float* __restrict__ s_,
    float* __restrict__ sa_,
    scalar_t* dw_,
    scalar_t* dq_,
    scalar_t* dk_,
    scalar_t* dv_,
    scalar_t* da_,
    scalar_t* db_
) {{
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {{0}}, dstate[C] = {{0}}, dstateT[C] = {{0}};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T - 1; t >= 0; t--) {{
        int ind = bb * T * H * C + t * H * C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t + 1) % _CHUNK_LEN_ == 0) {{
            int base = (bb * H + hh) * (T / _CHUNK_LEN_) * C * C
                + (t / _CHUNK_LEN_) * C * C
                + i * C;
#pragma unroll
            for (int j = 0; j < C; j++) {{
                stateT[j] = s_[base + j];
            }}
        }}

        float dqi = 0.0f;
#pragma unroll
        for (int j = 0; j < C; j++) {{
            dqi += stateT[j] * dy[j];
        }}
        dq_[ind] = to_scalar(dqi);

        float iwi = 1.0f / wi;
#pragma unroll
        for (int j = 0; j < C; j++) {{
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }}

        float dwi = 0.0f, dki = 0.0f, dvi = 0.0f, dbi = 0.0f, dSb = 0.0f;
#pragma unroll
        for (int j = 0; j < C; j++) {{
            dwi += dstateT[j] * stateT[j];
            dki += dstateT[j] * v[j];
            dvi += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            dbi += dstateT[j] * sa[j];
        }}
        dw_[ind] = to_scalar(dwi * wi * wi_fac);
        dk_[ind] = to_scalar(dki);
        dv_[ind] = to_scalar(dvi);
        db_[ind] = to_scalar(dbi);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float dai = 0.0f;
#pragma unroll
        for (int j = 0; j < C; j++) {{
            dai += stateT[j] * dSb_shared[j];
        }}
        da_[ind] = to_scalar(dai);

#pragma unroll
        for (int j = 0; j < C; j++) {{
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
        }}
    }}
}}

void cuda_forward(
    int B,
    int T,
    int H,
    void* w,
    void* q,
    void* k,
    void* v,
    void* a,
    void* b,
    void* y,
    float* s,
    float* sa
) {{
    auto stream = at::cuda::getCurrentCUDAStream();
    forward_kernel<<<dim3(H, B), dim3(_C_), 0, stream>>>(
        T,
        H,
        static_cast<F_>(w),
        static_cast<F_>(q),
        static_cast<F_>(k),
        static_cast<F_>(v),
        static_cast<F_>(a),
        static_cast<F_>(b),
        static_cast<scalar_t*>(y),
        s,
        sa
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

void cuda_backward(
    int B,
    int T,
    int H,
    void* w,
    void* q,
    void* k,
    void* v,
    void* a,
    void* b,
    void* dy,
    float* s,
    float* sa,
    void* dw,
    void* dq,
    void* dk,
    void* dv,
    void* da,
    void* db
) {{
    assert(T % _CHUNK_LEN_ == 0);
    auto stream = at::cuda::getCurrentCUDAStream();
    backward_kernel<<<dim3(H, B), dim3(_C_), 0, stream>>>(
        T,
        H,
        static_cast<F_>(w),
        static_cast<F_>(q),
        static_cast<F_>(k),
        static_cast<F_>(v),
        static_cast<F_>(a),
        static_cast<F_>(b),
        static_cast<F_>(dy),
        s,
        sa,
        static_cast<scalar_t*>(dw),
        static_cast<scalar_t*>(dq),
        static_cast<scalar_t*>(dk),
        static_cast<scalar_t*>(dv),
        static_cast<scalar_t*>(da),
        static_cast<scalar_t*>(db)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""


def can_use_cuda_wind_backstepping(
    *tensors: torch.Tensor,
    chunk_len: int,
) -> bool:
    if os.environ.get("SHOUJEN_RWKV7_CUDA", "1") in {"0", "false", "False", "off", "OFF"}:
        return False
    if not torch.cuda.is_available():
        return False
    if not tensors:
        return False
    first = tensors[0]
    valid_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    if first.device.type != "cuda" or first.dtype not in valid_dtypes or first.ndim != 4:
        return False
    if first.shape[1] % chunk_len != 0:
        return False
    if first.shape[-1] <= 0 or first.shape[-1] > 256:
        return False
    return all(
        t.device.type == "cuda"
        and t.dtype == first.dtype
        and t.shape == first.shape
        and t.is_contiguous()
        for t in tensors
    )


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise TypeError(f"Unsupported CUDA wind-backstepping dtype: {dtype}")


def _cuda_source(dtype_name: str) -> str:
    if dtype_name == "fp32":
        scalar_type = "using scalar_t = float;"
        to_float = "return u;"
        to_scalar = "return u;"
    elif dtype_name == "fp16":
        scalar_type = "using scalar_t = half;"
        to_float = "return __half2float(u);"
        to_scalar = "return __float2half_rn(u);"
    elif dtype_name == "bf16":
        scalar_type = "using scalar_t = __nv_bfloat16;"
        to_float = "return __bfloat162float(u);"
        to_scalar = "return __float2bfloat16_rn(u);"
    else:
        raise TypeError(f"Unsupported CUDA wind-backstepping dtype name: {dtype_name}")
    return _CUDA_SOURCE_TEMPLATE.format(
        scalar_type=scalar_type,
        to_float=to_float,
        to_scalar=to_scalar,
    )


@lru_cache(maxsize=24)
def _load_cuda_extension(head_dim: int, chunk_len: int, dtype_name: str):
    from torch.utils.cpp_extension import load_inline

    name = f"shoujen_rwkv7_wind_{dtype_name}_c{head_dim}_cl{chunk_len}"
    flags = [
        f"-D_C_={head_dim}",
        f"-D_CHUNK_LEN_={chunk_len}",
        "--use_fast_math",
        "-O3",
        "-Xptxas=-O3",
        "--extra-device-vectorization",
    ]
    return load_inline(
        name=name,
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_cuda_source(dtype_name)],
        functions=None,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=flags,
        verbose=os.environ.get("SHOUJEN_RWKV7_CUDA_VERBOSE", "0") in {"1", "true", "True", "on"},
    )


class CUDAWindBackstepping(torch.autograd.Function):
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
        if not can_use_cuda_wind_backstepping(w, q, k, v, a, b, chunk_len=chunk_len):
            raise RuntimeError("CUDA wind-backstepping requires contiguous CUDA tensors [B,T,H,C]")

        B, T, H, C = w.shape
        ext = _load_cuda_extension(C, chunk_len, _dtype_name(w.dtype))
        y = torch.empty_like(v)
        s = torch.empty((B, H, T // chunk_len, C, C), device=w.device, dtype=torch.float32)
        sa = torch.empty(w.shape, device=w.device, dtype=torch.float32)
        ext.forward(w, q, k, v, a, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, a, b, s, sa)
        ctx.chunk_len = chunk_len
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        w, q, k, v, a, b, s, sa = ctx.saved_tensors
        B, T, H, C = w.shape
        ext = _load_cuda_extension(C, ctx.chunk_len, _dtype_name(w.dtype))
        dy = dy.contiguous()
        dw = torch.empty_like(w)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        ext.backward(w, q, k, v, a, b, dy, s, sa, dw, dq, dk, dv, da, db)
        return dw, dq, dk, dv, da, db, None


def cuda_wind_backstepping(
    w: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    chunk_len: int = 16,
) -> torch.Tensor:
    return CUDAWindBackstepping.apply(w, q, k, v, a, b, chunk_len)
