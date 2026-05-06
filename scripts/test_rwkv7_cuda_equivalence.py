"""Numerical-equivalence test for the RWKV-7 CUDA wind-backstepping op."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_shoujen_pkg = sys.modules.setdefault("shoujen", type(sys)("shoujen"))
_shoujen_pkg.__path__ = [str(REPO_ROOT / "shoujen")]
_modules_pkg = sys.modules.setdefault("shoujen.modules", type(sys)("shoujen.modules"))
_modules_pkg.__path__ = [str(REPO_ROOT / "shoujen" / "modules")]

_rwkv7_cuda = _load_module(
    "shoujen.modules.rwkv7_cuda",
    REPO_ROOT / "shoujen" / "modules" / "rwkv7_cuda.py",
)
can_use_cuda_wind_backstepping = _rwkv7_cuda.can_use_cuda_wind_backstepping
cuda_wind_backstepping = _rwkv7_cuda.cuda_wind_backstepping

_rwkv7_mps_stub = _load_module(
    "shoujen.modules.rwkv7_mps",
    REPO_ROOT / "shoujen" / "modules" / "rwkv7_mps.py",
)
sys.modules["shoujen.modules.rwkv7_cuda"] = _rwkv7_cuda
sys.modules["shoujen.modules.rwkv7_mps"] = _rwkv7_mps_stub

_rwkv7 = _load_module(
    "shoujen.modules.rwkv7",
    REPO_ROOT / "shoujen" / "modules" / "rwkv7.py",
)
rwkv7_recurrent = _rwkv7.rwkv7_recurrent
_cuda_wind_backstepping_with_resets = _rwkv7._cuda_wind_backstepping_with_resets


def _make_inputs(B: int, T: int, H: int, C: int, dtype: torch.dtype, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
            dtype=dtype,
            device=device,
        ).contiguous()

    shape = (B, T, H, C)
    w = (torch.rand(*shape, generator=g, dtype=torch.float32) * 3.0 - 3.0).to(
        dtype=dtype,
        device=device,
    ).contiguous()
    return {
        "w": w,
        "r": rand(shape, 0.5),
        "k": rand(shape, 0.5),
        "v": rand(shape, 1.0),
        "neg_kk": rand(shape, 0.3),
        "kka": rand(shape, 0.3),
    }


def _run_cuda(inp: dict[str, torch.Tensor], chunk_len: int):
    args = [
        t.detach().clone().requires_grad_(True)
        for t in (inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"])
    ]
    out = cuda_wind_backstepping(*args, chunk_len=chunk_len)
    return out, args


def _run_ref(inp: dict[str, torch.Tensor]):
    args = [
        t.detach().clone().requires_grad_(True)
        for t in (inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"])
    ]
    w_seq, r_seq, k_seq, v_seq, neg_kk_seq, kka_seq = args
    out_h, _ = rwkv7_recurrent(
        r_seq.transpose(1, 2).contiguous(),
        k_seq.transpose(1, 2).contiguous(),
        v_seq.transpose(1, 2).contiguous(),
        torch.exp(-torch.exp(w_seq.float())).to(w_seq.dtype).transpose(1, 2).contiguous(),
        neg_kk_seq.transpose(1, 2).contiguous(),
        kka_seq.transpose(1, 2).contiguous(),
        state=None,
    )
    return out_h.transpose(1, 2).contiguous(), args


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float().detach().cpu() - b.float().detach().cpu()).abs().max().item()


def run_one(dtype: torch.dtype, *, atol: float, rtol: float) -> bool:
    B, T, H, C, chunk_len = 2, 32, 4, 64, 16
    device = torch.device("cuda")
    inp = _make_inputs(B, T, H, C, dtype, device)
    if not can_use_cuda_wind_backstepping(
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"], chunk_len=chunk_len
    ):
        print(f"{dtype}: skipped by can_use_cuda_wind_backstepping")
        return True

    out_cuda, cuda_args = _run_cuda(inp, chunk_len)
    out_ref, ref_args = _run_ref(inp)

    g = torch.Generator(device="cpu").manual_seed(123)
    dy = torch.randn(B, T, H, C, generator=g, dtype=torch.float32).to(
        dtype=dtype,
        device=device,
    ).contiguous()
    out_cuda.backward(dy)
    out_ref.backward(dy)

    fwd_diff = _max_abs_diff(out_cuda, out_ref)
    grad_diffs = [_max_abs_diff(c.grad, r.grad) for c, r in zip(cuda_args, ref_args)]
    fwd_scale = max(out_ref.float().abs().max().item(), 1.0)
    grad_scales = [max(ref.grad.float().abs().max().item(), 1.0) for ref in ref_args]
    grads_finite = all(torch.isfinite(arg.grad).all().item() for arg in cuda_args)
    ok = (
        fwd_diff <= max(atol, rtol * fwd_scale)
        and grads_finite
        and all(diff <= max(atol, rtol * scale) for diff, scale in zip(grad_diffs, grad_scales))
    )
    print(f"{dtype}: forward max diff={fwd_diff:.3e}")
    print("  grad max diff=" + ", ".join(f"{d:.3e}" for d in grad_diffs))
    print(f"  equivalence: {'PASS' if ok else 'FAIL'}")
    return ok


def run_reset_segments(dtype: torch.dtype, *, atol: float, rtol: float) -> bool:
    B, T, H, C, chunk_len = 2, 64, 4, 64, 16
    device = torch.device("cuda")
    inp = _make_inputs(B, T, H, C, dtype, device, seed=77)
    reset_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    reset_mask[:, 0] = True
    reset_mask[0, 17] = True
    reset_mask[1, 9] = True
    reset_mask[1, 33] = True

    cuda_args = [
        t.detach().clone().requires_grad_(True)
        for t in (inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"])
    ]
    out_cuda = _cuda_wind_backstepping_with_resets(tuple(cuda_args), reset_mask, chunk_len=chunk_len)
    if out_cuda is None:
        print(f"{dtype}: reset segment CUDA dispatch returned None")
        return False

    ref_args = [
        t.detach().clone().requires_grad_(True)
        for t in (inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"])
    ]
    w_seq, r_seq, k_seq, v_seq, neg_kk_seq, kka_seq = ref_args
    out_ref_h, _ = rwkv7_recurrent(
        r_seq.transpose(1, 2).contiguous(),
        k_seq.transpose(1, 2).contiguous(),
        v_seq.transpose(1, 2).contiguous(),
        torch.exp(-torch.exp(w_seq.float())).to(w_seq.dtype).transpose(1, 2).contiguous(),
        neg_kk_seq.transpose(1, 2).contiguous(),
        kka_seq.transpose(1, 2).contiguous(),
        state=None,
        reset_mask=reset_mask,
    )
    out_ref = out_ref_h.transpose(1, 2).contiguous()

    g = torch.Generator(device="cpu").manual_seed(456)
    dy = torch.randn(B, T, H, C, generator=g, dtype=torch.float32).to(
        dtype=dtype,
        device=device,
    ).contiguous()
    out_cuda.backward(dy)
    out_ref.backward(dy)

    fwd_diff = _max_abs_diff(out_cuda, out_ref)
    grad_diffs = [_max_abs_diff(c.grad, r.grad) for c, r in zip(cuda_args, ref_args)]
    fwd_scale = max(out_ref.float().abs().max().item(), 1.0)
    grad_scales = [max(ref.grad.float().abs().max().item(), 1.0) for ref in ref_args]
    grads_finite = all(torch.isfinite(arg.grad).all().item() for arg in cuda_args)
    ok = (
        fwd_diff <= max(atol, rtol * fwd_scale)
        and grads_finite
        and all(diff <= max(atol, rtol * scale) for diff, scale in zip(grad_diffs, grad_scales))
    )
    print(f"{dtype} reset: forward max diff={fwd_diff:.3e}")
    print("  reset grad max diff=" + ", ".join(f"{d:.3e}" for d in grad_diffs))
    print(f"  reset equivalence: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA device not available -- aborting.")
        return 2

    checks: list[bool] = []
    checks.append(run_one(torch.float32, atol=5e-4, rtol=5e-3))
    checks.append(run_one(torch.float16, atol=5e-2, rtol=5e-2))
    if torch.cuda.is_bf16_supported():
        checks.append(run_one(torch.bfloat16, atol=5e-2, rtol=5e-2))
        checks.append(run_reset_segments(torch.bfloat16, atol=5e-2, rtol=5e-2))
    else:
        print("bfloat16 not supported on this CUDA device -- skipped")

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
