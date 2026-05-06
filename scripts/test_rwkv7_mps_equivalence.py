"""Numerical-equivalence test for the RWKV-7 wind-backstepping MPS shader.

Compares `mps_wind_backstepping` (Metal) against the pure-PyTorch reference
`rwkv7_recurrent`, using the *exact same parameter mapping* the production
forward in `rwkv7.py` uses (see lines ~217-241):

    mps_wind_backstepping(w_seq,    r_seq,  k_seq, v_seq, neg_kk_seq, kka_seq)
    #                    `--MPS w   `--q    `--k   `--v   `--a        `--b
    rwkv7_recurrent(rh, kh, vh, wh=exp(-exp(w_seq)), neg_kk_h, kka_h)

The script runs three comparisons:
  1) Production-as-written: shader vs PyRef under the actual rwkv7.py call.
  2) ref_role_swap        : pass (kka, neg_kk) into PyRef's (neg_kk, kka)
                            slots so PyRef's *naming* aligns with the kernel
                            convention (a=inner projector, b=outer factor).
  3) shader_convention_ref: an inlined fp64 implementation of the SHADER's
                            recurrence used as ground truth, run on CPU.

Results we will print:
  * forward and all six grad max-abs-diffs for every config and dtype.
  * a diagnosis line indicating which of (1)/(2)/(3) the shader matches.
"""

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


# Import the two modules in isolation, so we don't trigger `shoujen/__init__.py`
# (which pulls transformers/torchvision and can fail in shared envs).
_rwkv7_mps = _load_module(
    "shoujen.modules.rwkv7_mps",
    REPO_ROOT / "shoujen" / "modules" / "rwkv7_mps.py",
)
can_use_mps_wind_backstepping = _rwkv7_mps.can_use_mps_wind_backstepping
mps_wind_backstepping = _rwkv7_mps.mps_wind_backstepping


_shoujen_pkg = sys.modules.setdefault("shoujen", type(sys)("shoujen"))
_shoujen_pkg.__path__ = [str(REPO_ROOT / "shoujen")]
_modules_pkg = sys.modules.setdefault("shoujen.modules", type(sys)("shoujen.modules"))
_modules_pkg.__path__ = [str(REPO_ROOT / "shoujen" / "modules")]
sys.modules["shoujen.modules.rwkv7_mps"] = _rwkv7_mps

_rwkv7 = _load_module(
    "shoujen.modules.rwkv7",
    REPO_ROOT / "shoujen" / "modules" / "rwkv7.py",
)
rwkv7_recurrent = _rwkv7.rwkv7_recurrent
_mps_wind_backstepping_with_resets = _rwkv7._mps_wind_backstepping_with_resets


def _make_inputs(B, T, H, C, dtype, device, seed=0):
    """Build a representative input set in [B,T,H,C] layout."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
            dtype=dtype, device=device
        ).contiguous()

    shape = (B, T, H, C)
    # w_log in roughly [-3, 0]: exp(-exp(w_log)) lives in [0.37, 0.95] -- a
    # sensible decay range that won't blow up.
    w = (torch.rand(*shape, generator=g, dtype=torch.float32) * 3.0 - 3.0).to(
        dtype=dtype, device=device
    ).contiguous()
    return {
        "w": w,
        "r": rand(shape, 0.5),
        "k": rand(shape, 0.5),
        "v": rand(shape, 1.0),
        "neg_kk": rand(shape, 0.3),
        "kka": rand(shape, 0.3),
    }


def _run_mps(inp, chunk_len=16):
    """Production-order MPS call: (w, r, k, v, neg_kk, kka)."""
    args = [t.detach().clone().requires_grad_(True) for t in (
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"]
    )]
    w, r, k, v, neg_kk, kka = args
    out = mps_wind_backstepping(w, r, k, v, neg_kk, kka, chunk_len=chunk_len)
    return out, args


def _run_pytorch(inp, role_swap: bool = False):
    """Production-order PyRef call (or arg-swapped)."""
    args = [t.detach().clone().requires_grad_(True) for t in (
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"]
    )]
    w_seq, r_seq, k_seq, v_seq, neg_kk_seq, kka_seq = args

    rh = r_seq.transpose(1, 2).contiguous()
    kh = k_seq.transpose(1, 2).contiguous()
    vh = v_seq.transpose(1, 2).contiguous()
    wh = torch.exp(-torch.exp(w_seq.float())).to(w_seq.dtype).transpose(1, 2).contiguous()
    neg_kk_h = neg_kk_seq.transpose(1, 2).contiguous()
    kka_h = kka_seq.transpose(1, 2).contiguous()

    if role_swap:
        # Feed kka into PyRef's `neg_kk` slot and vice versa.
        out_h, _ = rwkv7_recurrent(rh, kh, vh, wh, kka_h, neg_kk_h, state=None)
    else:
        out_h, _ = rwkv7_recurrent(rh, kh, vh, wh, neg_kk_h, kka_h, state=None)
    return out_h.transpose(1, 2).contiguous(), args


def _shader_convention_ref(w, r, k, v, a, b):
    """fp64 CPU reference implementing the SHADER's recurrence verbatim.

    sa = sum_j(a[j] * state[i,j])     (using PRE-decay state)
    state[i,j] = state[i,j]*w[j] + sa*b[j] + k[j]*v[i]
    out[i] = sum_j(state[i,j] * r[j])

    All inputs in [B,T,H,C]. No grad here -- this is just for forward
    correctness as a numerical-truth oracle.
    """
    w = w.detach().to("cpu").float().double()
    r = r.detach().to("cpu").float().double()
    k = k.detach().to("cpu").float().double()
    v = v.detach().to("cpu").float().double()
    a = a.detach().to("cpu").float().double()
    b = b.detach().to("cpu").float().double()

    Bn, Tn, Hn, Cn = w.shape
    state = torch.zeros(Bn, Hn, Cn, Cn, dtype=torch.float64)
    out = torch.empty(Bn, Tn, Hn, Cn, dtype=torch.float64)
    for t in range(Tn):
        wt = torch.exp(-torch.exp(w[:, t]))
        sa = (state * a[:, t].unsqueeze(-2)).sum(-1)               # PRE-decay
        state = state * wt.unsqueeze(-2) + sa.unsqueeze(-1) * b[:, t].unsqueeze(-2)
        state = state + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
        out[:, t] = (state * r[:, t].unsqueeze(-2)).sum(-1)
    return out


def _max_abs_diff(a, b):
    return (a.float().cpu() - b.float().cpu()).abs().max().item()


def _print_grad_diffs(mps_args, ref_args):
    names = ["dw", "d(r=q)", "dk", "dv", "d(neg_kk=a)", "d(kka=b)"]
    diffs = {}
    for name, gm, gr in zip(names, mps_args, ref_args):
        d = _max_abs_diff(gm.grad, gr.grad)
        scale = gr.grad.float().abs().mean().item()
        diffs[name] = d
        print(f"    {name:<18} max|diff|={d:.3e}  grad_mean={scale:.3e}")
    return diffs


def _scenario_grad(mps_args, ref_args):
    """Return list of grads in zip order from mps_args and ref_args."""
    return list(zip(mps_args, ref_args))


def run_one(B, T, H, C, dtype, device, chunk_len=16, atol=5e-4, rtol=5e-3):
    print(f"\n{'='*72}")
    print(f"Config: B={B} T={T} H={H} C={C} chunk_len={chunk_len} dtype={dtype}")
    print(f"{'='*72}")

    inp = _make_inputs(B, T, H, C, dtype, device)
    if not can_use_mps_wind_backstepping(
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"],
        chunk_len=chunk_len,
    ):
        print("  [skip] can_use_mps_wind_backstepping returned False")
        return None

    # Run shader once.
    out_mps, mps_args = _run_mps(inp, chunk_len=chunk_len)
    # Run pyref two ways.
    out_ref_prod, ref_args_prod = _run_pytorch(inp, role_swap=False)
    out_ref_swap, ref_args_swap = _run_pytorch(inp, role_swap=True)
    # fp64 oracle (forward only).
    out_oracle = _shader_convention_ref(
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"]
    ).to(out_mps.dtype).to(out_mps.device)

    out_mean_oracle = out_oracle.float().abs().mean().item()
    print(f"  forward magnitudes  oracle_mean={out_mean_oracle:.3e}")
    print(f"  --- forward max|diff| ---")
    print(f"    mps    vs oracle (fp64-shader-conv) : {_max_abs_diff(out_mps, out_oracle):.3e}")
    print(f"    mps    vs pyref (production order)  : {_max_abs_diff(out_mps, out_ref_prod):.3e}")
    print(f"    mps    vs pyref (role-swap)         : {_max_abs_diff(out_mps, out_ref_swap):.3e}")
    print(f"    pyref  vs oracle (production order) : {_max_abs_diff(out_ref_prod, out_oracle):.3e}")
    print(f"    pyref  vs oracle (role-swap)        : {_max_abs_diff(out_ref_swap, out_oracle):.3e}")

    # Backward only against pyref (production and swap).
    g = torch.Generator(device="cpu").manual_seed(123)
    dy = torch.randn(B, T, H, C, generator=g, dtype=torch.float32).to(
        dtype=dtype, device=device
    ).contiguous()

    # Need a fresh shader run so its grad isn't shared between two backward()
    # calls; the previous run already has saved tensors so we backward once.
    out_mps.backward(dy)
    out_ref_prod.backward(dy)
    out_ref_swap.backward(dy)

    print(f"  --- backward max|diff| (mps vs pyref-production) ---")
    diffs_prod = _print_grad_diffs(mps_args, ref_args_prod)
    print(f"  --- backward max|diff| (mps vs pyref-role-swap) ---")
    diffs_swap = _print_grad_diffs(mps_args, ref_args_swap)

    fwd_scale = max(out_ref_prod.float().abs().max().item(), 1.0)
    grad_scales = [
        max(ref.grad.float().abs().max().item(), 1.0)
        for ref in ref_args_prod
    ]
    grads_finite = all(torch.isfinite(arg.grad).all().item() for arg in mps_args)
    fwd_ok = _max_abs_diff(out_mps, out_ref_prod) <= max(atol, rtol * fwd_scale)
    grads_ok = grads_finite and all(
        diff <= max(atol, rtol * scale)
        for diff, scale in zip(diffs_prod.values(), grad_scales)
    )
    return {
        "fwd_mps_vs_oracle": _max_abs_diff(out_mps, out_oracle),
        "fwd_mps_vs_prod":   _max_abs_diff(out_mps, out_ref_prod),
        "fwd_mps_vs_swap":   _max_abs_diff(out_mps, out_ref_swap),
        "diffs_prod": diffs_prod,
        "diffs_swap": diffs_swap,
        "grads_finite": grads_finite,
        "production_pass": fwd_ok and grads_ok,
    }


def run_reset_segments(dtype, device, atol=5e-2, rtol=5e-2) -> bool:
    B, T, H, C, chunk_len = 2, 64, 4, 64, 16
    print(f"\n{'='*72}")
    print(f"Reset-packed dispatch: B={B} T={T} H={H} C={C} dtype={dtype}")
    print(f"{'='*72}")

    inp = _make_inputs(B, T, H, C, dtype, device, seed=77)
    reset_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    reset_mask[:, 0] = True
    reset_mask[0, 17] = True
    reset_mask[1, 9] = True
    reset_mask[1, 33] = True

    mps_args = [t.detach().clone().requires_grad_(True) for t in (
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"]
    )]
    out_mps = _mps_wind_backstepping_with_resets(tuple(mps_args), reset_mask, chunk_len=chunk_len)
    if out_mps is None:
        print("  reset-packed MPS dispatch returned None")
        return False

    ref_args = [t.detach().clone().requires_grad_(True) for t in (
        inp["w"], inp["r"], inp["k"], inp["v"], inp["neg_kk"], inp["kka"]
    )]
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
    dy = torch.randn(B, T, H, C, generator=g, dtype=torch.float32).to(dtype=dtype, device=device).contiguous()
    out_mps.backward(dy)
    out_ref.backward(dy)

    fwd_diff = _max_abs_diff(out_mps, out_ref)
    grad_diffs = [_max_abs_diff(m.grad, r.grad) for m, r in zip(mps_args, ref_args)]
    fwd_scale = max(out_ref.float().abs().max().item(), 1.0)
    grad_scales = [max(ref.grad.float().abs().max().item(), 1.0) for ref in ref_args]
    grads_finite = all(torch.isfinite(arg.grad).all().item() for arg in mps_args)
    ok = (
        fwd_diff <= max(atol, rtol * fwd_scale)
        and grads_finite
        and all(diff <= max(atol, rtol * scale) for diff, scale in zip(grad_diffs, grad_scales))
    )
    print(f"  forward max|diff|={fwd_diff:.3e}")
    print("  grad max|diff|=" + ", ".join(f"{d:.3e}" for d in grad_diffs))
    print(f"  reset-packed equivalence : {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.backends.mps.is_available():
        print("MPS device not available -- aborting.")
        return 2
    if not hasattr(torch.mps, "compile_shader"):
        print("torch.mps.compile_shader missing -- need a newer torch with Metal shader compile.")
        return 2

    device = torch.device("mps")

    configs = [
        # (B, T, H, C, chunk_len)
        (2, 32, 4, 64, 16),
        (1, 16, 2, 32, 16),
        (2, 64, 4, 64, 16),
    ]
    dtype_tols = [
        # dtype, atol (forward/grad), rtol
        (torch.float32, 5e-4, 5e-3),
        (torch.float16, 5e-2, 5e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ]

    summary = []
    reset_passes = []
    for dtype, atol, rtol in dtype_tols:
        print(f"\n\n############# dtype = {dtype} (atol={atol}, rtol={rtol}) #############")
        for (B, T, H, C, cl) in configs:
            r = run_one(B, T, H, C, dtype, device, chunk_len=cl, atol=atol, rtol=rtol)
            if r is not None:
                summary.append((dtype, B, T, H, C, r))
        if dtype in {torch.float32, torch.float16}:
            reset_passes.append(run_reset_segments(dtype, device, atol=atol, rtol=rtol))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'dtype':<14}{'B,T,H,C':<14}{'mps↔oracle':<14}{'mps↔prod':<14}{'mps↔swap':<14}")
    for dtype, B, T, H, C, r in summary:
        print(f"{str(dtype):<14}{f'{B},{T},{H},{C}':<14}"
              f"{r['fwd_mps_vs_oracle']:<14.3e}{r['fwd_mps_vs_prod']:<14.3e}"
              f"{r['fwd_mps_vs_swap']:<14.3e}")

    all_prod_pass = all(r["production_pass"] for *_, r in summary) and all(reset_passes)

    print("\n" + "=" * 72)
    print(f"PRODUCTION-DISPATCH equivalence : {'PASS' if all_prod_pass else 'FAIL'}")
    if not all_prod_pass:
        print("\nDiagnosis:")
        print("  Production dispatch should match the fp64 shader-convention oracle")
        print("  and the PyTorch reference within dtype-scaled tolerances.")
        print("  Check the per-gradient max|diff| rows above; non-finite MPS grads")
        print("  are treated as a failure, especially for fp16/bfloat16 backward.")
    print("=" * 72)
    return 0 if all_prod_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
