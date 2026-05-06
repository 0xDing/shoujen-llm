"""Short tinyshakespeare loss-drop benchmark: Shoujen vs GPT-2 small.

The comparison is intentionally small and repeatable:
  * character-level vocabulary built from tinyshakespeare
  * identical sampled batches for both models
  * next-token LM loss only, no checkpoint writes

By default, Shoujen uses the project's Muon+AdamW optimizer split and GPT-2
uses AdamW. Pass ``--optimizer-mode same-adamw`` to train both with AdamW.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import sys
import time
from pathlib import Path

import requests
import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.config import ShoujenConfig
from shoujen.losses import compute_lm_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.train_utils import autocast_dtype, pick_device, set_optimizer_lr, warmup_cosine_lr


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tinyshakespeare/input.txt")
    p.add_argument("--download-url", default=DATA_URL)
    p.add_argument("--out", default="runs/tinyshakespeare_compare/metrics.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--force-attention-mask", action="store_true")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--lr-min-ratio", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["auto", "fp16", "bf16", "none"], default="auto")
    p.add_argument("--optimizer-mode", choices=["project", "same-adamw"], default="project")
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--gpt2-lr", type=float, default=6e-4)
    p.add_argument("--adamw-wd", type=float, default=0.1)
    p.add_argument("--adamw-foreach", action="store_true")
    p.add_argument("--muon-lr", type=float, default=3e-4)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--muon-wd", type=float, default=0.0)
    return p.parse_args()


def ensure_data(path: Path, url: str) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def encode_text(text: str) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    chars = sorted(set(text))
    stoi = {ch: i + 2 for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return ids, stoi, chars


def make_starts(
    *,
    num_batches: int,
    batch_size: int,
    max_start: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, max_start, (batch_size,), generator=generator)
        for _ in range(num_batches)
    ]


def make_batch(
    data: torch.Tensor,
    starts: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(block_size)
    idx = starts[:, None] + offsets[None, :]
    # Keep the indexed CPU tensors alive until after the device copy completes.
    # Directly calling `.to(non_blocking=True)` on the temporary advanced-index
    # result can corrupt integer labels on MPS.
    x_cpu = data[idx]
    y_cpu = data[idx + 1]
    return x_cpu.to(device), y_cpu.to(device)


def build_shoujen(vocab_size: int, block_size: int) -> ShoujenLM:
    config = ShoujenConfig(
        vocab_size=vocab_size,
        max_seq_len=block_size,
        max_position_embeddings=block_size,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    return ShoujenLM(config)


def build_gpt2_small(vocab_size: int, block_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
        use_cache=False,
    )
    return GPT2LMHeadModel(config)


def build_adamw(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    foreach: bool | None = None,
) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "ln_" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    kwargs = {}
    if foreach is not None:
        kwargs["foreach"] = foreach
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        **kwargs,
    )


def build_optimizer_set(
    name: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> list[torch.optim.Optimizer]:
    if name == "shoujen" and args.optimizer_mode == "project":
        muon, adamw = build_optimizers(
            model,
            muon_lr=args.muon_lr,
            muon_ns_steps=args.muon_ns_steps,
            muon_wd=args.muon_wd,
            adamw_lr=args.adamw_lr,
            adamw_wd=args.adamw_wd,
            adamw_foreach=True if args.adamw_foreach else None,
        )
        return [muon, adamw]
    lr = args.gpt2_lr if name == "gpt2-small" else args.adamw_lr
    return [
        build_adamw(
            model,
            lr=lr,
            weight_decay=args.adamw_wd,
            foreach=True if args.adamw_foreach else None,
        )
    ]


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    starts: list[torch.Tensor],
    block_size: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    force_attention_mask: bool = False,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_starts in starts:
        x, y = make_batch(data, batch_starts, block_size, device)
        attention_mask = torch.ones_like(x) if force_attention_mask else None
        with autocast_context(device, amp_dtype):
            logits = model(x, attention_mask=attention_mask, use_cache=False).logits
            loss, _ = compute_lm_loss(logits.float(), y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train_one(
    *,
    name: str,
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    train_starts: list[torch.Tensor],
    eval_starts: list[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict:
    torch.manual_seed(args.seed)
    model.to(device)
    model.train()
    optimizers = build_optimizer_set(name, model, args)
    base_lrs = [[group["lr"] for group in opt.param_groups] for opt in optimizers]

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    initial_val = estimate_loss(
        model,
        val_data,
        eval_starts,
        args.block_size,
        device,
        amp_dtype,
        force_attention_mask=args.force_attention_mask,
    )
    print(f"{name}: params={params_m:.2f}M initial_val={initial_val:.4f}", flush=True)

    logs = [{"step": 0, "train_loss": None, "val_loss": initial_val, "tok_per_sec": None}]
    start_time = time.time()
    last_time = start_time
    last_tokens = 0

    for step, batch_starts in enumerate(train_starts, start=1):
        x, y = make_batch(train_data, batch_starts, args.block_size, device)
        attention_mask = torch.ones_like(x) if args.force_attention_mask else None
        if args.warmup > 0 or args.lr_min_ratio < 1.0:
            lr_mult = warmup_cosine_lr(
                step - 1,
                warmup=args.warmup,
                max_steps=args.max_steps,
                min_ratio=args.lr_min_ratio,
            )
            for opt, lrs in zip(optimizers, base_lrs):
                set_optimizer_lr(opt, lrs, lr_mult)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            logits = model(x, attention_mask=attention_mask, use_cache=False).logits
            loss, _ = compute_lm_loss(logits.float(), y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()

        last_tokens += int(x.numel())
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            now = time.time()
            tok_per_sec = last_tokens / max(now - last_time, 1e-6)
            val_loss = estimate_loss(
                model,
                val_data,
                eval_starts,
                args.block_size,
                device,
                amp_dtype,
                force_attention_mask=args.force_attention_mask,
            )
            logs.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "tok_per_sec": tok_per_sec,
                }
            )
            print(
                f"{name}: step={step:04d} train={loss.item():.4f} "
                f"val={val_loss:.4f} tok/s={tok_per_sec:.0f}",
                flush=True,
            )
            last_time = now
            last_tokens = 0

    elapsed = time.time() - start_time
    return {
        "name": name,
        "params_m": params_m,
        "initial_val_loss": initial_val,
        "final_val_loss": logs[-1]["val_loss"],
        "val_loss_delta": initial_val - logs[-1]["val_loss"],
        "elapsed_sec": elapsed,
        "logs": logs,
    }


def release_model(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device) if args.device else pick_device()
    if args.no_amp or args.amp_dtype == "none":
        amp_dtype = None
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    elif args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = autocast_dtype(device)
    text = ensure_data(Path(args.data), args.download_url)
    ids, stoi, chars = encode_text(text)
    vocab_size = len(stoi) + 2

    split = int(0.9 * len(ids))
    train_data = ids[:split]
    val_data = ids[split:]
    min_len = args.block_size + 2
    if len(train_data) < min_len or len(val_data) < min_len:
        raise SystemExit("Dataset split is too small for the requested --block-size")

    train_starts = make_starts(
        num_batches=args.max_steps,
        batch_size=args.batch_size,
        max_start=len(train_data) - args.block_size - 1,
        seed=args.seed + 1,
    )
    eval_starts = make_starts(
        num_batches=args.eval_batches,
        batch_size=args.batch_size,
        max_start=len(val_data) - args.block_size - 1,
        seed=args.seed + 2,
    )

    print(
        f"device={device} amp={amp_dtype} chars={len(chars)} vocab={vocab_size} "
        f"tokens={len(ids)} batch={args.batch_size} block={args.block_size} "
        f"steps={args.max_steps} optimizer_mode={args.optimizer_mode}",
        flush=True,
    )

    results = []
    torch.manual_seed(args.seed)
    shoujen = build_shoujen(vocab_size, args.block_size)
    results.append(
        train_one(
            name="shoujen",
            model=shoujen,
            train_data=train_data,
            val_data=val_data,
            train_starts=train_starts,
            eval_starts=eval_starts,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )
    )
    release_model(shoujen)

    torch.manual_seed(args.seed)
    gpt2 = build_gpt2_small(vocab_size, args.block_size)
    results.append(
        train_one(
            name="gpt2-small",
            model=gpt2,
            train_data=train_data,
            val_data=val_data,
            train_starts=train_starts,
            eval_starts=eval_starts,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )
    )
    release_model(gpt2)

    print("\nsummary:", flush=True)
    for result in results:
        print(
            f"{result['name']}: initial={result['initial_val_loss']:.4f} "
            f"final={result['final_val_loss']:.4f} "
            f"delta={result['val_loss_delta']:.4f} "
            f"elapsed={result['elapsed_sec']:.1f}s",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "device": str(device),
                "amp_dtype": str(amp_dtype),
                "num_chars": len(chars),
                "vocab_size": vocab_size,
                "num_tokens": len(ids),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
