"""Probe fp16 MPS training capacity for Shoujen.

The probe runs real forward/backward/optimizer steps on synthetic causal-LM
batches so results are not dominated by parquet streaming or tokenizer work.
It is intended to choose a per-device micro-batch size before the formal
Trainer run, then decide whether gradient accumulation or checkpointing is
needed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_config
from shoujen.data import build_packed_causal_mask
from shoujen.losses import compute_lm_loss, compute_z_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer
from shoujen.train_utils import autocast_dtype, pick_device


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_bool_list(raw: str) -> list[bool]:
    out: list[bool] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"1", "true", "yes", "on"}:
            out.append(True)
        elif item in {"0", "false", "no", "off"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected bool, got {item!r}")
    if not out:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vocab",
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="Hugging Face tokenizer id/URL or legacy local vocab.json",
    )
    p.add_argument("--config", help="JSON file overriding default model config")
    p.add_argument("--output", type=Path, default=Path("runs/mps-batch-probe/results.json"))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1,2,4,8"))
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument(
        "--packed-segment-length",
        type=int,
        default=0,
        help="If positive, simulate packed documents with this approximate segment length",
    )
    p.add_argument("--gradient-checkpointing", type=parse_bool_list, default=parse_bool_list("false,true"))
    p.add_argument("--gradient-accumulation-steps", type=parse_int_list, default=parse_int_list("1"))
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--muon-lr", type=float, default=3e-4)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--muon-adaptive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--muon-adaptive-beta2", type=float, default=0.95)
    p.add_argument("--muon-adaptive-eps", type=float, default=1e-8)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--muon-wd", type=float, default=0.0)
    p.add_argument("--adamw-wd", type=float, default=0.1)
    p.add_argument("--adamw-independent-wd", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adamw-embed-wd", type=float, default=0.0)
    p.add_argument("--adamw-foreach", action="store_true")
    p.add_argument("--shuffle-order", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def clear_device_cache() -> None:
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mps_memory() -> dict[str, int | None]:
    if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
        return {"current_allocated": None, "driver_allocated": None, "recommended_max": None}
    torch.mps.synchronize()
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
        "recommended_max": int(torch.mps.recommended_max_memory()),
    }


def make_batch(
    *,
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids_cpu = torch.randint(4, vocab_size, (batch_size, block_size + 1), generator=generator, dtype=torch.long)
    return {
        "input_ids": ids_cpu[:, :-1].to(device),
        "labels": ids_cpu[:, 1:].to(device),
        "loss_mask": torch.ones(batch_size, block_size, dtype=torch.float32, device=device),
    }


def make_packed_like_batch(
    *,
    batch_size: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
    segment_length: int,
) -> dict[str, torch.Tensor]:
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids_cpu = torch.randint(4, vocab_size, (batch_size, block_size + 1), generator=generator, dtype=torch.long)
    seq_ids = torch.empty(batch_size, block_size + 1, dtype=torch.long)
    position_ids = torch.empty(batch_size, block_size + 1, dtype=torch.long)
    sequence_start_mask = torch.zeros(batch_size, block_size + 1, dtype=torch.bool)

    for batch_idx in range(batch_size):
        start = 0
        seq_id = 0
        while start < block_size + 1:
            # Stagger boundaries across batch rows so the mask is not identical
            # everywhere while keeping the mean segment length deterministic.
            jitter = ((batch_idx + seq_id) % 3) - 1
            length = max(2, segment_length + jitter * max(1, segment_length // 8))
            end = min(block_size + 1, start + length)
            seq_ids[batch_idx, start:end] = seq_id
            position_ids[batch_idx, start:end] = torch.arange(end - start, dtype=torch.long)
            sequence_start_mask[batch_idx, start] = True
            start = end
            seq_id += 1

    input_seq_ids = seq_ids[:, :-1]
    label_seq_ids = seq_ids[:, 1:]
    same_sequence_target = input_seq_ids == label_seq_ids
    labels = ids_cpu[:, 1:].clone()
    labels = torch.where(same_sequence_target, labels, torch.full_like(labels, -100))
    input_seq_ids = input_seq_ids.to(device)
    return {
        "input_ids": ids_cpu[:, :-1].to(device),
        "labels": labels.to(device),
        "loss_mask": same_sequence_target.float().to(device),
        "position_ids": position_ids[:, :-1].to(device),
        "sequence_start_mask": sequence_start_mask[:, :-1].to(device),
        "attention_mask": build_packed_causal_mask(input_seq_ids),
    }


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def run_case(
    *,
    args: argparse.Namespace,
    tokenizer: ShoujenTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    gradient_checkpointing: bool,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    clear_device_cache()
    torch.manual_seed(args.seed)

    config = build_config(args, tokenizer)
    config.use_cache = False
    model = ShoujenLM(config).to(device).train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    muon, adamw = build_optimizers(
        model,
        muon_lr=args.muon_lr,
        muon_ns_steps=args.muon_ns_steps,
        muon_wd=args.muon_wd,
        muon_adaptive=args.muon_adaptive,
        muon_adaptive_beta2=args.muon_adaptive_beta2,
        muon_adaptive_eps=args.muon_adaptive_eps,
        adamw_lr=args.adamw_lr,
        adamw_wd=args.adamw_wd,
        adamw_embed_wd=args.adamw_embed_wd if args.adamw_independent_wd else None,
        adamw_foreach=True if args.adamw_foreach else None,
    )
    batch_seed = args.seed + batch_size + 100 * int(gradient_checkpointing)
    if args.packed_segment_length:
        batch = make_packed_like_batch(
            batch_size=batch_size,
            block_size=args.block_size,
            vocab_size=tokenizer.vocab_size,
            device=device,
            seed=batch_seed,
            segment_length=args.packed_segment_length,
        )
    else:
        batch = make_batch(
            batch_size=batch_size,
            block_size=args.block_size,
            vocab_size=tokenizer.vocab_size,
            device=device,
            seed=batch_seed,
        )

    measured_times: list[float] = []
    measured_tokens = 0
    max_current = 0
    max_driver = 0
    last_loss = None
    last_grad_norm = None
    total_steps = args.warmup_steps + args.steps
    start_all = time.time()

    for step in range(total_steps):
        step_start = time.time()
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    position_ids=batch.get("position_ids"),
                    sequence_start_mask=batch.get("sequence_start_mask"),
                    use_cache=False,
                )
                lm_loss, _ = compute_lm_loss(
                    outputs.logits.float(),
                    batch["labels"],
                    loss_mask=batch["loss_mask"],
                    ignore_index=-100,
                )
                loss = lm_loss
                if args.z_loss_weight:
                    z_loss = compute_z_loss(
                        outputs.logits,
                        batch["labels"],
                        loss_mask=batch["loss_mask"],
                        ignore_index=-100,
                    )
                    loss = loss + args.z_loss_weight * z_loss
            (loss / gradient_accumulation_steps).backward()
            accum_loss += float(loss.detach().float().cpu().item())

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            last_grad_norm = float(grad_norm.detach().float().cpu().item())
            if not math.isfinite(last_grad_norm):
                raise RuntimeError(f"non-finite grad_norm={last_grad_norm}")
        muon.step()
        adamw.step()
        if device.type == "mps":
            torch.mps.synchronize()

        mem = mps_memory()
        max_current = max(max_current, int(mem["current_allocated"] or 0))
        max_driver = max(max_driver, int(mem["driver_allocated"] or 0))
        step_time = time.time() - step_start
        if step >= args.warmup_steps:
            measured_times.append(step_time)
            measured_tokens += batch_size * args.block_size * gradient_accumulation_steps
        last_loss = accum_loss / gradient_accumulation_steps

    elapsed = time.time() - start_all
    tok_per_sec = measured_tokens / max(sum(measured_times), 1e-9)
    return {
        "batch_size": batch_size,
        "block_size": args.block_size,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "ok": True,
        "loss": last_loss,
        "grad_norm": last_grad_norm,
        "elapsed_sec": elapsed,
        "measured_step_sec_mean": sum(measured_times) / max(len(measured_times), 1),
        "tok_per_sec": tok_per_sec,
        "memory": {
            "max_current_allocated": max_current,
            "max_driver_allocated": max_driver,
            "recommended_max": mps_memory()["recommended_max"],
        },
        "failure": None,
    }


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    tokenizer = ShoujenTokenizer.load(args.vocab)

    cases = [
        (batch_size, checkpointing, accum)
        for checkpointing in args.gradient_checkpointing
        for accum in args.gradient_accumulation_steps
        for batch_size in args.batch_sizes
    ]
    if args.shuffle_order:
        import random

        random.Random(args.seed).shuffle(cases)

    results: list[dict[str, Any]] = []
    print(
        f"device={device} amp={amp_dtype} block={args.block_size} "
        f"cases={len(cases)} steps={args.steps} warmup={args.warmup_steps}",
        flush=True,
    )
    for batch_size, checkpointing, accum in cases:
        print(
            f"probe batch={batch_size} accum={accum} checkpointing={checkpointing}",
            flush=True,
        )
        try:
            result = run_case(
                args=args,
                tokenizer=tokenizer,
                device=device,
                amp_dtype=amp_dtype,
                batch_size=batch_size,
                gradient_checkpointing=checkpointing,
                gradient_accumulation_steps=accum,
            )
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            clear_device_cache()
            result = {
                "batch_size": batch_size,
                "block_size": args.block_size,
                "gradient_checkpointing": checkpointing,
                "gradient_accumulation_steps": accum,
                "effective_batch_size": batch_size * accum,
                "ok": False,
                "failure": "out of memory",
                "memory": mps_memory(),
            }
        results.append(result)
        if result["ok"]:
            mem = result["memory"]
            print(
                f"  ok tok/s={result['tok_per_sec']:.0f} "
                f"step={result['measured_step_sec_mean']:.2f}s "
                f"driver_gb={mem['max_driver_allocated'] / 1e9:.2f}",
                flush=True,
            )
        else:
            print(f"  failed: {result['failure']}", flush=True)

    successful = [r for r in results if r.get("ok")]
    best = max(successful, key=lambda r: r["tok_per_sec"], default=None)
    summary = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "best_by_tok_per_sec": best,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    if best is not None:
        print(
            "best "
            f"batch={best['batch_size']} accum={best['gradient_accumulation_steps']} "
            f"checkpointing={best['gradient_checkpointing']} tok/s={best['tok_per_sec']:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
