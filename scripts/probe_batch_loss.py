"""Loss-aware batch-size sweep on real packed parquet data.

This is the second-stage probe after `probe_mps_batch.py`: once candidate
micro-batches fit in memory, run short real-data training trials with the same
token budget and compare validation loss, train loss, gradient norms, memory,
and throughput.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_config
from scripts.train_staged_packed import (
    DEFAULT_TRAIN_FILES,
    evaluate,
    lr_multiplier,
    make_loader,
    make_autocast,
    move_batch,
    resolve_data_path,
)
from shoujen.losses import compute_lm_loss, compute_z_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.tokenizer import ShoujenTokenizer
from shoujen.train_utils import autocast_dtype, pick_device, set_optimizer_lr


@dataclass(frozen=True)
class Candidate:
    batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def label(self) -> str:
        gc_suffix = "gc" if self.gradient_checkpointing else "nogc"
        return f"b{self.batch_size}-a{self.gradient_accumulation_steps}-{gc_suffix}"


def parse_candidate(raw: str) -> Candidate:
    parts = [part.strip().lower() for part in raw.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("candidate must be batch:accum:checkpointing")
    batch_size = int(parts[0])
    accum = int(parts[1])
    checkpointing_raw = parts[2]
    if checkpointing_raw in {"1", "true", "yes", "on", "gc"}:
        checkpointing = True
    elif checkpointing_raw in {"0", "false", "no", "off", "nogc"}:
        checkpointing = False
    else:
        raise argparse.ArgumentTypeError(f"invalid checkpointing value: {parts[2]!r}")
    if batch_size <= 0 or accum <= 0:
        raise argparse.ArgumentTypeError("batch and accumulation must be positive")
    return Candidate(batch_size, accum, checkpointing)


def parse_candidates(raw: str) -> list[Candidate]:
    candidates = [parse_candidate(item) for item in raw.split(",") if item.strip()]
    if not candidates:
        raise argparse.ArgumentTypeError("expected at least one candidate")
    return candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", default="data/vocab.json")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed-clean"))
    p.add_argument("--train-files", nargs="+", default=["s-init.parquet"])
    p.add_argument("--val-file", default="s0-val.parquet")
    p.add_argument("--text-column", default="text")
    p.add_argument("--config", help="JSON file overriding default model config")
    p.add_argument("--output", type=Path, default=Path("runs/batch-loss-probe/results.json"))

    p.add_argument(
        "--candidates",
        type=parse_candidates,
        default=parse_candidates("3:1:false,4:1:false,5:1:false,4:2:false,5:1:true,6:1:true"),
        help="Comma list of batch:accum:checkpointing, e.g. 4:1:false,4:2:false",
    )
    p.add_argument("--target-tokens", type=int, default=65536)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shuffle-buffer-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")

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
    p.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--lr-schedule-steps", type=int, default=100000)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--lr-stable-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    return p.parse_args()


def clear_device_cache() -> None:
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def device_memory(device: torch.device) -> dict[str, int | None]:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        props = torch.cuda.get_device_properties(device)
        return {
            "current_allocated": int(torch.cuda.memory_allocated(device)),
            "driver_allocated": int(torch.cuda.memory_reserved(device)),
            "max_current_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_driver_allocated": int(torch.cuda.max_memory_reserved(device)),
            "recommended_max": int(props.total_memory),
        }

    if not hasattr(torch, "mps") or not torch.backends.mps.is_available() or device.type != "mps":
        return {
            "current_allocated": None,
            "driver_allocated": None,
            "max_current_allocated": None,
            "max_driver_allocated": None,
            "recommended_max": None,
        }
    torch.mps.synchronize()
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
        "max_current_allocated": None,
        "max_driver_allocated": None,
        "recommended_max": int(torch.mps.recommended_max_memory()),
    }


def json_safe_args(args: argparse.Namespace) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            safe[key] = str(value)
        elif isinstance(value, list) and value and isinstance(value[0], Candidate):
            safe[key] = [candidate.label for candidate in value]
        else:
            safe[key] = value
    return safe


def train_args_for_candidate(args: argparse.Namespace, candidate: Candidate) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        batch_size=candidate.batch_size,
        max_steps=0,
        max_steps_per_stage=0,
        eval_every=0,
        log_every=0,
        save_every=0,
        init_ckpt=None,
        output=str(args.output.parent),
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode=None,
        log_max_qk_logit=False,
    )
    return argparse.Namespace(**values)


def next_batch(loader_iter: Iterator[dict[str, torch.Tensor]], loader_factory) -> tuple[dict[str, torch.Tensor], Iterator[dict[str, torch.Tensor]]]:
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader_factory())
        return next(loader_iter), loader_iter


def run_candidate(
    *,
    args: argparse.Namespace,
    candidate: Candidate,
    tokenizer: ShoujenTokenizer,
    train_path: Path,
    val_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    clear_device_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)

    train_args = train_args_for_candidate(args, candidate)
    config = build_config(train_args, tokenizer)
    if args.qk_norm:
        config.qk_norm = True
    config.use_cache = False

    model = ShoujenLM(config).to(device).train()
    if candidate.gradient_checkpointing:
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
    base_muon_lrs = [group["lr"] for group in muon.param_groups]
    base_adamw_lrs = [group["lr"] for group in adamw.param_groups]

    initial_val_loss = None
    initial_val_tokens = None
    if args.eval_initial:
        initial_val_loss, initial_val_tokens, _ = evaluate(
            model,
            tokenizer,
            train_args,
            val_path=val_path,
            device=device,
            amp_dtype=amp_dtype,
        )

    def make_train_loader():
        return make_loader(
            train_path,
            tokenizer,
            train_args,
            shuffle=True,
            seed=args.seed,
            device=device,
            drop_last=True,
        )

    loader_iter = iter(make_train_loader())
    train_losses: list[float] = []
    grad_norms: list[float] = []
    active_tokens_seen = 0.0
    input_tokens_seen = 0
    optimizer_steps = 0
    max_current = 0
    max_driver = 0
    start_time = time.time()

    while active_tokens_seen < args.target_tokens:
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        step_active_tokens = 0.0
        step_input_tokens = 0
        step_losses: list[float] = []

        for _ in range(candidate.gradient_accumulation_steps):
            batch_cpu, loader_iter = next_batch(loader_iter, make_train_loader)
            batch = move_batch(
                batch_cpu,
                device,
                vocab_size=tokenizer.vocab_size,
                block_size=args.block_size,
            )
            with make_autocast(device, amp_dtype):
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    sequence_start_mask=batch["sequence_start_mask"],
                    use_cache=False,
                )
                lm_loss, active_tokens = compute_lm_loss(
                    outputs.logits.float(),
                    batch["labels"],
                    loss_mask=batch["loss_mask"],
                    ignore_index=-100,
                )
                total_loss = lm_loss
                if args.z_loss_weight:
                    z_loss = compute_z_loss(
                        outputs.logits,
                        batch["labels"],
                        loss_mask=batch["loss_mask"],
                        ignore_index=-100,
                    )
                    total_loss = total_loss + args.z_loss_weight * z_loss

            loss_value = float(lm_loss.detach().float().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite lm_loss for {candidate.label}")
            (total_loss / candidate.gradient_accumulation_steps).backward()
            token_count = float(active_tokens.detach().float().item())
            step_active_tokens += token_count
            step_input_tokens += int(batch["input_ids"].numel())
            step_losses.append(loss_value)

        if args.grad_clip > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm = float(grad_norm_t.detach().float().item())
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite grad_norm for {candidate.label}")
            grad_norms.append(grad_norm)

        mult = lr_multiplier(train_args, optimizer_steps, max(1, args.lr_schedule_steps))
        set_optimizer_lr(muon, base_muon_lrs, mult)
        set_optimizer_lr(adamw, base_adamw_lrs, mult)
        muon.step()
        adamw.step()

        if device.type == "mps":
            torch.mps.synchronize()
        mem = device_memory(device)
        max_current = max(
            max_current,
            int(mem["current_allocated"] or 0),
            int(mem["max_current_allocated"] or 0),
        )
        max_driver = max(
            max_driver,
            int(mem["driver_allocated"] or 0),
            int(mem["max_driver_allocated"] or 0),
        )

        optimizer_steps += 1
        active_tokens_seen += step_active_tokens
        input_tokens_seen += step_input_tokens
        train_losses.append(sum(step_losses) / len(step_losses))

    elapsed = time.time() - start_time
    final_val_loss, final_val_tokens, final_val_batches = evaluate(
        model,
        tokenizer,
        train_args,
        val_path=val_path,
        device=device,
        amp_dtype=amp_dtype,
    )
    if device.type == "mps":
        torch.mps.synchronize()
    final_mem = device_memory(device)
    max_current = max(max_current, int(final_mem["max_current_allocated"] or 0))
    max_driver = max(max_driver, int(final_mem["max_driver_allocated"] or 0))

    train_loss_mean = sum(train_losses) / max(1, len(train_losses))
    train_loss_final = train_losses[-1] if train_losses else None
    grad_norm_mean = sum(grad_norms) / max(1, len(grad_norms))
    return {
        "candidate": candidate.label,
        "batch_size": candidate.batch_size,
        "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
        "effective_batch_size": candidate.effective_batch_size,
        "gradient_checkpointing": candidate.gradient_checkpointing,
        "ok": True,
        "optimizer_steps": optimizer_steps,
        "active_tokens_seen": active_tokens_seen,
        "input_tokens_seen": input_tokens_seen,
        "elapsed_sec": elapsed,
        "tok_per_sec": input_tokens_seen / max(elapsed, 1e-9),
        "initial_val_loss": initial_val_loss,
        "initial_val_tokens": initial_val_tokens,
        "final_val_loss": final_val_loss,
        "final_val_tokens": final_val_tokens,
        "final_val_batches": final_val_batches,
        "val_loss_delta": None if initial_val_loss is None else final_val_loss - initial_val_loss,
        "train_loss_mean": train_loss_mean,
        "train_loss_final": train_loss_final,
        "grad_norm_mean": grad_norm_mean,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "memory": {
            "max_current_allocated": max_current,
            "max_driver_allocated": max_driver,
            "recommended_max": final_mem["recommended_max"],
        },
        "failure": None,
    }


def main() -> None:
    args = parse_args()
    if args.target_tokens <= 0:
        raise SystemExit("--target-tokens must be positive")
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    tokenizer = ShoujenTokenizer.load(args.vocab)
    train_paths = [resolve_data_path(args.data_dir, p) for p in args.train_files]
    val_path = resolve_data_path(args.data_dir, args.val_file)
    missing = [p for p in [*train_paths, val_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing parquet file(s): " + ", ".join(str(p) for p in missing))

    print(
        f"device={device} amp={amp_dtype} block={args.block_size} "
        f"target_tokens={args.target_tokens} candidates={len(args.candidates)}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for candidate in args.candidates:
        print(f"probe {candidate.label}", flush=True)
        try:
            result = run_candidate(
                args=args,
                candidate=candidate,
                tokenizer=tokenizer,
                train_path=train_paths[0],
                val_path=val_path,
                device=device,
                amp_dtype=amp_dtype,
            )
        except RuntimeError as exc:
            clear_device_cache()
            mem = device_memory(device)
            result = {
                "candidate": candidate.label,
                "batch_size": candidate.batch_size,
                "gradient_accumulation_steps": candidate.gradient_accumulation_steps,
                "effective_batch_size": candidate.effective_batch_size,
                "gradient_checkpointing": candidate.gradient_checkpointing,
                "ok": False,
                "failure": str(exc),
                "memory": mem,
            }
        results.append(result)
        if result["ok"]:
            print(
                f"  ok val={result['final_val_loss']:.4f} "
                f"delta={result['val_loss_delta']:.4f} "
                f"train={result['train_loss_final']:.4f} "
                f"tok/s={result['tok_per_sec']:.0f} "
                f"driver_gb={result['memory']['max_driver_allocated'] / 1e9:.2f}",
                flush=True,
            )
        else:
            print(f"  failed: {result['failure']}", flush=True)

    successful = [r for r in results if r.get("ok")]
    best_loss = min(successful, key=lambda r: r["final_val_loss"], default=None)
    best_score = min(
        successful,
        key=lambda r: r["final_val_loss"] - 0.01 * math.log(max(r["tok_per_sec"], 1.0)),
        default=None,
    )
    summary = {
        "args": json_safe_args(args),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "best_by_val_loss": best_loss,
        "best_by_loss_speed_score": best_score,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    if best_loss is not None:
        print(f"best_val_loss {best_loss['candidate']} val={best_loss['final_val_loss']:.4f}", flush=True)
    if best_score is not None:
        print(f"best_score {best_score['candidate']} val={best_score['final_val_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
