"""Boundary-safe staged pretraining over cleaned parquet shards.

Default stage order:
    s-init.parquet -> s0-train.parquet -> s1.parquet -> s2.parquet

Validation always uses:
    s0-val.parquet

Example:
    uv run python scripts/train_staged_packed.py \
        --output runs/staged-packed \
        --batch-size 4 --block-size 2048 \
        --eval-every 500 --wandb
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_config
from shoujen.data import (
    PackedParquetPretrainDataset,
    PrepackedParquetPretrainDataset,
    is_prepacked_parquet,
    packed_collate,
)
from shoujen.losses import compute_lm_loss, compute_z_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer
from shoujen.train_utils import (
    autocast_dtype,
    load_checkpoint,
    move_tensor_to_device,
    pick_device,
    save_checkpoint,
    set_optimizer_lr,
    set_optimizer_momentum,
    warmup_cosine_lr,
    warmup_momentum,
    warmup_stable_decay_lr,
)

DEFAULT_TRAIN_FILES = [
    "s-init.parquet",
    "s0-train.parquet",
    "s1.parquet",
    "s2.parquet",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="Hugging Face tokenizer id/URL or saved tokenizer directory",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/processed-clean"))
    p.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES)
    p.add_argument("--val-file", default="s0-val.parquet")
    p.add_argument("--text-column", default="text")
    p.add_argument(
        "--data-format",
        choices=["auto", "text", "packed"],
        default="auto",
        help="Input parquet format. auto detects offline packed-tokenized shards.",
    )
    p.add_argument("--output", default="runs/staged-packed-pretrain")
    p.add_argument("--init-ckpt", help="Resume / initialize from a checkpoint")
    p.add_argument("--config", help="JSON file overriding default model config")

    p.add_argument("--batch-size", type=int, default=4, help="Packed sequences per micro-batch")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trade extra compute for lower activation memory.",
    )
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=0, help="Global cap; 0 means no cap")
    p.add_argument("--max-steps-per-stage", type=int, default=0, help="Per-stage cap; 0 means full shard")
    p.add_argument("--lr-schedule-steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--muon-lr", type=float, default=3e-4)
    p.add_argument("--muon-momentum", type=float, default=0.95, help="Target Muon momentum after warmup")
    p.add_argument(
        "--muon-momentum-start",
        type=float,
        default=0.85,
        help="Initial Muon momentum at step 0 (linearly ramped to --muon-momentum over --muon-momentum-warmup steps)",
    )
    p.add_argument(
        "--muon-momentum-warmup",
        type=int,
        default=None,
        help="Steps to ramp Muon momentum from start to target (defaults to --warmup if unset; pass 0 to disable)",
    )
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--muon-adaptive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--muon-adaptive-beta2", type=float, default=0.95)
    p.add_argument("--muon-adaptive-eps", type=float, default=1e-8)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--adamw-beta1", type=float, default=0.8)
    p.add_argument("--adamw-beta2", type=float, default=0.95)
    p.add_argument("--muon-wd", type=float, default=0.0)
    p.add_argument("--adamw-wd", type=float, default=0.1)
    p.add_argument("--adamw-independent-wd", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adamw-embed-wd", type=float, default=0.0)
    p.add_argument("--adamw-foreach", action="store_true")
    p.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="cosine")
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--lr-stable-steps", type=int, default=0)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument(
        "--attention-window",
        type=int,
        default=256,
        help="Sliding-window size for attention layers (W tokens). Pass 0 or a "
        "negative value to disable; RWKV layers are unaffected.",
    )
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--log-max-qk-logit", action="store_true")

    p.add_argument("--shuffle-buffer-size", type=int, default=10000)
    p.add_argument(
        "--packed-shuffle-buffer-size",
        type=int,
        default=1024,
        help="Sample shuffle buffer for offline packed-tokenized shards.",
    )
    p.add_argument(
        "--packed-read-batch-size",
        type=int,
        default=1024,
        help="Parquet read batch size for offline packed-tokenized shards.",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="shoujen-llm")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default=None)
    return p.parse_args()


def resolve_data_path(data_dir: Path, name: str | Path) -> Path:
    path = Path(name)
    return path if path.is_absolute() else data_dir / path


def resolve_data_format(path: Path, args: argparse.Namespace) -> str:
    data_format = getattr(args, "data_format", "auto")
    if data_format == "auto":
        return "packed" if is_prepacked_parquet(path) else "text"
    if data_format == "packed" and not is_prepacked_parquet(path):
        raise ValueError(f"--data-format packed was set, but {path} is not a packed-tokenized parquet")
    return data_format


def make_loader(
    path: Path,
    tokenizer: ShoujenTokenizer,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed: int,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    data_format = resolve_data_format(path, args)
    if data_format == "packed":
        dataset = PrepackedParquetPretrainDataset(
            path,
            block_size=args.block_size,
            shuffle=shuffle,
            shuffle_buffer_size=getattr(args, "packed_shuffle_buffer_size", 1024),
            seed=seed,
            read_batch_size=getattr(args, "packed_read_batch_size", 1024),
        )
    else:
        dataset = PackedParquetPretrainDataset(
            path,
            tokenizer,
            block_size=args.block_size,
            text_column=args.text_column,
            shuffle=shuffle,
            shuffle_buffer_size=args.shuffle_buffer_size,
            seed=seed,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=packed_collate,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
    )


def _valid_moved_batch(
    batch: dict[str, torch.Tensor],
    *,
    vocab_size: int | None,
    block_size: int | None,
) -> bool:
    if vocab_size is None:
        return True
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    position_ids = batch["position_ids"]

    valid_inputs = ((input_ids >= 0) & (input_ids < vocab_size)).all()
    valid_labels = ((labels == -100) | ((labels >= 0) & (labels < vocab_size))).all()
    valid_loss_mask = torch.isfinite(loss_mask).all() & (loss_mask >= 0).all() & (loss_mask <= 1).all()
    del block_size
    valid_positions = (position_ids >= 0).all()
    return bool((valid_inputs & valid_labels & valid_loss_mask & valid_positions).detach().cpu().item())


def _batch_debug_stats(batch: dict[str, torch.Tensor]) -> str:
    return (
        f"input_ids=[{batch['input_ids'].min().item()}, {batch['input_ids'].max().item()}] "
        f"labels=[{batch['labels'].min().item()}, {batch['labels'].max().item()}] "
        f"loss_mask_sum={float(batch['loss_mask'].sum().item()):.6g} "
        f"position_ids=[{batch['position_ids'].min().item()}, {batch['position_ids'].max().item()}]"
    )


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    vocab_size: int | None = None,
    block_size: int | None = None,
    retries: int = 3,
) -> dict[str, torch.Tensor]:
    keys = (
        "input_ids",
        "labels",
        "loss_mask",
        "attention_mask",
        "position_ids",
        "sequence_start_mask",
    )
    for attempt in range(max(1, retries)):
        moved = {k: move_tensor_to_device(batch[k], device) for k in keys}
        if _valid_moved_batch(moved, vocab_size=vocab_size, block_size=block_size):
            return moved
        if device.type == "mps":
            torch.mps.empty_cache()
    raise RuntimeError(
        "Invalid batch after device transfer; refusing to train on corrupted tensors. "
        + _batch_debug_stats(moved)
    )


def maybe_init_wandb(
    args: argparse.Namespace,
    config: Any,
    train_paths: list[Path],
    val_path: Path,
):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install it or run without --wandb.") from exc

    args_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "entity": args.wandb_entity,
        "config": {
            **args_config,
            "train_paths": [str(p) for p in train_paths],
            "val_path": str(val_path),
            "model_config": config.to_dict(),
        },
    }
    if args.wandb_mode:
        kwargs["mode"] = args.wandb_mode
    return wandb.init(**{k: v for k, v in kwargs.items() if v is not None})


def log_wandb(run, payload: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(payload, step=step)


def make_autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def lr_multiplier(args: argparse.Namespace, step: int, max_steps: int) -> float:
    if args.lr_schedule == "wsd":
        return warmup_stable_decay_lr(
            step,
            warmup=args.warmup,
            max_steps=max_steps,
            stable_steps=args.lr_stable_steps,
        )
    return warmup_cosine_lr(
        step,
        warmup=args.warmup,
        max_steps=max_steps,
        min_ratio=args.lr_min_ratio,
    )


@torch.no_grad()
def evaluate(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    args: argparse.Namespace,
    *,
    val_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float, int]:
    was_training = model.training
    model.eval()
    loader = make_loader(
        val_path,
        tokenizer,
        args,
        shuffle=False,
        seed=args.seed,
        device=device,
        drop_last=False,
    )

    total_loss = 0.0
    total_tokens = 0.0
    batches = 0
    for batch in loader:
        if args.eval_batches and batches >= args.eval_batches:
            break
        batch = move_batch(
            batch,
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
            loss, active_tokens = compute_lm_loss(
                outputs.logits.float(),
                batch["labels"],
                loss_mask=batch["loss_mask"],
                ignore_index=-100,
            )
        token_count = float(active_tokens.item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        batches += 1

    if was_training:
        model.train()
    if total_tokens == 0:
        return math.nan, 0.0, batches
    return total_loss / total_tokens, total_tokens, batches


def save_training_checkpoint(
    path: Path,
    *,
    model: ShoujenLM,
    config: Any,
    step: int,
    muon: torch.optim.Optimizer,
    adamw: torch.optim.Optimizer,
    stage_idx: int,
    stage_name: str,
) -> None:
    save_checkpoint(
        path,
        model=model,
        config=config,
        step=step,
        muon=muon,
        adamw=adamw,
        extra={"stage_idx": stage_idx, "stage_name": stage_name},
    )


def main():
    args = parse_args()
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [resolve_data_path(args.data_dir, p) for p in args.train_files]
    val_path = resolve_data_path(args.data_dir, args.val_file)
    missing = [p for p in [*train_paths, val_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing parquet file(s): " + ", ".join(str(p) for p in missing))

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    print(f"device={device} amp={amp_dtype}", flush=True)
    print("train stages: " + " -> ".join(p.name for p in train_paths), flush=True)
    print(f"validation: {val_path.name}", flush=True)
    print(
        "data formats: "
        + ", ".join(f"{p.name}={resolve_data_format(p, args)}" for p in [*train_paths, val_path]),
        flush=True,
    )

    tokenizer = ShoujenTokenizer.load(args.tokenizer)
    config = build_config(args, tokenizer)
    if args.qk_norm:
        config.qk_norm = True
    config.to_json(out_dir / "config.json")

    model = ShoujenLM(config).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("gradient_checkpointing=enabled", flush=True)
    print(f"model: {model.num_parameters() / 1e6:.2f}M params", flush=True)

    muon, adamw = build_optimizers(
        model,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_wd=args.muon_wd,
        muon_adaptive=args.muon_adaptive,
        muon_adaptive_beta2=args.muon_adaptive_beta2,
        muon_adaptive_eps=args.muon_adaptive_eps,
        adamw_lr=args.adamw_lr,
        adamw_betas=(args.adamw_beta1, args.adamw_beta2),
        adamw_wd=args.adamw_wd,
        adamw_embed_wd=args.adamw_embed_wd if args.adamw_independent_wd else None,
        adamw_foreach=True if args.adamw_foreach else None,
    )
    base_muon_lrs = [g["lr"] for g in muon.param_groups]
    base_adamw_lrs = [g["lr"] for g in adamw.param_groups]
    muon_mom_warmup = args.warmup if args.muon_momentum_warmup is None else args.muon_momentum_warmup

    global_step = 0
    if args.init_ckpt:
        state = load_checkpoint(args.init_ckpt, model, map_location=device)
        if "muon" in state:
            muon.load_state_dict(state["muon"])
        if "adamw" in state:
            try:
                adamw.load_state_dict(state["adamw"])
            except ValueError:
                print("WARN: AdamW state shape/group mismatch - skipping.", flush=True)
        global_step = int(state.get("step", 0))
        print(f"loaded ckpt {args.init_ckpt} step={global_step}", flush=True)

    wandb_run = maybe_init_wandb(args, config, train_paths, val_path)
    model.train()
    model.set_track_max_qk_logit(args.log_max_qk_logit)
    last_log_t = time.time()
    last_log_tokens = 0
    schedule_steps = max(1, args.lr_schedule_steps)

    stop_training = False
    for stage_idx, stage_path in enumerate(train_paths):
        if stop_training:
            break
        stage_name = stage_path.stem
        print(f"stage {stage_idx + 1}/{len(train_paths)}: {stage_path}", flush=True)
        loader = make_loader(
            stage_path,
            tokenizer,
            args,
            shuffle=True,
            seed=args.seed + stage_idx,
            device=device,
            drop_last=True,
        )

        stage_step = 0
        accum_count = 0
        last_lm_loss = None
        last_total_loss = None
        last_z_loss = None
        last_active_tokens = None
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        for batch in loader:
            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break
            if args.max_steps_per_stage and stage_step >= args.max_steps_per_stage:
                break

            batch = move_batch(
                batch,
                device,
                vocab_size=tokenizer.vocab_size,
                block_size=args.block_size,
            )
            if accum_count == 0:
                mult = lr_multiplier(args, global_step, schedule_steps)
                set_optimizer_lr(muon, base_muon_lrs, mult)
                set_optimizer_lr(adamw, base_adamw_lrs, mult)
                mom_now = warmup_momentum(
                    global_step,
                    warmup=muon_mom_warmup,
                    start=args.muon_momentum_start,
                    end=args.muon_momentum,
                )
                set_optimizer_momentum(muon, mom_now)

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
                z_loss = None
                if args.z_loss_weight:
                    z_loss = compute_z_loss(
                        outputs.logits,
                        batch["labels"],
                        loss_mask=batch["loss_mask"],
                        ignore_index=-100,
                    )
                    total_loss = total_loss + args.z_loss_weight * z_loss

            (total_loss / args.gradient_accumulation_steps).backward()
            accum_count += 1
            last_lm_loss = lm_loss
            last_total_loss = total_loss
            last_z_loss = z_loss
            last_active_tokens = active_tokens
            last_log_tokens += int(batch["input_ids"].numel())

            if accum_count < args.gradient_accumulation_steps:
                continue

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm at step {global_step + 1}: {grad_norm.item()}")
            muon.step()
            adamw.step()
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            accum_count = 0

            global_step += 1
            stage_step += 1

            if args.log_every and global_step % args.log_every == 0:
                dt = time.time() - last_log_t
                tok_per_sec = last_log_tokens / max(dt, 1e-6)
                extras = ""
                if args.z_loss_weight and last_z_loss is not None and last_total_loss is not None:
                    extras += f" z={last_z_loss.item():.4f} total={last_total_loss.item():.4f}"
                max_qk = None
                if args.log_max_qk_logit:
                    max_qk = model.max_qk_logit()
                    if max_qk is not None:
                        extras += f" max_qk={max_qk:.2f}"
                print(
                    f"stage={stage_name} step={global_step} stage_step={stage_step} "
                    f"lm={last_lm_loss.item():.4f}{extras} lr_mult={mult:.3f} tok/s={tok_per_sec:.0f}",
                    flush=True,
                )
                wandb_payload = {
                    "train/lm_loss": float(last_lm_loss.item()),
                    "train/active_tokens": float(last_active_tokens.item()) if last_active_tokens is not None else 0.0,
                    "train/lr_multiplier": mult,
                    "train/tok_per_sec": tok_per_sec,
                    "stage/index": stage_idx,
                    "stage/step": stage_step,
                }
                if last_z_loss is not None and last_total_loss is not None:
                    wandb_payload["train/z_loss"] = float(last_z_loss.item())
                    wandb_payload["train/total_loss"] = float(last_total_loss.item())
                if max_qk is not None:
                    wandb_payload["train/max_qk_logit"] = max_qk
                log_wandb(
                    wandb_run,
                    wandb_payload,
                    global_step,
                )
                last_log_t = time.time()
                last_log_tokens = 0

            if args.eval_every and global_step % args.eval_every == 0:
                val_loss, val_tokens, val_batches = evaluate(
                    model,
                    tokenizer,
                    args,
                    val_path=val_path,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                print(
                    f"eval step={global_step} val_lm={val_loss:.4f} "
                    f"tokens={val_tokens:.0f} batches={val_batches}",
                    flush=True,
                )
                log_wandb(
                    wandb_run,
                    {
                        "val/lm_loss": val_loss,
                        "val/active_tokens": val_tokens,
                        "val/batches": val_batches,
                    },
                    global_step,
                )

            if args.save_every and global_step % args.save_every == 0:
                save_training_checkpoint(
                    out_dir / f"step{global_step}.pt",
                    model=model,
                    config=config,
                    step=global_step,
                    muon=muon,
                    adamw=adamw,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                )
                save_training_checkpoint(
                    out_dir / "last.pt",
                    model=model,
                    config=config,
                    step=global_step,
                    muon=muon,
                    adamw=adamw,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                )
                print(f"saved {out_dir / f'step{global_step}.pt'}", flush=True)

        val_loss, val_tokens, val_batches = evaluate(
            model,
            tokenizer,
            args,
            val_path=val_path,
            device=device,
            amp_dtype=amp_dtype,
        )
        print(
            f"stage_done={stage_name} step={global_step} val_lm={val_loss:.4f} "
            f"tokens={val_tokens:.0f} batches={val_batches}",
            flush=True,
        )
        log_wandb(
            wandb_run,
            {
                "val/lm_loss": val_loss,
                "val/active_tokens": val_tokens,
                "val/batches": val_batches,
                "stage/completed_index": stage_idx,
            },
            global_step,
        )
        save_training_checkpoint(
            out_dir / f"{stage_idx:02d}-{stage_name}.pt",
            model=model,
            config=config,
            step=global_step,
            muon=muon,
            adamw=adamw,
            stage_idx=stage_idx,
            stage_name=stage_name,
        )
        save_training_checkpoint(
            out_dir / "last.pt",
            model=model,
            config=config,
            step=global_step,
            muon=muon,
            adamw=adamw,
            stage_idx=stage_idx,
            stage_name=stage_name,
        )

    if wandb_run is not None:
        wandb_run.finish()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
