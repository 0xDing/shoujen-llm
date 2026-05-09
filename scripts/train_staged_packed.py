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
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
from shoujen.evaluation import (
    CoreMetrics,
    EvalMetrics,
    bpb_from_sums,
    compute_lm_eval_metrics,
    evaluate_core,
    token_byte_lengths,
)
from shoujen.losses import compute_lm_loss, compute_z_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer
from shoujen.train_utils import (
    autocast_dtype,
    filter_optimizer_state_for_param_shapes,
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


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw != "" else default


def init_distributed(args: argparse.Namespace) -> tuple[DistributedContext, torch.device]:
    env_world_size = _env_int("WORLD_SIZE", 1)
    enabled = env_world_size > 1 if args.distributed is None else bool(args.distributed)
    if not enabled:
        return DistributedContext(), torch.device(args.device) if args.device else pick_device()

    if env_world_size <= 1:
        raise SystemExit("--distributed requires launching with torchrun or another env:// launcher")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA DDP was requested, but torch.cuda.is_available() is false")

    local_rank = args.local_rank
    if local_rank is None:
        local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    cuda_device_count = torch.cuda.device_count()

    requested_device = torch.device(args.device) if args.device else None
    if requested_device is not None and requested_device.type != "cuda":
        raise SystemExit("Distributed training only supports CUDA devices")
    if requested_device is not None and requested_device.index is not None:
        if requested_device.index != local_rank:
            raise SystemExit(
                "In distributed mode, omit --device or pass --device cuda; "
                "each torchrun process is bound to cuda:${LOCAL_RANK}."
            )
        device = requested_device
    else:
        device = torch.device("cuda", local_rank)
    if device.index is None:
        raise SystemExit("Distributed CUDA device must have an explicit index")
    if device.index >= cuda_device_count:
        raise SystemExit(
            f"LOCAL_RANK={local_rank} maps to {device}, but only "
            f"{cuda_device_count} CUDA device(s) are visible"
        )

    torch.cuda.set_device(device)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    timeout = timedelta(seconds=args.distributed_timeout_seconds)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
            timeout=timeout,
        )
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
    return DistributedContext(
        enabled=True,
        rank=dist.get_rank() if dist.is_initialized() else rank,
        local_rank=local_rank,
        world_size=dist.get_world_size() if dist.is_initialized() else env_world_size,
    ), device


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        try:
            dist.barrier(device_ids=[ctx.local_rank])
        except TypeError:
            dist.barrier()


def print_main(ctx: DistributedContext, *args, **kwargs) -> None:
    if ctx.is_main:
        print(*args, **kwargs)


def print_rank(ctx: DistributedContext, message: str) -> None:
    print(
        f"[rank{ctx.rank} local_rank={ctx.local_rank} pid={os.getpid()}] {message}",
        file=sys.stderr,
        flush=True,
    )


def distributed_startup_check(ctx: DistributedContext, device: torch.device) -> None:
    if not ctx.enabled:
        return
    print_rank(ctx, f"startup collective begin device={device}")
    tensor = torch.tensor(ctx.rank + 1, dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    expected = ctx.world_size * (ctx.world_size + 1) // 2
    actual = int(tensor.item())
    if actual != expected:
        raise RuntimeError(f"DDP startup collective returned {actual}, expected {expected}")
    print_rank(ctx, "startup collective ok")


def distributed_sum_float(value: float, *, device: torch.device, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def distributed_max_float(value: float, *, device: torch.device, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def next_distributed_batch(
    iterator: Any,
    *,
    device: torch.device,
    ctx: DistributedContext,
) -> dict[str, torch.Tensor] | None:
    try:
        batch = next(iterator)
        has_batch = 1
    except StopIteration:
        batch = None
        has_batch = 0

    if ctx.enabled:
        flag = torch.tensor(has_batch, dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if int(flag.item()) == 0:
            return None
    return batch


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
    p.add_argument(
        "--core-eval-dir",
        type=Path,
        default=None,
        help="Optional CORE eval bundle directory containing core.yaml and eval_data/.",
    )
    p.add_argument(
        "--core-metric-every",
        type=int,
        default=0,
        help="Evaluate CORE every N optimizer steps when --core-eval-dir is set. 0 disables CORE.",
    )
    p.add_argument(
        "--core-metric-max-per-task",
        type=int,
        default=100,
        help="Max CORE examples per task; pass -1 for all examples.",
    )
    p.add_argument(
        "--core-tasks",
        default="",
        help="Comma-separated CORE task labels to run; empty runs all tasks from core.yaml.",
    )
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
    p.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable single-node CUDA DDP. Defaults to on when WORLD_SIZE > 1.",
    )
    p.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for NCCL distributed collectives.",
    )
    p.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

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
    rank: int = 0,
    world_size: int = 1,
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
            rank=rank,
            world_size=world_size,
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
            rank=rank,
            world_size=world_size,
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


def parse_core_task_labels(raw: str) -> list[str] | None:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    return labels or None


def maybe_evaluate_core(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    args: argparse.Namespace,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    step: int,
    force: bool = False,
) -> CoreMetrics | None:
    if args.core_eval_dir is None or args.core_metric_every <= 0:
        return None
    if not force and (step <= 0 or step % args.core_metric_every != 0):
        return None
    core = evaluate_core(
        model,
        tokenizer,
        eval_dir=args.core_eval_dir,
        device=device,
        amp_dtype=amp_dtype,
        max_per_task=args.core_metric_max_per_task,
        task_labels=parse_core_task_labels(args.core_tasks),
        seed=args.seed,
    )
    print(f"core step={step} score={core.score:.4f}", flush=True)
    return core


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
            min_ratio=args.lr_min_ratio,
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
    token_bytes: torch.Tensor | None = None,
) -> EvalMetrics:
    was_training = model.training
    model.eval()
    if token_bytes is None:
        token_bytes = token_byte_lengths(tokenizer, device=device)
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
    total_bpb_nats = 0.0
    total_bytes = 0.0
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
            loss, active_tokens, bpb_nats, byte_count = compute_lm_eval_metrics(
                outputs.logits.float(),
                batch["labels"],
                loss_mask=batch["loss_mask"],
                token_bytes=token_bytes,
                ignore_index=-100,
            )
        token_count = float(active_tokens.item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
        total_bpb_nats += float(bpb_nats.item())
        total_bytes += float(byte_count.item())
        batches += 1

    if was_training:
        model.train()
    if total_tokens == 0:
        return EvalMetrics(lm_loss=math.nan, active_tokens=0.0, batches=batches)
    return EvalMetrics(
        lm_loss=total_loss / total_tokens,
        active_tokens=total_tokens,
        batches=batches,
        bpb=bpb_from_sums(total_bpb_nats, total_bytes),
        bytes=total_bytes,
    )


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
    dist_ctx, device = init_distributed(args)
    if dist_ctx.enabled:
        print_rank(
            dist_ctx,
            f"distributed initialized backend=nccl world_size={dist_ctx.world_size} device={device}",
        )
        distributed_startup_check(dist_ctx, device)
    else:
        print_main(dist_ctx, "distributed disabled", flush=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_main(dist_ctx, f"output directory ready: {out_dir}", flush=True)

    train_paths = [resolve_data_path(args.data_dir, p) for p in args.train_files]
    val_path = resolve_data_path(args.data_dir, args.val_file)
    missing = [p for p in [*train_paths, val_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing parquet file(s): " + ", ".join(str(p) for p in missing))

    amp_dtype = None if args.no_amp else autocast_dtype(device)
    distributed_summary = (
        f" distributed=ddp rank={dist_ctx.rank}/{dist_ctx.world_size} local_rank={dist_ctx.local_rank}"
        if dist_ctx.enabled
        else ""
    )
    print_main(dist_ctx, f"device={device} amp={amp_dtype}{distributed_summary}", flush=True)
    print_main(dist_ctx, "train stages: " + " -> ".join(p.name for p in train_paths), flush=True)
    print_main(dist_ctx, f"validation: {val_path.name}", flush=True)
    print_main(
        dist_ctx,
        "data formats: "
        + ", ".join(f"{p.name}={resolve_data_format(p, args)}" for p in [*train_paths, val_path]),
        flush=True,
    )

    tokenizer = ShoujenTokenizer.load(args.tokenizer)
    eval_token_bytes = token_byte_lengths(tokenizer, device=device)
    config = build_config(args, tokenizer)
    if args.qk_norm:
        config.qk_norm = True
    if dist_ctx.is_main:
        config.to_json(out_dir / "config.json")

    model = ShoujenLM(config).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_main(dist_ctx, "gradient_checkpointing=enabled", flush=True)
    print_main(dist_ctx, f"model: {model.num_parameters() / 1e6:.2f}M params", flush=True)

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
        state = load_checkpoint(args.init_ckpt, model, map_location=device, allow_appended_vocab=True)
        appended_vocab = state.get("_shoujen_appended_vocab_keys") or []
        if appended_vocab:
            details = ", ".join(
                f"{item['name']} {item['checkpoint_shape']} -> {item['model_shape']}"
                for item in appended_vocab
            )
            print_main(
                dist_ctx,
                f"loaded ckpt with appended tokenizer rows; initialized new rows for {details}",
                flush=True,
            )
        if "muon" in state:
            muon.load_state_dict(state["muon"])
        if "adamw" in state:
            try:
                adamw_state = state["adamw"]
                if appended_vocab:
                    adamw_state, skipped = filter_optimizer_state_for_param_shapes(adamw, adamw_state)
                    if skipped:
                        print_main(
                            dist_ctx,
                            f"WARN: skipped AdamW state for {len(skipped)} resized parameter(s).",
                            flush=True,
                        )
                adamw.load_state_dict(adamw_state)
            except ValueError:
                print_main(dist_ctx, "WARN: AdamW state shape/group mismatch - skipping.", flush=True)
        global_step = int(state.get("step", 0))
        print_main(dist_ctx, f"loaded ckpt {args.init_ckpt} step={global_step}", flush=True)

    model.train()
    model.set_track_max_qk_logit(args.log_max_qk_logit)
    train_model: torch.nn.Module = model
    if dist_ctx.enabled:
        train_model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    wandb_run = maybe_init_wandb(args, config, train_paths, val_path) if dist_ctx.is_main else None
    last_log_t = time.time()
    last_log_tokens = 0
    schedule_steps = max(1, args.lr_schedule_steps)

    stop_training = False
    for stage_idx, stage_path in enumerate(train_paths):
        if stop_training:
            break
        stage_name = stage_path.stem
        print_main(dist_ctx, f"stage {stage_idx + 1}/{len(train_paths)}: {stage_path}", flush=True)
        loader = make_loader(
            stage_path,
            tokenizer,
            args,
            shuffle=True,
            seed=args.seed + stage_idx,
            device=device,
            drop_last=True,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )

        stage_step = 0
        accum_count = 0
        last_lm_loss = None
        last_total_loss = None
        last_z_loss = None
        last_active_tokens = None
        last_global_active_tokens = 0.0
        last_lm_loss_sum = 0.0
        last_total_loss_sum = 0.0
        last_z_loss_sum = None
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loader_iter = iter(loader)
        while True:
            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                break
            if args.max_steps_per_stage and stage_step >= args.max_steps_per_stage:
                break
            batch = next_distributed_batch(loader_iter, device=device, ctx=dist_ctx)
            if batch is None:
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

            sync_grad = accum_count + 1 >= args.gradient_accumulation_steps
            sync_context = (
                train_model.no_sync()
                if dist_ctx.enabled and isinstance(train_model, DDP) and not sync_grad
                else nullcontext()
            )
            with sync_context:
                with make_autocast(device, amp_dtype):
                    outputs = train_model(
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

                local_active_tokens = float(active_tokens.detach().item())
                global_active_tokens = distributed_sum_float(
                    local_active_tokens,
                    device=device,
                    ctx=dist_ctx,
                )
                if dist_ctx.enabled:
                    loss_scale = (
                        dist_ctx.world_size
                        * local_active_tokens
                        / max(global_active_tokens, 1.0)
                    )
                else:
                    loss_scale = 1.0
                backward_loss = total_loss * loss_scale
                (backward_loss / args.gradient_accumulation_steps).backward()
            accum_count += 1
            last_lm_loss = lm_loss
            last_total_loss = total_loss
            last_z_loss = z_loss
            last_active_tokens = active_tokens
            last_global_active_tokens = global_active_tokens
            last_lm_loss_sum = float(lm_loss.detach().item()) * local_active_tokens
            last_total_loss_sum = float(total_loss.detach().item()) * local_active_tokens
            last_z_loss_sum = (
                float(z_loss.detach().item()) * local_active_tokens if z_loss is not None else None
            )
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
                total_log_tokens = distributed_sum_float(
                    float(last_log_tokens),
                    device=device,
                    ctx=dist_ctx,
                )
                tok_per_sec = total_log_tokens / max(dt, 1e-6)
                lm_log = distributed_sum_float(last_lm_loss_sum, device=device, ctx=dist_ctx) / max(
                    last_global_active_tokens,
                    1.0,
                )
                active_tokens_log = last_global_active_tokens if last_active_tokens is not None else 0.0
                z_log = None
                total_loss_log = None
                if args.z_loss_weight and last_z_loss is not None and last_total_loss is not None:
                    if last_z_loss_sum is not None:
                        z_log = distributed_sum_float(
                            last_z_loss_sum,
                            device=device,
                            ctx=dist_ctx,
                        ) / max(last_global_active_tokens, 1.0)
                    total_loss_log = distributed_sum_float(
                        last_total_loss_sum,
                        device=device,
                        ctx=dist_ctx,
                    ) / max(last_global_active_tokens, 1.0)
                max_qk = None
                if args.log_max_qk_logit:
                    local_max_qk = model.max_qk_logit()
                    max_qk_value = distributed_max_float(
                        float(local_max_qk) if local_max_qk is not None else float("-inf"),
                        device=device,
                        ctx=dist_ctx,
                    )
                    if math.isfinite(max_qk_value):
                        max_qk = max_qk_value
                if dist_ctx.is_main:
                    extras = ""
                    if z_log is not None and total_loss_log is not None:
                        extras += f" z={z_log:.4f} total={total_loss_log:.4f}"
                    if max_qk is not None:
                        extras += f" max_qk={max_qk:.2f}"
                    print(
                        f"stage={stage_name} step={global_step} stage_step={stage_step} "
                        f"lm={lm_log:.4f}{extras} lr_mult={mult:.3f} tok/s={tok_per_sec:.0f}",
                        flush=True,
                    )
                    wandb_payload = {
                        "train/lm_loss": lm_log,
                        "train/active_tokens": active_tokens_log,
                        "train/lr_multiplier": mult,
                        "train/tok_per_sec": tok_per_sec,
                        "stage/index": stage_idx,
                        "stage/step": stage_step,
                    }
                    if z_log is not None and total_loss_log is not None:
                        wandb_payload["train/z_loss"] = z_log
                        wandb_payload["train/total_loss"] = total_loss_log
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
                if dist_ctx.is_main:
                    val_metrics = evaluate(
                        model,
                        tokenizer,
                        args,
                        val_path=val_path,
                        device=device,
                        amp_dtype=amp_dtype,
                        token_bytes=eval_token_bytes,
                    )
                    print(
                        f"eval step={global_step} val_lm={val_metrics.lm_loss:.4f} "
                        f"val_bpb={val_metrics.bpb:.6f} tokens={val_metrics.active_tokens:.0f} "
                        f"bytes={val_metrics.bytes:.0f} batches={val_metrics.batches}",
                        flush=True,
                    )
                    payload = val_metrics.as_log_dict("val")
                    core_metrics = maybe_evaluate_core(
                        model,
                        tokenizer,
                        args,
                        device=device,
                        amp_dtype=amp_dtype,
                        step=global_step,
                    )
                    if core_metrics is not None:
                        payload.update(core_metrics.as_log_dict("core"))
                    log_wandb(
                        wandb_run,
                        payload,
                        global_step,
                    )
                distributed_barrier(dist_ctx)

            if args.save_every and global_step % args.save_every == 0:
                if dist_ctx.is_main:
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
                distributed_barrier(dist_ctx)

        if dist_ctx.is_main:
            val_metrics = evaluate(
                model,
                tokenizer,
                args,
                val_path=val_path,
                device=device,
                amp_dtype=amp_dtype,
                token_bytes=eval_token_bytes,
            )
            print(
                f"stage_done={stage_name} step={global_step} val_lm={val_metrics.lm_loss:.4f} "
                f"val_bpb={val_metrics.bpb:.6f} tokens={val_metrics.active_tokens:.0f} "
                f"bytes={val_metrics.bytes:.0f} batches={val_metrics.batches}",
                flush=True,
            )
            payload = val_metrics.as_log_dict("val")
            payload["stage/completed_index"] = stage_idx
            core_metrics = maybe_evaluate_core(
                model,
                tokenizer,
                args,
                device=device,
                amp_dtype=amp_dtype,
                step=global_step,
                force=True,
            )
            if core_metrics is not None:
                payload.update(core_metrics.as_log_dict("core"))
            log_wandb(
                wandb_run,
                payload,
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
        distributed_barrier(dist_ctx)

    if wandb_run is not None:
        wandb_run.finish()
    print_main(dist_ctx, "done.", flush=True)
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
