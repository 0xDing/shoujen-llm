"""Train Shoujen-LM on offline packed SFT parquet with STP auxiliary loss.

Typical flow:
    uv run python scripts/build_packed_tokenized_sft.py --block-size 2048
    uv run python scripts/train_sft_packed.py \
        --train-parquet data/sft-packed/train.parquet \
        --init-ckpt runs/pretrain/last.pt \
        --output runs/sft-packed \
        --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_config, lr_multiplier  # noqa: E402
from shoujen.data import PrepackedParquetSFTDataset, packed_sft_collate  # noqa: E402
from shoujen.losses import (  # noqa: E402
    SemanticTubePredictionLoss,
    compute_lm_loss,
    compute_z_loss,
)
from shoujen.model import ShoujenLM  # noqa: E402
from shoujen.optim import build_optimizers  # noqa: E402
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer  # noqa: E402
from shoujen.train_utils import (  # noqa: E402
    autocast_dtype,
    load_checkpoint,
    move_tensor_to_device,
    pick_device,
    save_checkpoint,
    set_optimizer_lr,
    set_optimizer_momentum,
    warmup_momentum,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", type=Path, default=Path("data/sft-packed/train.parquet"))
    p.add_argument("--output", required=True)
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="Hugging Face tokenizer id/URL or saved tokenizer directory",
    )
    p.add_argument("--init-ckpt", help="Resume / initialize from a checkpoint")
    p.add_argument("--config", help="JSON file overriding default model config")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--muon-lr", type=float, default=3e-4)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-momentum-start", type=float, default=0.85)
    p.add_argument("--muon-momentum-warmup", type=int, default=None)
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
    p.add_argument("--attention-window", type=int, default=256)

    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--stp-weight", type=float, default=0.042)
    p.add_argument("--stp-samples-per-span", type=int, default=1)
    p.add_argument("--stp-max-width", type=int, default=None)
    p.add_argument("--log-max-qk-logit", action="store_true")

    p.add_argument("--shuffle-buffer-size", type=int, default=1024)
    p.add_argument("--read-batch-size", type=int, default=1024)
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


def make_loader(args: argparse.Namespace, *, epoch: int) -> DataLoader:
    dataset = PrepackedParquetSFTDataset(
        args.train_parquet,
        args.block_size,
        shuffle=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed + epoch * 1009,
        repeat=False,
        read_batch_size=args.read_batch_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=packed_sft_collate,
        drop_last=True,
        pin_memory=False,
    )


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def shard_row_count(row_count: int, shard_index: int, shard_count: int) -> int:
    if shard_index >= row_count:
        return 0
    return ((row_count - 1 - shard_index) // shard_count) + 1


def full_batches_per_epoch(args: argparse.Namespace, row_count: int) -> int:
    shard_count = max(1, args.num_workers)
    return sum(
        shard_row_count(row_count, shard_index, shard_count) // args.batch_size
        for shard_index in range(shard_count)
    )


def maybe_use_checkpoint_config(args: argparse.Namespace) -> Path | None:
    if args.config or not args.init_ckpt:
        return None
    config_path = Path(args.init_ckpt).with_name("config.json")
    if not config_path.exists():
        return None
    args.config = str(config_path)
    return config_path


def maybe_init_wandb(args: argparse.Namespace, config: Any, rows_per_epoch: int, total_steps: int):
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
            "train_parquet": str(args.train_parquet),
            "rows_per_epoch": rows_per_epoch,
            "total_steps": total_steps,
            "model_config": config.to_dict(),
        },
    }
    if args.wandb_mode:
        kwargs["mode"] = args.wandb_mode
    return wandb.init(**{key: value for key, value in kwargs.items() if value is not None})


def log_wandb(run, payload: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(payload, step=step)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: move_tensor_to_device(value, device) for key, value in batch.items()}


def validate_batch(batch: dict[str, torch.Tensor], *, vocab_size: int) -> None:
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    valid_inputs = ((batch["input_ids"] >= 0) & (batch["input_ids"] < vocab_size)).all()
    valid_labels = ((labels == -100) | ((labels >= 0) & (labels < vocab_size))).all()
    valid_loss_mask = torch.isfinite(loss_mask).all() & (loss_mask >= 0).all() & (loss_mask <= 1).all()
    if not bool((valid_inputs & valid_labels & valid_loss_mask).detach().cpu().item()):
        raise RuntimeError("Invalid packed SFT batch; refusing to train on corrupted tensors.")


def main() -> None:
    args = parse_args()
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.stp_weight and args.stp_samples_per_span <= 0:
        raise SystemExit("--stp-samples-per-span must be positive when STP is enabled")

    rows_per_epoch = parquet_row_count(args.train_parquet)
    batches_per_epoch = full_batches_per_epoch(args, rows_per_epoch)
    if batches_per_epoch <= 0:
        raise SystemExit(
            f"No full training batches from {rows_per_epoch} rows; "
            "lower --batch-size or --num-workers"
        )
    steps_per_epoch = ceil_div(batches_per_epoch, args.gradient_accumulation_steps)
    total_steps = args.epochs * steps_per_epoch

    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    print(f"device={device} amp={amp_dtype}", flush=True)
    print(
        f"epochs={args.epochs} rows/epoch={rows_per_epoch} "
        f"batches/epoch={batches_per_epoch} optimizer_steps={total_steps}",
        flush=True,
    )

    tokenizer = ShoujenTokenizer.load(args.tokenizer)
    checkpoint_config_path = maybe_use_checkpoint_config(args)
    if checkpoint_config_path is not None:
        print(f"using checkpoint config {checkpoint_config_path}", flush=True)
    config = build_config(args, tokenizer)
    config.to_json(out / "config.json")

    model = ShoujenLM(config).to(device)
    if args.init_ckpt:
        try:
            state = load_checkpoint(args.init_ckpt, model, map_location=device, allow_appended_vocab=True)
        except RuntimeError as exc:
            message = str(exc)
            if "q_norm.weight" in message or "k_norm.weight" in message:
                raise SystemExit(
                    "Checkpoint appears to have been trained with qk_norm enabled. "
                    "Run with --qk-norm or provide the checkpoint's config.json via --config."
                ) from exc
            raise
        appended_vocab = state.get("_shoujen_appended_vocab_keys") or []
        if appended_vocab:
            details = ", ".join(
                f"{item['name']} {item['checkpoint_shape']} -> {item['model_shape']}"
                for item in appended_vocab
            )
            print(
                f"loaded ckpt with appended tokenizer rows; initialized new rows for {details}",
                flush=True,
            )
        print(f"loaded ckpt {args.init_ckpt} step={state.get('step', 0)}", flush=True)
    print(f"model: {model.num_parameters() / 1e6:.2f}M params", flush=True)
    wandb_run = maybe_init_wandb(args, config, rows_per_epoch, total_steps)

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
    base_muon_lrs = [group["lr"] for group in muon.param_groups]
    base_adamw_lrs = [group["lr"] for group in adamw.param_groups]
    muon_mom_warmup = args.warmup if args.muon_momentum_warmup is None else args.muon_momentum_warmup

    stp_loss_fn = SemanticTubePredictionLoss(
        samples_per_sequence=args.stp_samples_per_span,
        max_width=args.stp_max_width,
    )

    model.train()
    model.set_track_max_qk_logit(args.log_max_qk_logit)
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)

    step = 0
    accum_count = 0
    last_log_t = time.time()
    last_log_tokens = 0
    last_lm_loss = last_total_loss = last_stp_loss = last_z_loss = None

    def autocast():
        if amp_dtype is None:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    def optimizer_step() -> None:
        nonlocal accum_count

        grad_scale = args.gradient_accumulation_steps / accum_count
        if grad_scale != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(grad_scale)

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step + 1}: {grad_norm.item()}")
        muon.step()
        adamw.step()
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        accum_count = 0

    for epoch in range(args.epochs):
        loader = make_loader(args, epoch=epoch)
        saw_batch = False
        for batch in loader:
            saw_batch = True

            batch = move_batch(batch, device)
            validate_batch(batch, vocab_size=tokenizer.vocab_size)

            if accum_count == 0:
                mult = lr_multiplier(args, step, total_steps)
                set_optimizer_lr(muon, base_muon_lrs, mult)
                set_optimizer_lr(adamw, base_adamw_lrs, mult)
                mom_now = warmup_momentum(
                    step,
                    warmup=muon_mom_warmup,
                    start=args.muon_momentum_start,
                    end=args.muon_momentum,
                )
                set_optimizer_momentum(muon, mom_now)

            with autocast():
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    sequence_start_mask=batch["sequence_start_mask"],
                    use_cache=False,
                    output_hidden_states=bool(args.stp_weight),
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

                stp_loss = outputs.logits.new_zeros(())
                if args.stp_weight:
                    if not outputs.hidden_states:
                        raise RuntimeError("STP requires output_hidden_states=True")
                    stp_loss = stp_loss_fn(
                        outputs.hidden_states[-1],
                        batch["stp_spans"],
                        span_mask=batch["stp_span_mask"],
                    )
                    total_loss = total_loss + args.stp_weight * stp_loss

            (total_loss / args.gradient_accumulation_steps).backward()
            accum_count += 1
            last_lm_loss = lm_loss.detach()
            last_total_loss = total_loss.detach()
            last_stp_loss = stp_loss.detach()
            last_z_loss = z_loss.detach() if z_loss is not None else None
            last_log_tokens += int(batch["input_ids"].numel())

            if accum_count < args.gradient_accumulation_steps:
                continue

            optimizer_step()
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - last_log_t
                tps = last_log_tokens / max(dt, 1e-6)
                extras = (
                    f" stp={last_stp_loss.item():.4f}"
                    f" total={last_total_loss.item():.4f}"
                    f" active={active_tokens.item():.1f}"
                )
                if args.z_loss_weight and last_z_loss is not None:
                    extras += f" z={last_z_loss.item():.4f}"
                if args.log_max_qk_logit:
                    max_qk = model.max_qk_logit()
                    if max_qk is not None:
                        extras += f" max_qk={max_qk:.2f}"
                print(
                    f"step={step} lm={last_lm_loss.item():.4f}{extras} "
                    f"epoch={epoch + 1}/{args.epochs} lr_mult={mult:.3f} tok/s={tps:.0f}",
                    flush=True,
                )
                wandb_payload = {
                    "train/lm_loss": last_lm_loss.item(),
                    "train/stp_loss": last_stp_loss.item(),
                    "train/total_loss": last_total_loss.item(),
                    "train/active_tokens": active_tokens.item(),
                    "train/lr_multiplier": mult,
                    "train/tok_per_sec": tps,
                    "epoch/index": epoch,
                    "epoch/current": epoch + 1,
                }
                if args.z_loss_weight and last_z_loss is not None:
                    wandb_payload["train/z_loss"] = last_z_loss.item()
                if args.log_max_qk_logit:
                    max_qk = model.max_qk_logit()
                    if max_qk is not None:
                        wandb_payload["train/max_qk_logit"] = max_qk
                log_wandb(wandb_run, wandb_payload, step)
                last_log_t = time.time()
                last_log_tokens = 0

            if step % args.save_every == 0:
                save_checkpoint(
                    out / f"step{step}.pt",
                    model=model,
                    config=config,
                    step=step,
                    muon=muon,
                    adamw=adamw,
                )
                save_checkpoint(
                    out / "last.pt",
                    model=model,
                    config=config,
                    step=step,
                    muon=muon,
                    adamw=adamw,
                )
                print(f"saved {out / f'step{step}.pt'}", flush=True)
        if not saw_batch:
            raise RuntimeError(f"No training batches were produced by {args.train_parquet}")
        if accum_count:
            optimizer_step()
            step += 1
            if step % args.save_every == 0:
                save_checkpoint(
                    out / f"step{step}.pt",
                    model=model,
                    config=config,
                    step=step,
                    muon=muon,
                    adamw=adamw,
                )
                save_checkpoint(
                    out / "last.pt",
                    model=model,
                    config=config,
                    step=step,
                    muon=muon,
                    adamw=adamw,
                )
                print(f"saved {out / f'step{step}.pt'}", flush=True)
        print(f"finished epoch={epoch + 1}/{args.epochs} step={step}", flush=True)

    save_checkpoint(
        out / "last.pt",
        model=model,
        config=config,
        step=step,
        muon=muon,
        adamw=adamw,
    )
    print(f"done epochs={args.epochs} step={step} saved {out / 'last.pt'}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
