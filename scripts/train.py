"""Train Shoujen-LM (pretraining or SFT).

Examples:
    # Pretraining
    python scripts/train.py \
        --mode pretrain \
        --vocab data/vocab.json \
        --corpus data/corpus.jsonl \
        --output runs/pretrain \
        --batch-size 4 --block-size 1024 --max-steps 20000

    # SFT (initialise from a pretraining checkpoint)
    python scripts/train.py \
        --mode sft \
        --vocab data/vocab.json \
        --sft data/sft.jsonl \
        --init-ckpt runs/pretrain/last.pt \
        --output runs/sft \
        --batch-size 4 --block-size 1024 --max-steps 5000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.config import ShoujenConfig
from shoujen.data import PackedPretrainDataset, SFTDataset, collate
from shoujen.losses import compute_lm_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.tokenizer import ShoujenTokenizer
from shoujen.train_utils import (
    autocast_dtype,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    set_optimizer_lr,
    warmup_cosine_lr,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    p.add_argument("--vocab", required=True, help="Path to tokenizer vocab.json")
    p.add_argument("--corpus", help="Path to pretraining corpus (jsonl/dir/.txt)")
    p.add_argument("--sft", help="Path to SFT jsonl")
    p.add_argument("--cache", help="Path to pretokenized memmap (pretraining only)")
    p.add_argument("--output", required=True)
    p.add_argument("--init-ckpt", help="Resume / initialize from a checkpoint")

    p.add_argument("--config", help="JSON file overriding default model config")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--muon-lr", type=float, default=3e-4)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--muon-wd", type=float, default=0.0)
    p.add_argument("--adamw-wd", type=float, default=0.1)
    p.add_argument("--adamw-foreach", action="store_true")
    p.add_argument("--lr-min-ratio", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_config(args, tokenizer: ShoujenTokenizer) -> ShoujenConfig:
    if args.config:
        cfg = ShoujenConfig.from_json(args.config)
    else:
        cfg = ShoujenConfig()
    original_vocab_size = cfg.vocab_size
    cfg.vocab_size = tokenizer.vocab_size
    if (
        cfg.vocab_size_per_layer_input == original_vocab_size
        or cfg.vocab_size_per_layer_input < tokenizer.vocab_size
    ):
        cfg.vocab_size_per_layer_input = tokenizer.vocab_size
    cfg.pad_token_id = tokenizer.pad_id
    cfg.eos_token_id = tokenizer.eos_id
    cfg.im_start_token_id = tokenizer.im_start_id
    cfg.im_end_token_id = tokenizer.im_end_id
    cfg.max_seq_len = max(cfg.max_seq_len, args.block_size)
    return cfg


def build_dataset(args, tokenizer):
    if args.mode == "pretrain":
        if not args.corpus:
            raise SystemExit("--corpus is required for pretraining")
        return PackedPretrainDataset(
            args.corpus, tokenizer, block_size=args.block_size, cache_path=args.cache
        )
    if not args.sft:
        raise SystemExit("--sft is required for sft mode")
    return SFTDataset(args.sft, tokenizer, block_size=args.block_size)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    print(f"device={device} amp={amp_dtype}", flush=True)

    tokenizer = ShoujenTokenizer.load(args.vocab)
    config = build_config(args, tokenizer)
    config.to_json(out / "config.json")

    model = ShoujenLM(config).to(device)
    print(f"model: {model.num_parameters() / 1e6:.2f}M params", flush=True)

    muon, adamw = build_optimizers(
        model,
        muon_lr=args.muon_lr,
        muon_ns_steps=args.muon_ns_steps,
        muon_wd=args.muon_wd,
        adamw_lr=args.adamw_lr,
        adamw_wd=args.adamw_wd,
        adamw_foreach=True if args.adamw_foreach else None,
    )
    base_muon_lrs = [g["lr"] for g in muon.param_groups]
    base_adamw_lrs = [g["lr"] for g in adamw.param_groups]

    start_step = 0
    if args.init_ckpt:
        state = load_checkpoint(args.init_ckpt, model, map_location=device)
        if args.mode == "pretrain" and "muon" in state:
            muon.load_state_dict(state["muon"])
        if args.mode == "pretrain" and "adamw" in state:
            try:
                adamw.load_state_dict(state["adamw"])
            except ValueError:
                print("WARN: AdamW state shape/group mismatch — skipping.")
        start_step = state.get("step", 0) if args.mode == "pretrain" else 0
        print(f"loaded ckpt {args.init_ckpt} step={start_step}", flush=True)

    dataset = build_dataset(args, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model.train()

    step = start_step
    last_log_t = time.time()
    last_log_tokens = 0

    def autocast():
        if amp_dtype is None:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            mult = warmup_cosine_lr(
                step, warmup=args.warmup, max_steps=args.max_steps, min_ratio=args.lr_min_ratio
            )
            set_optimizer_lr(muon, base_muon_lrs, mult)
            set_optimizer_lr(adamw, base_adamw_lrs, mult)

            with autocast():
                out_m = model(input_ids, use_cache=False)
                lm_loss, _ = compute_lm_loss(
                    out_m.logits, labels, loss_mask=loss_mask, ignore_index=-100
                )
                total_loss = lm_loss

            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step()
            adamw.step()

            step += 1
            last_log_tokens += input_ids.numel()

            if step % args.log_every == 0:
                dt = time.time() - last_log_t
                tps = last_log_tokens / max(dt, 1e-6)
                print(
                    f"step={step} lm={lm_loss.item():.4f} lr_mult={mult:.3f} tok/s={tps:.0f}",
                    flush=True,
                )
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

    save_checkpoint(
        out / "last.pt",
        model=model,
        config=config,
        step=step,
        muon=muon,
        adamw=adamw,
    )
    print("done.", flush=True)


if __name__ == "__main__":
    main()
