"""Trainer-based staged packed pretraining.

This keeps the staged parquet data path and the project's Muon+AdamW optimizer
split, while delegating batch sizing, gradient accumulation, checkpointing,
logging, evaluation, and checkpoint save/resume mechanics to Transformers
Trainer.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_config
from scripts.train_staged_packed import (
    DEFAULT_TRAIN_FILES,
    evaluate as evaluate_staged,
    parse_core_task_labels,
    resolve_data_path,
)
from shoujen.data import PackedParquetPretrainDataset, packed_collate
from shoujen.evaluation import evaluate_core, token_byte_lengths
from shoujen.losses import compute_lm_loss, compute_z_loss
from shoujen.model import ShoujenLM
from shoujen.optim import MultipleOptimizer, build_optimizers
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer
from shoujen.train_utils import autocast_dtype, load_checkpoint, pick_device, warmup_cosine_lr, warmup_stable_decay_lr


class StagedPackedParquetDataset(IterableDataset):
    def __init__(
        self,
        paths: list[Path],
        tokenizer: ShoujenTokenizer,
        *,
        block_size: int,
        text_column: str,
        shuffle: bool,
        shuffle_buffer_size: int,
        seed: int,
        max_samples: int = 0,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        emitted = 0
        for stage_idx, path in enumerate(self.paths):
            dataset = PackedParquetPretrainDataset(
                path,
                self.tokenizer,
                block_size=self.block_size,
                text_column=self.text_column,
                shuffle=self.shuffle,
                shuffle_buffer_size=self.shuffle_buffer_size,
                seed=self.seed + stage_idx,
            )
            for sample in dataset:
                yield sample
                emitted += 1
                if self.max_samples and emitted >= self.max_samples:
                    return


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--output", default="runs/staged-packed-trainer")
    p.add_argument("--init-ckpt", help="Initialize from the legacy .pt checkpoint format")
    p.add_argument("--resume-from-checkpoint", help="Resume from a Trainer checkpoint directory")
    p.add_argument("--config", help="JSON file overriding default model config")

    p.add_argument("--batch-size", type=int, default=4, help="Per-device micro-batch size")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--block-size", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=20000)
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
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--lr-stable-steps", type=int, default=0)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--log-max-qk-logit", action="store_true")

    p.add_argument("--shuffle-buffer-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--disable-tqdm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--report-to", default="none")
    p.add_argument("--wandb-project", default="shoujen-llm")
    args = p.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps must be positive")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive for Trainer over streaming parquet datasets")
    if args.eval_batches < 0:
        raise SystemExit("--eval-batches cannot be negative")
    return args


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


class ShoujenTrainer(Trainer):
    def __init__(
        self,
        *trainer_args: Any,
        shoujen_args: argparse.Namespace,
        shoujen_tokenizer: ShoujenTokenizer,
        val_path: Path,
        eval_token_bytes: torch.Tensor,
        amp_dtype: torch.dtype | None,
        **trainer_kwargs: Any,
    ):
        super().__init__(*trainer_args, **trainer_kwargs)
        self.shoujen_args = shoujen_args
        self.shoujen_tokenizer = shoujen_tokenizer
        self.val_path = val_path
        self.eval_token_bytes = eval_token_bytes
        self.amp_dtype = amp_dtype
        self.model_accepts_loss_kwargs = False

    def create_optimizer(self, model=None):  # type: ignore[override]
        if self.optimizer is None:
            model = self.model if model is None else model
            muon, adamw = build_optimizers(
                model,
                muon_lr=self.shoujen_args.muon_lr,
                muon_ns_steps=self.shoujen_args.muon_ns_steps,
                muon_wd=self.shoujen_args.muon_wd,
                muon_adaptive=self.shoujen_args.muon_adaptive,
                muon_adaptive_beta2=self.shoujen_args.muon_adaptive_beta2,
                muon_adaptive_eps=self.shoujen_args.muon_adaptive_eps,
                adamw_lr=self.shoujen_args.adamw_lr,
                adamw_wd=self.shoujen_args.adamw_wd,
                adamw_embed_wd=self.shoujen_args.adamw_embed_wd
                if self.shoujen_args.adamw_independent_wd
                else None,
                adamw_foreach=True if self.shoujen_args.adamw_foreach else None,
            )
            self.optimizer = MultipleOptimizer([muon, adamw])
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):  # type: ignore[override]
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            schedule_steps = max(1, self.shoujen_args.lr_schedule_steps or num_training_steps)
            self.lr_scheduler = LambdaLR(
                optimizer,
                lambda step: lr_multiplier(self.shoujen_args, step, schedule_steps),
            )
        return self.lr_scheduler

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        loss_mask = inputs.pop("loss_mask", None)
        inputs.pop("seq_ids", None)

        device_type = labels.device.type
        autocast_enabled = self.amp_dtype is not None and device_type != "cpu"
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=autocast_enabled):
            outputs = model(**inputs, use_cache=False)
            lm_loss, active_tokens = compute_lm_loss(
                outputs.logits.float(),
                labels,
                loss_mask=loss_mask,
                ignore_index=-100,
            )
            loss = lm_loss
            if self.shoujen_args.z_loss_weight:
                z_loss = compute_z_loss(
                    outputs.logits,
                    labels,
                    loss_mask=loss_mask,
                    ignore_index=-100,
                )
                loss = loss + self.shoujen_args.z_loss_weight * z_loss

        if self.shoujen_args.log_max_qk_logit and hasattr(model, "max_qk_logit"):
            max_qk = model.max_qk_logit()
            if max_qk is not None:
                self.log({"train/max_qk_logit": max_qk})
        self.log({"train/active_tokens": float(active_tokens.detach().float().item())})
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:  # type: ignore[override]
        metrics = super().evaluate(*args, **kwargs)
        staged_metrics = evaluate_staged(
            self.model,
            self.shoujen_tokenizer,
            self.shoujen_args,
            val_path=self.val_path,
            device=self.args.device,
            amp_dtype=self.amp_dtype,
            token_bytes=self.eval_token_bytes,
        )
        extra: dict[str, float] = {
            "eval_lm_loss_staged": staged_metrics.lm_loss,
            "eval_active_tokens": staged_metrics.active_tokens,
            "eval_batches": float(staged_metrics.batches),
        }
        if math.isfinite(staged_metrics.bpb):
            extra["eval_bpb"] = staged_metrics.bpb
            extra["eval_bytes"] = staged_metrics.bytes

        if (
            self.shoujen_args.core_eval_dir is not None
            and self.shoujen_args.core_metric_every > 0
            and self.state.global_step > 0
            and self.state.global_step % self.shoujen_args.core_metric_every == 0
        ):
            core = evaluate_core(
                self.model,
                self.shoujen_tokenizer,
                eval_dir=self.shoujen_args.core_eval_dir,
                device=self.args.device,
                amp_dtype=self.amp_dtype,
                max_per_task=self.shoujen_args.core_metric_max_per_task,
                task_labels=parse_core_task_labels(self.shoujen_args.core_tasks),
                seed=self.shoujen_args.seed,
            )
            extra["core_metric"] = core.score
            for label, value in core.results.items():
                key = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()
                extra[f"core_accuracy/{key}"] = value
            for label, value in core.centered_results.items():
                key = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()
                extra[f"core_centered/{key}"] = value

        self.log(extra)
        metrics.update(extra)
        print(
            f"eval_extra step={self.state.global_step} val_lm={staged_metrics.lm_loss:.4f} "
            f"val_bpb={staged_metrics.bpb:.6f} tokens={staged_metrics.active_tokens:.0f} "
            f"bytes={staged_metrics.bytes:.0f}",
            flush=True,
        )
        return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [resolve_data_path(args.data_dir, p) for p in args.train_files]
    val_path = resolve_data_path(args.data_dir, args.val_file)
    missing = [p for p in [*train_paths, val_path] if not p.exists()]
    if missing:
        raise SystemExit("Missing parquet file(s): " + ", ".join(str(p) for p in missing))

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)

    tokenizer = ShoujenTokenizer.load(args.tokenizer)
    eval_token_bytes = token_byte_lengths(tokenizer, device=device)
    config = build_config(args, tokenizer)
    if args.qk_norm:
        config.qk_norm = True
    config.use_cache = False
    config.to_json(out_dir / "config.json")

    model = ShoujenLM(config)
    if args.init_ckpt:
        state = load_checkpoint(args.init_ckpt, model, map_location="cpu")
        print(f"loaded legacy ckpt {args.init_ckpt} step={state.get('step', 0)}", flush=True)
    model.set_track_max_qk_logit(args.log_max_qk_logit)

    train_dataset = StagedPackedParquetDataset(
        train_paths,
        tokenizer,
        block_size=args.block_size,
        text_column=args.text_column,
        shuffle=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
    )
    eval_dataset = StagedPackedParquetDataset(
        [val_path],
        tokenizer,
        block_size=args.block_size,
        text_column=args.text_column,
        shuffle=False,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        max_samples=args.eval_batches * args.batch_size if args.eval_batches else 0,
    )

    report_to = [] if args.report_to in {"none", ""} else [args.report_to]
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        max_grad_norm=args.grad_clip,
        logging_strategy="steps",
        logging_steps=args.log_every,
        logging_first_step=True,
        eval_strategy="steps" if args.eval_every else "no",
        eval_steps=args.eval_every if args.eval_every else None,
        save_strategy="steps" if args.save_every else "no",
        save_steps=args.save_every if args.save_every else 500,
        save_total_limit=3,
        prediction_loss_only=True,
        dataloader_drop_last=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=device.type == "cuda",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=report_to,
        project=args.wandb_project,
        run_name=out_dir.name,
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=args.disable_tqdm,
        fp16=False,
        bf16=False,
        skip_memory_metrics=False,
        use_cpu=device.type == "cpu",
    )

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(
        f"device={device} amp={amp_dtype} trainer=true batch={args.batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} effective_batch={effective_batch} "
        f"gradient_checkpointing={args.gradient_checkpointing}",
        flush=True,
    )
    print("train stages: " + " -> ".join(p.name for p in train_paths), flush=True)
    print(f"validation: {val_path.name}", flush=True)
    print(f"model: {model.num_parameters() / 1e6:.2f}M params", flush=True)

    trainer = ShoujenTrainer(
        model=model,
        args=training_args,
        data_collator=packed_collate,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        shoujen_args=args,
        shoujen_tokenizer=tokenizer,
        val_path=val_path,
        eval_token_bytes=eval_token_bytes,
        amp_dtype=amp_dtype,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(out_dir / "final"))

    if trainer.state.log_history:
        best_eval = [
            row["eval_loss"]
            for row in trainer.state.log_history
            if "eval_loss" in row and math.isfinite(float(row["eval_loss"]))
        ]
        if best_eval:
            print(f"best_eval_loss={min(best_eval):.4f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
