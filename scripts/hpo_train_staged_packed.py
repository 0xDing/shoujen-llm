"""Hyperparameter search for the staged packed pretraining flow.

This follows the same shape as Hugging Face Trainer.hyperparameter_search:
sample a trial-specific hyperparameter space, reinitialize the model for every
trial, train for a short budget, evaluate, and optimize an objective.

Examples:
    # Dependency-free random search, useful for quick smoke runs.
    uv run python scripts/hpo_train_staged_packed.py \
        --backend random --n-trials 4 --trial-steps 80 --eval-every 40

    # Optuna search after installing the hpo extra.
    uv sync --extra hpo
    uv run python scripts/hpo_train_staged_packed.py \
        --backend optuna --n-trials 20 --trial-steps 300 --eval-every 100
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.train import build_config
from scripts.train_staged_packed import (
    DEFAULT_TRAIN_FILES,
    evaluate,
    lr_multiplier,
    make_loader,
    move_batch,
    resolve_data_path,
)
from shoujen.losses import compute_lm_loss, compute_z_loss
from shoujen.model import ShoujenLM
from shoujen.optim import build_optimizers
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer
from shoujen.train_utils import (
    autocast_dtype,
    pick_device,
    set_optimizer_lr,
    set_optimizer_momentum,
    warmup_momentum,
)


MIN_VALID_LOSS = -1e-3


class TrialFailed(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RandomSearchTrial:
    """Small Optuna-compatible subset for local dependency-free searches."""

    def __init__(self, number: int, rng: random.Random):
        self.number = number
        self.rng = rng
        self.params: dict[str, Any] = {}
        self.user_attrs: dict[str, Any] = {}

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = self.rng.choice(list(choices))
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        if log:
            value = math.exp(self.rng.uniform(math.log(low), math.log(high)))
        else:
            value = self.rng.uniform(low, high)
        self.params[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        return None

    def should_prune(self) -> bool:
        return False

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"1", "true", "yes", "on"}:
            values.append(True)
        elif item in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected boolean, got {item!r}")
    if not values:
        raise argparse.ArgumentTypeError("expected at least one boolean")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--vocab",
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="Hugging Face tokenizer id/URL or legacy local vocab.json",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/processed-clean"))
    p.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES)
    p.add_argument("--val-file", default="s0-val.parquet")
    p.add_argument("--text-column", default="text")
    p.add_argument("--config", help="JSON file overriding default model config")
    p.add_argument("--init-ckpt", help="Initialize every trial from the same checkpoint weights")
    p.add_argument("--strict-init-ckpt", action="store_true")

    p.add_argument("--output", type=Path, default=Path("runs/hpo-staged-packed"))
    p.add_argument("--backend", choices=["optuna", "random"], default="optuna")
    p.add_argument("--study-name", default="shoujen-staged-packed-hpo")
    p.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///runs/hpo.db")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds")
    p.add_argument("--multi-objective", action="store_true")
    p.add_argument("--seeds-per-trial", type=int, default=1)
    p.add_argument("--seed-std-weight", type=float, default=0.15)

    p.add_argument("--trial-steps", type=int, default=300)
    p.add_argument("--trial-steps-per-stage", type=int, default=0)
    p.add_argument(
        "--balanced-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap each stage so short trials sample all staged shards.",
    )
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle-buffer-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--amp-choices",
        type=parse_str_list,
        default=parse_str_list("auto,none"),
        help="Trial AMP choices. auto uses the same autocast dtype as formal training; none emits --no-amp.",
    )
    p.add_argument("--track-qk", action="store_true")
    p.add_argument("--qk-logit-limit", type=float, default=80.0)

    p.add_argument("--speed-weight", type=float, default=0.03)
    p.add_argument("--stability-weight", type=float, default=0.20)

    p.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("2,4,8"))
    p.add_argument("--gradient-accumulation-steps", type=parse_int_list, default=parse_int_list("1"))
    p.add_argument("--block-sizes", type=parse_int_list, default=parse_int_list("1024,1536,2048"))
    p.add_argument("--muon-ns-steps", type=parse_int_list, default=parse_int_list("3,5,7"))
    p.add_argument("--grad-clips", type=parse_float_list, default=parse_float_list("0.5,1.0,2.0"))
    p.add_argument("--z-loss-weights", type=parse_float_list, default=parse_float_list("0,1e-5,1e-4,3e-4"))
    p.add_argument("--adamw-wd-choices", type=parse_float_list, default=parse_float_list("0.01,0.05,0.1,0.2"))
    p.add_argument("--adamw-independent-wd-choices", type=parse_bool_list, default=parse_bool_list("true"))
    p.add_argument("--adamw-embed-wd-choices", type=parse_float_list, default=parse_float_list("0,0.01,0.05"))
    p.add_argument("--muon-wd-choices", type=parse_float_list, default=parse_float_list("0,0.01"))
    p.add_argument("--lr-schedules", type=parse_str_list, default=parse_str_list("cosine,wsd"))
    p.add_argument("--qk-norm-choices", type=parse_bool_list, default=parse_bool_list("false,true"))
    p.add_argument("--muon-adaptive-choices", type=parse_bool_list, default=parse_bool_list("true,false"))
    p.add_argument("--adamw-foreach-choices", type=parse_bool_list, default=parse_bool_list("false,true"))

    p.add_argument("--muon-lr-min", type=float, default=1e-4)
    p.add_argument("--muon-lr-max", type=float, default=8e-4)
    p.add_argument("--adamw-lr-min", type=float, default=5e-5)
    p.add_argument("--adamw-lr-max", type=float, default=6e-4)
    p.add_argument(
        "--muon-momentum-start-min",
        type=float,
        default=0.70,
        help="Lower bound for the searched initial Muon momentum (start of linear warmup to --muon-momentum-target).",
    )
    p.add_argument(
        "--muon-momentum-start-max",
        type=float,
        default=0.95,
        help="Upper bound for the searched initial Muon momentum.",
    )
    p.add_argument(
        "--muon-momentum-target",
        type=float,
        default=0.95,
        help="Fixed Muon momentum after warmup completes (not searched).",
    )
    p.add_argument(
        "--attention-window",
        type=int,
        default=256,
        help="Sliding-window size (W tokens) applied to all attention layers in trials and formal training. Pass 0 to disable.",
    )
    p.add_argument(
        "--adamw-beta1",
        type=float,
        default=0.8,
        help="AdamW beta1 used in trials and formal training (not searched).",
    )
    p.add_argument(
        "--adamw-beta2",
        type=float,
        default=0.95,
        help="AdamW beta2 used in trials and formal training (not searched).",
    )
    p.add_argument("--warmup-ratio-min", type=float, default=0.02)
    p.add_argument("--warmup-ratio-max", type=float, default=0.10)
    p.add_argument("--lr-min-ratio-min", type=float, default=0.03)
    p.add_argument("--lr-min-ratio-max", type=float, default=0.30)
    p.add_argument("--lr-stable-ratio-min", type=float, default=0.45)
    p.add_argument("--lr-stable-ratio-max", type=float, default=0.80)

    p.add_argument("--formal-output", default="runs/staged-packed-hpo-best")
    p.add_argument("--formal-max-steps", type=int, default=0)
    p.add_argument("--formal-max-steps-per-stage", type=int, default=0)
    p.add_argument("--formal-lr-schedule-steps", type=int, default=20000)
    p.add_argument("--formal-save-every", type=int, default=1000)
    p.add_argument("--formal-eval-every", type=int, default=500)
    p.add_argument("--formal-eval-batches", type=int, default=50)
    p.add_argument("--formal-log-every", type=int, default=20)
    p.add_argument("--formal-wandb", action="store_true")
    p.add_argument("--formal-wandb-project", default="shoujen-llm")

    args = p.parse_args()
    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be positive")
    if args.trial_steps <= 0:
        raise SystemExit("--trial-steps must be positive")
    if args.eval_every <= 0:
        raise SystemExit("--eval-every must be positive")
    if args.seeds_per_trial <= 0:
        raise SystemExit("--seeds-per-trial must be positive")
    if any(value <= 0 for value in args.gradient_accumulation_steps):
        raise SystemExit("--gradient-accumulation-steps values must be positive")
    if not set(args.lr_schedules).issubset({"cosine", "wsd"}):
        raise SystemExit("--lr-schedules may contain only cosine,wsd")
    if not set(args.amp_choices).issubset({"auto", "none"}):
        raise SystemExit("--amp-choices may contain only auto,none")
    return args


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_device_cache() -> None:
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Run `uv sync --extra hpo` after updating the lockfile, "
            "or use `--backend random` for dependency-free local search."
        ) from exc
    return optuna


def suggest_hparams(trial: Any, args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "batch_size": trial.suggest_categorical("batch_size", args.batch_sizes),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps",
            args.gradient_accumulation_steps,
        ),
        "block_size": trial.suggest_categorical("block_size", args.block_sizes),
        "amp": "none" if args.no_amp else trial.suggest_categorical("amp", args.amp_choices),
        "muon_lr": trial.suggest_float("muon_lr", args.muon_lr_min, args.muon_lr_max, log=True),
        "adamw_lr": trial.suggest_float("adamw_lr", args.adamw_lr_min, args.adamw_lr_max, log=True),
        "muon_ns_steps": trial.suggest_categorical("muon_ns_steps", args.muon_ns_steps),
        "muon_adaptive": trial.suggest_categorical("muon_adaptive", args.muon_adaptive_choices),
        "muon_wd": trial.suggest_categorical("muon_wd", args.muon_wd_choices),
        "adamw_wd": trial.suggest_categorical("adamw_wd", args.adamw_wd_choices),
        "adamw_independent_wd": trial.suggest_categorical(
            "adamw_independent_wd",
            args.adamw_independent_wd_choices,
        ),
        "adamw_embed_wd": trial.suggest_categorical("adamw_embed_wd", args.adamw_embed_wd_choices),
        "adamw_foreach": trial.suggest_categorical("adamw_foreach", args.adamw_foreach_choices),
        "lr_schedule": trial.suggest_categorical("lr_schedule", args.lr_schedules),
        "warmup_ratio": trial.suggest_float("warmup_ratio", args.warmup_ratio_min, args.warmup_ratio_max),
        "lr_min_ratio": trial.suggest_float("lr_min_ratio", args.lr_min_ratio_min, args.lr_min_ratio_max),
        "lr_stable_ratio": trial.suggest_float(
            "lr_stable_ratio",
            args.lr_stable_ratio_min,
            args.lr_stable_ratio_max,
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", args.grad_clips),
        "qk_norm": trial.suggest_categorical("qk_norm", args.qk_norm_choices),
        "z_loss_weight": trial.suggest_categorical("z_loss_weight", args.z_loss_weights),
    }
    if params["muon_adaptive"]:
        params["muon_adaptive_beta2"] = trial.suggest_categorical(
            "muon_adaptive_beta2",
            [0.90, 0.95, 0.98],
        )
    else:
        params["muon_adaptive_beta2"] = 0.95
    params["muon_momentum_start"] = trial.suggest_float(
        "muon_momentum_start",
        args.muon_momentum_start_min,
        args.muon_momentum_start_max,
    )
    return params


def make_trial_train_args(
    search_args: argparse.Namespace,
    params: dict[str, Any],
    *,
    seed: int,
) -> argparse.Namespace:
    schedule_steps = max(1, search_args.trial_steps)
    return argparse.Namespace(
        vocab=search_args.vocab,
        data_dir=search_args.data_dir,
        train_files=search_args.train_files,
        val_file=search_args.val_file,
        text_column=search_args.text_column,
        output=str(search_args.output),
        init_ckpt=search_args.init_ckpt,
        config=search_args.config,
        batch_size=int(params["batch_size"]),
        gradient_accumulation_steps=int(params["gradient_accumulation_steps"]),
        block_size=int(params["block_size"]),
        max_steps=int(search_args.trial_steps),
        max_steps_per_stage=0,
        lr_schedule_steps=schedule_steps,
        warmup=max(0, int(round(schedule_steps * float(params["warmup_ratio"])))),
        save_every=0,
        eval_every=int(search_args.eval_every),
        eval_batches=int(search_args.eval_batches),
        log_every=int(search_args.log_every),
        grad_clip=float(params["grad_clip"]),
        muon_lr=float(params["muon_lr"]),
        muon_momentum=float(search_args.muon_momentum_target),
        muon_momentum_start=float(params["muon_momentum_start"]),
        muon_momentum_warmup=None,
        muon_ns_steps=int(params["muon_ns_steps"]),
        muon_adaptive=bool(params["muon_adaptive"]),
        muon_adaptive_beta2=float(params["muon_adaptive_beta2"]),
        muon_adaptive_eps=1e-8,
        adamw_lr=float(params["adamw_lr"]),
        adamw_beta1=float(search_args.adamw_beta1),
        adamw_beta2=float(search_args.adamw_beta2),
        muon_wd=float(params["muon_wd"]),
        adamw_wd=float(params["adamw_wd"]),
        adamw_independent_wd=bool(params["adamw_independent_wd"]),
        adamw_embed_wd=float(params["adamw_embed_wd"]),
        adamw_foreach=bool(params["adamw_foreach"]),
        lr_schedule=str(params["lr_schedule"]),
        lr_min_ratio=float(params["lr_min_ratio"]),
        lr_stable_steps=max(0, int(round(schedule_steps * float(params["lr_stable_ratio"])))),
        qk_norm=bool(params["qk_norm"]),
        attention_window=int(search_args.attention_window),
        z_loss_weight=float(params["z_loss_weight"]),
        log_max_qk_logit=bool(search_args.track_qk),
        shuffle_buffer_size=int(search_args.shuffle_buffer_size),
        seed=seed,
        num_workers=int(search_args.num_workers),
        no_amp=bool(search_args.no_amp or params.get("amp") == "none"),
        device=search_args.device,
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_run_name=None,
        wandb_mode=None,
    )


def stage_step_cap(search_args: argparse.Namespace, num_stages: int) -> int:
    if search_args.trial_steps_per_stage > 0:
        return search_args.trial_steps_per_stage
    if search_args.balanced_stages:
        return max(1, math.ceil(search_args.trial_steps / max(1, num_stages)))
    return 0


def load_model_init_checkpoint(
    model: ShoujenLM,
    ckpt_path: str | Path,
    *,
    device: torch.device,
    strict: bool,
) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=strict)
    if missing or unexpected:
        print(
            f"init_ckpt loaded with missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )


def stability_penalty(
    *,
    eval_losses: list[float],
    train_losses: list[float],
    grad_norms: list[float],
    clip_count: int,
    steps: int,
    max_qk_logit: float | None,
    train_args: argparse.Namespace,
    search_args: argparse.Namespace,
) -> float:
    penalty = 0.0
    if steps <= 0:
        return 1000.0

    clip_ratio = clip_count / max(1, steps)
    penalty += clip_ratio

    finite_grad_norms = [x for x in grad_norms if math.isfinite(x)]
    if finite_grad_norms and train_args.grad_clip > 0:
        avg_grad = mean(finite_grad_norms)
        max_grad = max(finite_grad_norms)
        penalty += 0.10 * max(0.0, avg_grad / train_args.grad_clip - 1.0)
        penalty += 0.05 * max(0.0, math.log(max(max_grad / train_args.grad_clip, 1.0)))

    finite_eval_losses = [x for x in eval_losses if math.isfinite(x)]
    if len(finite_eval_losses) >= 2:
        best = min(finite_eval_losses)
        final = finite_eval_losses[-1]
        baseline = max(best, 1.0)
        penalty += max(0.0, final - best) / baseline
        penalty += pstdev(finite_eval_losses) / max(mean(finite_eval_losses), 1.0)

    finite_train_losses = [x for x in train_losses if math.isfinite(x)]
    if len(finite_train_losses) >= 2:
        penalty += 0.05 * pstdev(finite_train_losses) / max(mean(finite_train_losses), 1.0)

    if max_qk_logit is not None and search_args.qk_logit_limit > 0:
        penalty += max(0.0, max_qk_logit - search_args.qk_logit_limit) / search_args.qk_logit_limit

    return penalty


def objective_score(
    *,
    val_loss: float,
    tok_per_sec: float,
    stability: float,
    args: argparse.Namespace,
) -> float:
    if not math.isfinite(val_loss) or val_loss < MIN_VALID_LOSS:
        return 1e9
    speed_term = -args.speed_weight * math.log(max(tok_per_sec, 1.0))
    stability_term = args.stability_weight * stability
    return val_loss + speed_term + stability_term


def checked_loss(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise TrialFailed(f"non-finite {name} loss")
    if value < MIN_VALID_LOSS:
        raise TrialFailed(f"invalid negative {name} loss")
    return max(0.0, value)


def maybe_report_pruning(
    trial: Any,
    *,
    args: argparse.Namespace,
    step: int,
    val_loss: float,
    tok_per_sec: float,
    stability: float,
) -> None:
    if args.backend != "optuna" or args.multi_objective:
        return
    trial.report(
        objective_score(val_loss=val_loss, tok_per_sec=tok_per_sec, stability=stability, args=args),
        step,
    )
    if trial.should_prune():
        optuna = load_optuna()
        raise optuna.TrialPruned()


def run_one_seed(
    *,
    search_args: argparse.Namespace,
    params: dict[str, Any],
    seed: int,
    tokenizer: ShoujenTokenizer,
    train_paths: list[Path],
    val_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    trial_number: int,
    trial: Any,
) -> dict[str, Any]:
    train_args = make_trial_train_args(search_args, params, seed=seed)
    trial_amp_dtype = None if train_args.no_amp else amp_dtype
    seed_all(seed)

    config = build_config(train_args, tokenizer)
    model = ShoujenLM(config).to(device)
    if search_args.init_ckpt:
        load_model_init_checkpoint(
            model,
            search_args.init_ckpt,
            device=device,
            strict=search_args.strict_init_ckpt,
        )
    model.train()
    model.set_track_max_qk_logit(train_args.log_max_qk_logit)

    muon, adamw = build_optimizers(
        model,
        muon_lr=train_args.muon_lr,
        muon_momentum=train_args.muon_momentum,
        muon_ns_steps=train_args.muon_ns_steps,
        muon_wd=train_args.muon_wd,
        muon_adaptive=train_args.muon_adaptive,
        muon_adaptive_beta2=train_args.muon_adaptive_beta2,
        muon_adaptive_eps=train_args.muon_adaptive_eps,
        adamw_lr=train_args.adamw_lr,
        adamw_betas=(train_args.adamw_beta1, train_args.adamw_beta2),
        adamw_wd=train_args.adamw_wd,
        adamw_embed_wd=train_args.adamw_embed_wd if train_args.adamw_independent_wd else None,
        adamw_foreach=True if train_args.adamw_foreach else None,
    )
    base_muon_lrs = [group["lr"] for group in muon.param_groups]
    base_adamw_lrs = [group["lr"] for group in adamw.param_groups]
    muon_mom_warmup = (
        train_args.warmup if train_args.muon_momentum_warmup is None else train_args.muon_momentum_warmup
    )

    initial_val_loss = None
    if search_args.eval_initial:
        initial_val_loss, _, _ = evaluate(
            model,
            tokenizer,
            train_args,
            val_path=val_path,
            device=device,
            amp_dtype=trial_amp_dtype,
        )

    schedule_steps = max(1, train_args.lr_schedule_steps)
    max_steps_per_stage = stage_step_cap(search_args, len(train_paths))

    eval_records: list[dict[str, Any]] = []
    eval_losses: list[float] = []
    train_losses: list[float] = []
    grad_norms: list[float] = []
    clip_count = 0
    global_step = 0
    tokens_seen = 0
    max_qk_logit = None
    start_time = time.time()
    last_log_t = start_time
    last_log_tokens = 0

    try:
        stop_training = False
        for stage_idx, stage_path in enumerate(train_paths):
            if stop_training:
                break
            loader = make_loader(
                stage_path,
                tokenizer,
                train_args,
                shuffle=True,
                seed=seed + stage_idx,
                device=device,
                drop_last=True,
            )
            stage_step = 0
            accum_count = 0
            step_train_losses: list[float] = []
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            for batch in loader:
                if global_step >= search_args.trial_steps:
                    stop_training = True
                    break
                if max_steps_per_stage and stage_step >= max_steps_per_stage:
                    break

                batch = move_batch(
                    batch,
                    device,
                    vocab_size=tokenizer.vocab_size,
                    block_size=train_args.block_size,
                )
                if accum_count == 0:
                    mult = lr_multiplier(train_args, global_step, schedule_steps)
                    set_optimizer_lr(muon, base_muon_lrs, mult)
                    set_optimizer_lr(adamw, base_adamw_lrs, mult)
                    mom_now = warmup_momentum(
                        global_step,
                        warmup=muon_mom_warmup,
                        start=train_args.muon_momentum_start,
                        end=train_args.muon_momentum,
                    )
                    set_optimizer_momentum(muon, mom_now)
                    step_train_losses = []

                with torch.autocast(
                    device_type=device.type,
                    dtype=trial_amp_dtype or torch.float32,
                    enabled=trial_amp_dtype is not None,
                ):
                    outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        sequence_start_mask=batch["sequence_start_mask"],
                        use_cache=False,
                    )
                    lm_loss, _ = compute_lm_loss(
                        outputs.logits.float(),
                        batch["labels"],
                        loss_mask=batch["loss_mask"],
                        ignore_index=-100,
                    )
                    total_loss = lm_loss
                    if train_args.z_loss_weight:
                        z_loss = compute_z_loss(
                            outputs.logits,
                            batch["labels"],
                            loss_mask=batch["loss_mask"],
                            ignore_index=-100,
                        )
                        total_loss = total_loss + train_args.z_loss_weight * z_loss

                total_loss_value = float(total_loss.detach().float().item())
                checked_loss(total_loss_value, "training")

                (total_loss / train_args.gradient_accumulation_steps).backward()
                accum_count += 1
                step_train_losses.append(float(lm_loss.detach().float().item()))
                batch_tokens = int(batch["input_ids"].numel())
                tokens_seen += batch_tokens
                last_log_tokens += batch_tokens

                current_qk = model.max_qk_logit() if train_args.log_max_qk_logit else None
                if current_qk is not None:
                    max_qk_logit = current_qk if max_qk_logit is None else max(max_qk_logit, current_qk)

                if accum_count < train_args.gradient_accumulation_steps:
                    continue

                if train_args.grad_clip > 0:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_args.grad_clip,
                    )
                    grad_norm = float(grad_norm_t.detach().float().item())
                    if not math.isfinite(grad_norm):
                        raise TrialFailed("non-finite gradient norm")
                    grad_norms.append(grad_norm)
                    if math.isfinite(grad_norm) and grad_norm > train_args.grad_clip:
                        clip_count += 1
                muon.step()
                adamw.step()
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)
                accum_count = 0

                global_step += 1
                stage_step += 1
                train_loss_value = sum(step_train_losses) / max(1, len(step_train_losses))
                train_losses.append(train_loss_value)

                should_log = train_args.log_every and global_step % train_args.log_every == 0
                if should_log:
                    now = time.time()
                    local_tps = last_log_tokens / max(now - last_log_t, 1e-6)
                    print(
                        f"trial={trial_number} seed={seed} step={global_step} "
                        f"stage={stage_path.stem} lm={train_loss_value:.4f} "
                        f"lr_mult={mult:.3f} tok/s={local_tps:.0f}",
                        flush=True,
                    )
                    last_log_t = now
                    last_log_tokens = 0

                should_eval = global_step % train_args.eval_every == 0
                if should_eval or global_step == search_args.trial_steps:
                    val_loss, val_tokens, val_batches = evaluate(
                        model,
                        tokenizer,
                        train_args,
                        val_path=val_path,
                        device=device,
                        amp_dtype=trial_amp_dtype,
                    )
                    val_loss = checked_loss(val_loss, "validation")
                    elapsed = time.time() - start_time
                    tok_per_sec = tokens_seen / max(elapsed, 1e-6)
                    current_stability = stability_penalty(
                        eval_losses=[*eval_losses, val_loss],
                        train_losses=train_losses,
                        grad_norms=grad_norms,
                        clip_count=clip_count,
                        steps=global_step,
                        max_qk_logit=max_qk_logit,
                        train_args=train_args,
                        search_args=search_args,
                    )
                    eval_losses.append(val_loss)
                    eval_records.append(
                        {
                            "step": global_step,
                            "val_loss": val_loss,
                            "val_tokens": val_tokens,
                            "val_batches": val_batches,
                            "tok_per_sec": tok_per_sec,
                            "stability_penalty": current_stability,
                        }
                    )
                    print(
                        f"trial={trial_number} seed={seed} eval_step={global_step} "
                        f"val_lm={val_loss:.4f} tok/s={tok_per_sec:.0f} "
                        f"stability={current_stability:.4f}",
                        flush=True,
                    )
                    maybe_report_pruning(
                        trial,
                        args=search_args,
                        step=global_step,
                        val_loss=val_loss,
                        tok_per_sec=tok_per_sec,
                        stability=current_stability,
                    )

        if not eval_losses:
            val_loss, val_tokens, val_batches = evaluate(
                model,
                tokenizer,
                train_args,
                val_path=val_path,
                device=device,
                amp_dtype=trial_amp_dtype,
            )
            val_loss = checked_loss(val_loss, "validation")
            eval_losses.append(val_loss)
            eval_records.append(
                {
                    "step": global_step,
                    "val_loss": val_loss,
                    "val_tokens": val_tokens,
                    "val_batches": val_batches,
                    "tok_per_sec": tokens_seen / max(time.time() - start_time, 1e-6),
                    "stability_penalty": None,
                }
            )

        elapsed_sec = time.time() - start_time
        tok_per_sec = tokens_seen / max(elapsed_sec, 1e-6)
        final_stability = stability_penalty(
            eval_losses=eval_losses,
            train_losses=train_losses,
            grad_norms=grad_norms,
            clip_count=clip_count,
            steps=global_step,
            max_qk_logit=max_qk_logit,
            train_args=train_args,
            search_args=search_args,
        )
        best_val_loss = min(eval_losses)
        final_val_loss = eval_losses[-1]
        score = objective_score(
            val_loss=best_val_loss,
            tok_per_sec=tok_per_sec,
            stability=final_stability,
            args=search_args,
        )
        return {
            "seed": seed,
            "steps": global_step,
            "tokens": tokens_seen,
            "elapsed_sec": elapsed_sec,
            "tok_per_sec": tok_per_sec,
            "initial_val_loss": initial_val_loss,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "score": score,
            "stability_penalty": final_stability,
            "clip_ratio": clip_count / max(1, global_step),
            "avg_grad_norm": mean(grad_norms) if grad_norms else None,
            "max_grad_norm": max(grad_norms) if grad_norms else None,
            "max_qk_logit": max_qk_logit,
            "train_loss_mean": mean(train_losses) if train_losses else None,
            "train_loss_std": pstdev(train_losses) if len(train_losses) >= 2 else 0.0,
            "evals": eval_records,
            "failure": None,
        }
    finally:
        del model
        del muon
        del adamw
        clear_device_cache()


def failed_seed_result(seed: int, reason: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "steps": 0,
        "tokens": 0,
        "elapsed_sec": 0.0,
        "tok_per_sec": 0.0,
        "initial_val_loss": None,
        "best_val_loss": 1e9,
        "final_val_loss": 1e9,
        "score": 1e9,
        "stability_penalty": 1e9,
        "clip_ratio": None,
        "avg_grad_norm": None,
        "max_grad_norm": None,
        "max_qk_logit": None,
        "train_loss_mean": None,
        "train_loss_std": None,
        "evals": [],
        "failure": reason,
    }


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def aggregate_seed_results(
    *,
    trial_number: int,
    params: dict[str, Any],
    seed_results: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    best_losses = [float(r["best_val_loss"]) for r in seed_results]
    final_losses = [float(r["final_val_loss"]) for r in seed_results]
    speeds = [float(r["tok_per_sec"]) for r in seed_results]
    stabilities = [float(r["stability_penalty"]) for r in seed_results]
    failures = [r["failure"] for r in seed_results if r.get("failure")]

    best_val_loss_mean = mean(best_losses)
    seed_loss_std = pstdev(best_losses) if len(best_losses) >= 2 else 0.0
    tok_per_sec_mean = mean(speeds)
    stability_mean = mean(stabilities) + args.seed_std_weight * seed_loss_std
    score = objective_score(
        val_loss=best_val_loss_mean,
        tok_per_sec=tok_per_sec_mean,
        stability=stability_mean,
        args=args,
    )
    return {
        "trial": trial_number,
        "params": params,
        "score": score,
        "best_val_loss_mean": best_val_loss_mean,
        "best_val_loss_std": seed_loss_std,
        "final_val_loss_mean": mean(final_losses),
        "tok_per_sec_mean": tok_per_sec_mean,
        "stability_penalty_mean": stability_mean,
        "failures": failures,
        "seeds": seed_results,
    }


def run_trial(
    *,
    trial: Any,
    args: argparse.Namespace,
    tokenizer: ShoujenTokenizer,
    train_paths: list[Path],
    val_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    results_path: Path,
) -> dict[str, Any]:
    params = suggest_hparams(trial, args)
    seed_results: list[dict[str, Any]] = []
    print(f"trial={trial.number} params={params}", flush=True)

    for seed_idx in range(args.seeds_per_trial):
        seed = args.seed + trial.number * 1009 + seed_idx
        try:
            result = run_one_seed(
                search_args=args,
                params=params,
                seed=seed,
                tokenizer=tokenizer,
                train_paths=train_paths,
                val_path=val_path,
                device=device,
                amp_dtype=amp_dtype,
                trial_number=trial.number,
                trial=trial,
            )
        except TrialFailed as exc:
            print(f"trial={trial.number} seed={seed} failed: {exc.reason}", flush=True)
            result = failed_seed_result(seed, exc.reason)
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            print(f"trial={trial.number} seed={seed} failed: out of memory", flush=True)
            result = failed_seed_result(seed, "out of memory")
            clear_device_cache()
        seed_results.append(result)
        if result["failure"]:
            break

    aggregate = aggregate_seed_results(
        trial_number=trial.number,
        params=params,
        seed_results=seed_results,
        args=args,
    )
    trial.set_user_attr("aggregate", json_safe(aggregate))
    write_jsonl(results_path, aggregate)
    print(
        f"trial={trial.number} score={aggregate['score']:.4f} "
        f"best_val={aggregate['best_val_loss_mean']:.4f} "
        f"tok/s={aggregate['tok_per_sec_mean']:.0f} "
        f"stability={aggregate['stability_penalty_mean']:.4f}",
        flush=True,
    )
    return aggregate


def fmt_float(value: float) -> str:
    return f"{value:.8g}"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_formal_command(args: argparse.Namespace, params: dict[str, Any]) -> str:
    schedule_steps = max(1, args.formal_lr_schedule_steps)
    warmup = max(0, int(round(schedule_steps * float(params["warmup_ratio"]))))
    lr_stable_steps = max(0, int(round(schedule_steps * float(params["lr_stable_ratio"]))))

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train_staged_packed.py",
        "--vocab",
        str(args.vocab),
        "--data-dir",
        str(args.data_dir),
        "--train-files",
        *[str(item) for item in args.train_files],
        "--val-file",
        str(args.val_file),
        "--text-column",
        str(args.text_column),
        "--output",
        str(args.formal_output),
        "--batch-size",
        str(int(params["batch_size"])),
        "--gradient-accumulation-steps",
        str(int(params["gradient_accumulation_steps"])),
        "--block-size",
        str(int(params["block_size"])),
        "--lr-schedule-steps",
        str(schedule_steps),
        "--warmup",
        str(warmup),
        "--save-every",
        str(args.formal_save_every),
        "--eval-every",
        str(args.formal_eval_every),
        "--eval-batches",
        str(args.formal_eval_batches),
        "--log-every",
        str(args.formal_log_every),
        "--grad-clip",
        fmt_float(float(params["grad_clip"])),
        "--muon-lr",
        fmt_float(float(params["muon_lr"])),
        "--muon-momentum",
        fmt_float(float(args.muon_momentum_target)),
        "--muon-momentum-start",
        fmt_float(float(params["muon_momentum_start"])),
        "--muon-ns-steps",
        str(int(params["muon_ns_steps"])),
        "--muon-wd",
        fmt_float(float(params["muon_wd"])),
        "--muon-adaptive-beta2",
        fmt_float(float(params["muon_adaptive_beta2"])),
        "--adamw-lr",
        fmt_float(float(params["adamw_lr"])),
        "--adamw-beta1",
        fmt_float(float(args.adamw_beta1)),
        "--adamw-beta2",
        fmt_float(float(args.adamw_beta2)),
        "--adamw-wd",
        fmt_float(float(params["adamw_wd"])),
        "--adamw-embed-wd",
        fmt_float(float(params["adamw_embed_wd"])),
        "--attention-window",
        str(int(args.attention_window)),
        "--lr-schedule",
        str(params["lr_schedule"]),
        "--lr-min-ratio",
        fmt_float(float(params["lr_min_ratio"])),
        "--lr-stable-steps",
        str(lr_stable_steps),
        "--z-loss-weight",
        fmt_float(float(params["z_loss_weight"])),
        "--shuffle-buffer-size",
        str(args.shuffle_buffer_size),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.init_ckpt:
        cmd.extend(["--init-ckpt", str(args.init_ckpt)])
    if args.formal_max_steps:
        cmd.extend(["--max-steps", str(args.formal_max_steps)])
    if args.formal_max_steps_per_stage:
        cmd.extend(["--max-steps-per-stage", str(args.formal_max_steps_per_stage)])
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.no_amp or params.get("amp") == "none":
        cmd.append("--no-amp")
    if not params["muon_adaptive"]:
        cmd.append("--no-muon-adaptive")
    if params["adamw_foreach"]:
        cmd.append("--adamw-foreach")
    if not params.get("adamw_independent_wd", True):
        cmd.append("--no-adamw-independent-wd")
    if params["qk_norm"]:
        cmd.append("--qk-norm")
    if args.track_qk:
        cmd.append("--log-max-qk-logit")
    if args.formal_wandb:
        cmd.extend(["--wandb", "--wandb-project", str(args.formal_wandb_project)])
    return shell_join(cmd)


def write_summary(
    *,
    args: argparse.Namespace,
    best: dict[str, Any],
    all_results: list[dict[str, Any]],
    pareto: list[dict[str, Any]] | None = None,
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    command = build_formal_command(args, best["params"])
    summary = {
        "backend": args.backend,
        "multi_objective": args.multi_objective,
        "objective": {
            "single_objective_score": (
                "best_val_loss - speed_weight*log(tok_per_sec) "
                "+ stability_weight*stability_penalty"
            ),
            "speed_weight": args.speed_weight,
            "stability_weight": args.stability_weight,
            "seed_std_weight": args.seed_std_weight,
        },
        "best": best,
        "pareto": pareto,
        "best_formal_command": command,
        "all_results": all_results,
        "docs": {
            "huggingface_hpo": "https://huggingface.co/docs/transformers/main/en/hpo_train"
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "best_train_command.txt").write_text(command + "\n", encoding="utf-8")
    print(f"best score={best['score']:.4f}", flush=True)
    print(f"best params={best['params']}", flush=True)
    print(f"wrote {args.output / 'summary.json'}", flush=True)
    print(f"best formal command:\n{command}", flush=True)


def successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if not result.get("failures") and float(result.get("best_val_loss_mean", 1e9)) < 1e8
    ]


def run_random_search(
    *,
    args: argparse.Namespace,
    tokenizer: ShoujenTokenizer,
    train_paths: list[Path],
    val_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    results_path = args.output / "trials.jsonl"
    all_results: list[dict[str, Any]] = []
    for number in range(args.n_trials):
        trial = RandomSearchTrial(number, rng)
        result = run_trial(
            trial=trial,
            args=args,
            tokenizer=tokenizer,
            train_paths=train_paths,
            val_path=val_path,
            device=device,
            amp_dtype=amp_dtype,
            results_path=results_path,
        )
        all_results.append(result)
    return all_results


def run_optuna_search(
    *,
    args: argparse.Namespace,
    tokenizer: ShoujenTokenizer,
    train_paths: list[Path],
    val_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    optuna = load_optuna()
    results_path = args.output / "trials.jsonl"
    all_results: list[dict[str, Any]] = []

    if args.multi_objective:
        sampler = optuna.samplers.NSGAIISampler(seed=args.seed)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=bool(args.storage),
            directions=["minimize", "maximize", "minimize"],
            sampler=sampler,
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, args.n_trials // 5))
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=bool(args.storage),
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

    def objective(trial: Any):
        result = run_trial(
            trial=trial,
            args=args,
            tokenizer=tokenizer,
            train_paths=train_paths,
            val_path=val_path,
            device=device,
            amp_dtype=amp_dtype,
            results_path=results_path,
        )
        all_results.append(result)
        if args.multi_objective:
            return (
                result["best_val_loss_mean"],
                result["tok_per_sec_mean"],
                result["stability_penalty_mean"],
            )
        return result["score"]

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    if args.multi_objective:
        pareto = [
            trial.user_attrs["aggregate"]
            for trial in study.best_trials
            if "aggregate" in trial.user_attrs
        ]
    else:
        pareto = None
    if not all_results:
        all_results = [
            trial.user_attrs["aggregate"]
            for trial in study.trials
            if "aggregate" in trial.user_attrs
        ]
    return all_results, pareto


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    train_paths = [resolve_data_path(args.data_dir, name) for name in args.train_files]
    val_path = resolve_data_path(args.data_dir, args.val_file)
    missing = [path for path in [*train_paths, val_path] if not path.exists()]
    if missing:
        raise SystemExit("Missing parquet file(s): " + ", ".join(str(path) for path in missing))

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = None if args.no_amp else autocast_dtype(device)
    tokenizer = ShoujenTokenizer.load(args.vocab)

    print(
        f"device={device} amp={amp_dtype} backend={args.backend} "
        f"trials={args.n_trials} trial_steps={args.trial_steps} "
        f"stages={' -> '.join(path.name for path in train_paths)}",
        flush=True,
    )
    if args.balanced_stages:
        print(
            f"balanced stage cap={stage_step_cap(args, len(train_paths))} steps/stage",
            flush=True,
        )

    if args.backend == "optuna":
        all_results, pareto = run_optuna_search(
            args=args,
            tokenizer=tokenizer,
            train_paths=train_paths,
            val_path=val_path,
            device=device,
            amp_dtype=amp_dtype,
        )
    else:
        all_results = run_random_search(
            args=args,
            tokenizer=tokenizer,
            train_paths=train_paths,
            val_path=val_path,
            device=device,
            amp_dtype=amp_dtype,
        )
        pareto = None

    if not all_results:
        raise SystemExit("No completed trials.")

    candidates = successful_results(pareto) if pareto else successful_results(all_results)
    if not candidates:
        raise SystemExit("No successful trials. Check trials.jsonl for failure reasons.")

    best = min(candidates, key=lambda item: item["score"])
    write_summary(args=args, best=best, all_results=all_results, pareto=pareto)


if __name__ == "__main__":
    main()
