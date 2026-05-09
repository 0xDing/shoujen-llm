"""Evaluation helpers for Shoujen training runs.

The LM loss is still the primary optimization signal, but BPB gives a
tokenizer-independent validation metric, and CORE gives an optional external
capability check when a nanochat/DCLM-style eval bundle is available.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from shoujen.model import ShoujenLM
from shoujen.tokenizer import ShoujenTokenizer


@dataclass
class EvalMetrics:
    lm_loss: float
    active_tokens: float
    batches: int
    bpb: float = math.nan
    bytes: float = 0.0

    def __iter__(self):
        """Keep backward-compatible tuple unpacking: loss, tokens, batches."""
        yield self.lm_loss
        yield self.active_tokens
        yield self.batches

    def as_log_dict(self, prefix: str = "val") -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            f"{prefix}/lm_loss": self.lm_loss,
            f"{prefix}/active_tokens": self.active_tokens,
            f"{prefix}/batches": self.batches,
        }
        if math.isfinite(self.bpb):
            payload[f"{prefix}/bpb"] = self.bpb
            payload[f"{prefix}/bytes"] = self.bytes
        return payload


@dataclass
class CoreMetrics:
    score: float
    results: dict[str, float] = field(default_factory=dict)
    centered_results: dict[str, float] = field(default_factory=dict)

    def as_log_dict(self, prefix: str = "core") -> dict[str, float]:
        payload = {f"{prefix}/metric": self.score}
        for label, value in self.results.items():
            payload[f"{prefix}/accuracy/{_metric_key(label)}"] = value
        for label, value in self.centered_results.items():
            payload[f"{prefix}/centered/{_metric_key(label)}"] = value
        return payload


def token_byte_lengths(
    tokenizer: ShoujenTokenizer,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return byte length per token id, with special tokens counted as 0 bytes."""
    cached = getattr(tokenizer, "_shoujen_token_byte_lengths_cpu", None)
    if cached is not None and cached.numel() == tokenizer.vocab_size:
        return cached.to(device=device) if device is not None else cached

    hf_tokenizer = tokenizer.hf_tokenizer
    special_ids = {int(tid) for tid in getattr(hf_tokenizer, "all_special_ids", []) or []}
    lengths: list[int] = []
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            lengths.append(0)
            continue
        text = hf_tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        lengths.append(len(text.encode("utf-8")))

    cached = torch.tensor(lengths, dtype=torch.long)
    setattr(tokenizer, "_shoujen_token_byte_lengths_cpu", cached)
    return cached.to(device=device) if device is not None else cached


def compute_lm_eval_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    token_bytes: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean LM loss, active token count, BPB numerator nats, and bytes."""
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
    flat_labels = labels.reshape(batch_size * seq_len)
    per_token = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
        ignore_index=ignore_index,
    ).view(batch_size, seq_len)

    keep = (labels != ignore_index).to(per_token.dtype)
    if loss_mask is not None:
        keep = keep * loss_mask.to(dtype=per_token.dtype)

    active_tokens = keep.sum().clamp(min=1.0)
    total_nats = (per_token * keep).sum()

    if token_bytes is None:
        return total_nats / active_tokens, active_tokens, logits.new_zeros(()), logits.new_zeros(())

    token_bytes = token_bytes.to(device=labels.device)
    labels_safe = torch.where(labels == ignore_index, torch.zeros_like(labels), labels)
    byte_counts = token_bytes[labels_safe].to(dtype=per_token.dtype)
    byte_keep = keep * (byte_counts > 0).to(dtype=per_token.dtype)
    bpb_nats = (per_token * byte_keep).sum()
    total_bytes = (byte_counts * keep).sum()
    return total_nats / active_tokens, active_tokens, bpb_nats, total_bytes


def bpb_from_sums(total_nats: float, total_bytes: float) -> float:
    if total_bytes <= 0:
        return math.nan
    return total_nats / (math.log(2.0) * total_bytes)


def evaluate_core(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    *,
    eval_dir: str | Path,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    max_per_task: int = -1,
    task_labels: Iterable[str] | None = None,
    seed: int = 1337,
) -> CoreMetrics:
    """Evaluate a nanochat/DCLM-style CORE bundle.

    `eval_dir` should contain `core.yaml`, `eval_data/`, and optionally
    `eval_meta_data.csv`. Passing the parent directory that contains
    `eval_bundle/` is also accepted.
    """
    bundle_dir = _resolve_core_eval_dir(eval_dir)
    tasks = _load_core_tasks(bundle_dir, task_labels=task_labels)
    random_baselines = _load_random_baselines(bundle_dir / "eval_meta_data.csv")

    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    centered_results: dict[str, float] = {}
    try:
        for task in tasks:
            label = str(task["label"])
            data_path = bundle_dir / "eval_data" / str(task["dataset_uri"])
            with data_path.open("r", encoding="utf-8") as fh:
                data = [json.loads(line) for line in fh if line.strip()]
            rng = random.Random(seed)
            rng.shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]

            accuracy = _evaluate_core_task(
                model,
                tokenizer,
                data,
                device=device,
                amp_dtype=amp_dtype,
                task_type=str(task["icl_task_type"]),
                num_fewshot=int((task.get("num_fewshot") or [0])[0]),
                continuation_delimiter=str(task.get("continuation_delimiter", " ")),
            )
            results[label] = accuracy
            random_baseline = random_baselines.get(label, 0.0) * 0.01
            if random_baseline >= 1.0:
                centered = accuracy
            else:
                centered = (accuracy - random_baseline) / (1.0 - random_baseline)
            centered_results[label] = centered
    finally:
        if was_training:
            model.train()

    if not centered_results:
        return CoreMetrics(score=math.nan, results=results, centered_results=centered_results)
    return CoreMetrics(
        score=sum(centered_results.values()) / len(centered_results),
        results=results,
        centered_results=centered_results,
    )


def _resolve_core_eval_dir(path: str | Path) -> Path:
    root = Path(path)
    if (root / "core.yaml").exists():
        return root
    if (root / "eval_bundle" / "core.yaml").exists():
        return root / "eval_bundle"
    raise FileNotFoundError(
        f"CORE eval bundle not found at {root}; expected core.yaml or eval_bundle/core.yaml"
    )


def _load_core_tasks(bundle_dir: Path, *, task_labels: Iterable[str] | None) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for CORE evaluation") from exc

    with (bundle_dir / "core.yaml").open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    tasks = list(config.get("icl_tasks", []))
    if task_labels is not None:
        wanted = {label for label in task_labels}
        tasks = [task for task in tasks if str(task.get("label")) in wanted]
    return tasks


def _load_random_baselines(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get("Eval Task")
            baseline = row.get("Random baseline")
            if label and baseline:
                out[label] = float(baseline)
    return out


def _evaluate_core_task(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    data: list[dict[str, Any]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    task_type: str,
    num_fewshot: int,
    continuation_delimiter: str,
) -> float:
    if not data:
        return math.nan
    correct = 0
    for idx, item in enumerate(data):
        fewshot_examples = _fewshot_examples(data, idx, num_fewshot)
        correct += int(
            _evaluate_core_example(
                model,
                tokenizer,
                item,
                fewshot_examples,
                device=device,
                amp_dtype=amp_dtype,
                task_type=task_type,
                continuation_delimiter=continuation_delimiter,
            )
        )
    return correct / len(data)


def _fewshot_examples(data: list[dict[str, Any]], idx: int, num_fewshot: int) -> list[dict[str, Any]]:
    if num_fewshot <= 0:
        return []
    available = [i for i in range(len(data)) if i != idx]
    rng = random.Random(1234 + idx)
    picked = rng.sample(available, min(num_fewshot, len(available)))
    return [data[i] for i in picked]


def _evaluate_core_example(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    item: dict[str, Any],
    fewshot_examples: list[dict[str, Any]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    task_type: str,
    continuation_delimiter: str,
) -> bool:
    if task_type == "multiple_choice":
        prompts = _render_mc_prompts(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = _batch_sequences_common_prefix(tokenizer, prompts)
    elif task_type == "schema":
        prompts = _render_schema_prompts(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = _batch_sequences_common_suffix(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = _render_lm_prompts(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = _batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported CORE task type: {task_type}")

    max_seq_len = int(getattr(model.config, "max_seq_len", 0) or 0)
    if max_seq_len > 0:
        tokens, start_idxs, end_idxs = _truncate_for_max_len(tokens, start_idxs, end_idxs, max_seq_len)

    input_ids = _stack_token_sequences(tokens, tokenizer.pad_id).to(device)
    losses, predictions = _forward_losses_predictions(
        model,
        input_ids,
        amp_dtype=amp_dtype,
    )

    if task_type == "language_modeling":
        start, end = start_idxs[0], end_idxs[0]
        if start <= 0 or end <= start:
            return False
        predicted_tokens = predictions[0, start - 1 : end - 1]
        actual_tokens = input_ids[0, start:end]
        return bool(torch.equal(predicted_tokens, actual_tokens))

    mean_losses: list[float] = []
    for row_idx, (start, end) in enumerate(zip(start_idxs, end_idxs)):
        if start <= 0 or end <= start:
            mean_losses.append(float("inf"))
        else:
            mean_losses.append(float(losses[row_idx, start - 1 : end - 1].mean().item()))
    pred_idx = min(range(len(mean_losses)), key=mean_losses.__getitem__)
    return pred_idx == int(item["gold"])


def _forward_losses_predictions(
    model: ShoujenLM,
    input_ids: torch.Tensor,
    *,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_type = input_ids.device.type
    autocast_enabled = amp_dtype is not None and device_type != "cpu"
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
        outputs = model(input_ids, use_cache=False)
    logits = outputs.logits.float()
    batch_size, seq_len, vocab_size = logits.shape
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    losses[:, -1] = float("nan")
    return losses, logits.argmax(dim=-1)


def _render_mc_prompts(
    item: dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    prefix = _render_fewshot_mc(fewshot_examples, continuation_delimiter)
    return [
        f"{prefix}{item['query']}{continuation_delimiter}{choice}"
        for choice in item["choices"]
    ]


def _render_fewshot_mc(examples: list[dict[str, Any]], continuation_delimiter: str) -> str:
    chunks = []
    for example in examples:
        gold = int(example["gold"])
        chunks.append(f"{example['query']}{continuation_delimiter}{example['choices'][gold]}")
    return ("\n\n".join(chunks) + "\n\n") if chunks else ""


def _render_schema_prompts(
    item: dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    prefix = _render_fewshot_schema(fewshot_examples, continuation_delimiter)
    return [
        f"{prefix}{context}{continuation_delimiter}{item['continuation']}"
        for context in item["context_options"]
    ]


def _render_fewshot_schema(examples: list[dict[str, Any]], continuation_delimiter: str) -> str:
    chunks = []
    for example in examples:
        gold = int(example["gold"])
        chunks.append(
            f"{example['context_options'][gold]}{continuation_delimiter}{example['continuation']}"
        )
    return ("\n\n".join(chunks) + "\n\n") if chunks else ""


def _render_lm_prompts(
    item: dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: list[dict[str, Any]],
) -> list[str]:
    chunks = []
    for example in fewshot_examples:
        chunks.append(f"{str(example['context']).strip()}{continuation_delimiter}{example['continuation']}")
    prefix = ("\n\n".join(chunks) + "\n\n") if chunks else ""
    prompt_without = f"{prefix}{str(item['context']).strip()}{continuation_delimiter}"
    return [prompt_without, f"{prompt_without}{item['continuation']}"]


def _batch_sequences_common_prefix(
    tokenizer: ShoujenTokenizer,
    prompts: list[str],
) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = [tokenizer.encode(prompt) for prompt in prompts]
    start = _common_length(tokens, direction="left")
    return tokens, [start] * len(tokens), [len(seq) for seq in tokens]


def _batch_sequences_common_suffix(
    tokenizer: ShoujenTokenizer,
    prompts: list[str],
) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = [tokenizer.encode(prompt) for prompt in prompts]
    suffix = _common_length(tokens, direction="right")
    ends = [len(seq) for seq in tokens]
    starts = [end - suffix for end in ends]
    return tokens, starts, ends


def _batch_sequences_lm(
    tokenizer: ShoujenTokenizer,
    prompts: list[str],
) -> tuple[list[list[int]], list[int], list[int]]:
    without, with_continuation = [tokenizer.encode(prompt) for prompt in prompts]
    start = len(without)
    if without != with_continuation[:start]:
        start = _common_length([without, with_continuation], direction="left")
    return [with_continuation], [start], [len(with_continuation)]


def _common_length(token_sequences: list[list[int]], *, direction: str) -> int:
    if not token_sequences:
        return 0
    min_len = min(len(seq) for seq in token_sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for offset, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if any(seq[idx] != token for seq in token_sequences[1:]):
            return offset
    return min_len


def _truncate_for_max_len(
    tokens: list[list[int]],
    start_idxs: list[int],
    end_idxs: list[int],
    max_seq_len: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    out_tokens: list[list[int]] = []
    out_starts: list[int] = []
    out_ends: list[int] = []
    for seq, start, end in zip(tokens, start_idxs, end_idxs):
        if len(seq) <= max_seq_len:
            out_tokens.append(seq)
            out_starts.append(start)
            out_ends.append(end)
            continue
        crop = len(seq) - max_seq_len
        out_tokens.append(seq[-max_seq_len:])
        out_starts.append(start - crop)
        out_ends.append(end - crop)
    return out_tokens, out_starts, out_ends


def _stack_token_sequences(tokens: list[list[int]], pad_token_id: int) -> torch.Tensor:
    if not tokens:
        raise ValueError("cannot stack an empty token batch")
    seq_len = max(len(seq) for seq in tokens)
    input_ids = torch.full((len(tokens), seq_len), pad_token_id, dtype=torch.long)
    for row, seq in enumerate(tokens):
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return input_ids


def _metric_key(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()
