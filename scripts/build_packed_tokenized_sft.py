"""Build offline tokenized and packed SFT parquet shards.

The input is JSONL ChatML:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

The output keeps full `block_size + 1` token windows plus packed-conversation
ids, per-token target loss weights, and assistant-answer spans for Semantic
Tube Prediction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.data import (  # noqa: E402
    SFT_ROLE_ASSISTANT,
    SFT_ROLE_CONTROL,
    SFT_ROLE_SYSTEM,
    SFT_ROLE_THINK,
    SFT_ROLE_THINK_BOUNDARY,
    SFT_ROLE_USER,
)
from shoujen.tokenizer import (  # noqa: E402
    DEFAULT_TOKENIZER_ID,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    ShoujenTokenizer,
)

DEFAULT_INPUT = Path("tools/openai_chatml_collector/datasets/sft.jsonl")
DEFAULT_OUT_DIR = Path("data/sft-packed")


@dataclass(frozen=True)
class EncodedSFTRecord:
    token_ids: list[int]
    target_loss_weights: list[float]
    target_role_ids: list[int]
    stp_spans: list[tuple[int, int]]


@dataclass(frozen=True)
class PackedSFTSample:
    token_ids: list[int]
    seq_ids: list[int]
    position_ids: list[int]
    sequence_starts: list[bool]
    target_loss_weights: list[float]
    target_role_ids: list[int]
    stp_span_starts: list[int]
    stp_span_ends: list[int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--train-name", default="train.parquet")
    ap.add_argument("--val-name", default="val.parquet")
    ap.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="Hugging Face tokenizer id/URL or saved tokenizer directory",
    )
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--think-loss-weight", type=float, default=0.1)
    ap.add_argument("--think-boundary-loss-weight", type=float, default=1.0)
    ap.add_argument("--write-batch-size", type=int, default=512)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text", item.get("content", ""))
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return "" if content is None else str(content)


def _append_tokens(
    token_ids: list[int],
    target_loss_weights: list[float],
    target_role_ids: list[int],
    ids: Sequence[int],
    *,
    weight: float,
    role_id: int,
) -> tuple[int, int]:
    start = len(token_ids)
    token_ids.extend(int(tid) for tid in ids)
    target_loss_weights.extend(float(weight) for _ in ids)
    target_role_ids.extend(int(role_id) for _ in ids)
    return start, len(token_ids)


def _append_assistant_content(
    token_ids: list[int],
    target_loss_weights: list[float],
    target_role_ids: list[int],
    tokenizer: ShoujenTokenizer,
    content: str,
    *,
    think_loss_weight: float,
    think_boundary_loss_weight: float,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    idx = 0
    in_think = False

    while idx < len(content):
        if content.startswith(THINK_START_TOKEN, idx):
            _append_tokens(
                token_ids,
                target_loss_weights,
                target_role_ids,
                tokenizer.encode(THINK_START_TOKEN),
                weight=think_boundary_loss_weight,
                role_id=SFT_ROLE_THINK_BOUNDARY,
            )
            in_think = True
            idx += len(THINK_START_TOKEN)
            continue
        if content.startswith(THINK_END_TOKEN, idx):
            _append_tokens(
                token_ids,
                target_loss_weights,
                target_role_ids,
                tokenizer.encode(THINK_END_TOKEN),
                weight=think_boundary_loss_weight,
                role_id=SFT_ROLE_THINK_BOUNDARY,
            )
            in_think = False
            idx += len(THINK_END_TOKEN)
            continue

        next_start = content.find(THINK_START_TOKEN, idx)
        next_end = content.find(THINK_END_TOKEN, idx)
        candidates = [pos for pos in (next_start, next_end) if pos != -1]
        next_idx = min(candidates) if candidates else len(content)
        segment = content[idx:next_idx]
        if segment:
            role_id = SFT_ROLE_THINK if in_think else SFT_ROLE_ASSISTANT
            weight = think_loss_weight if in_think else 1.0
            start, end = _append_tokens(
                token_ids,
                target_loss_weights,
                target_role_ids,
                tokenizer.encode(segment),
                weight=weight,
                role_id=role_id,
            )
            if not in_think and end - start >= 3:
                spans.append((start, end))
        idx = next_idx

    return spans


def encode_sft_messages(
    messages: Sequence[dict],
    tokenizer: ShoujenTokenizer,
    *,
    think_loss_weight: float = 0.1,
    think_boundary_loss_weight: float = 1.0,
) -> EncodedSFTRecord:
    token_ids: list[int] = []
    target_loss_weights: list[float] = []
    target_role_ids: list[int] = []
    stp_spans: list[tuple[int, int]] = []

    role_ids = {
        "system": SFT_ROLE_SYSTEM,
        "user": SFT_ROLE_USER,
        "assistant": SFT_ROLE_ASSISTANT,
    }

    for message in messages:
        role = str(message["role"])
        content = _content_text(message.get("content"))

        _append_tokens(
            token_ids,
            target_loss_weights,
            target_role_ids,
            [tokenizer.im_start_id],
            weight=0.0,
            role_id=SFT_ROLE_CONTROL,
        )
        _append_tokens(
            token_ids,
            target_loss_weights,
            target_role_ids,
            tokenizer.encode(role + "\n"),
            weight=0.0,
            role_id=SFT_ROLE_CONTROL,
        )

        if role == "assistant":
            stp_spans.extend(
                _append_assistant_content(
                    token_ids,
                    target_loss_weights,
                    target_role_ids,
                    tokenizer,
                    content,
                    think_loss_weight=think_loss_weight,
                    think_boundary_loss_weight=think_boundary_loss_weight,
                )
            )
        else:
            _append_tokens(
                token_ids,
                target_loss_weights,
                target_role_ids,
                tokenizer.encode(content),
                weight=0.0,
                role_id=role_ids.get(role, SFT_ROLE_CONTROL),
            )

        _append_tokens(
            token_ids,
            target_loss_weights,
            target_role_ids,
            [tokenizer.im_end_id],
            weight=1.0 if role == "assistant" else 0.0,
            role_id=SFT_ROLE_CONTROL,
        )
        _append_tokens(
            token_ids,
            target_loss_weights,
            target_role_ids,
            tokenizer.encode("\n"),
            weight=0.0,
            role_id=SFT_ROLE_CONTROL,
        )

    _append_tokens(
        token_ids,
        target_loss_weights,
        target_role_ids,
        [tokenizer.eos_id],
        weight=1.0,
        role_id=SFT_ROLE_CONTROL,
    )
    return EncodedSFTRecord(token_ids, target_loss_weights, target_role_ids, stp_spans)


def load_jsonl_records(path: Path, *, max_records: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if max_records is not None and len(records) >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"{path}:{line_no} missing messages")
            records.append(obj)
    return records


def pack_encoded_records(
    records: Sequence[EncodedSFTRecord],
    *,
    block_size: int,
    pad_id: int,
) -> list[PackedSFTSample]:
    token_ids: list[int] = []
    seq_ids: list[int] = []
    position_ids: list[int] = []
    sequence_starts: list[bool] = []
    target_loss_weights: list[float] = []
    target_role_ids: list[int] = []
    global_spans: list[tuple[int, int, int]] = []

    for seq_id, record in enumerate(records):
        if len(record.token_ids) < 2:
            continue
        base = len(token_ids)
        token_ids.extend(record.token_ids)
        seq_ids.extend(seq_id for _ in record.token_ids)
        position_ids.extend(range(len(record.token_ids)))
        sequence_starts.extend(pos == 0 for pos in range(len(record.token_ids)))
        target_loss_weights.extend(record.target_loss_weights)
        target_role_ids.extend(record.target_role_ids)
        for start, end in record.stp_spans:
            global_spans.append((seq_id, base + start, base + end))

    samples: list[PackedSFTSample] = []
    if len(token_ids) < 2:
        return samples

    sample_len = block_size + 1
    for start in range(0, len(token_ids) - 1, block_size):
        end = min(len(token_ids), start + sample_len)
        valid_len = end - start
        pad_len = sample_len - valid_len

        sample_token_ids = token_ids[start:end] + [pad_id] * pad_len
        sample_seq_ids = seq_ids[start:end] + [-1] * pad_len
        sample_position_ids = position_ids[start:end] + [0] * pad_len
        sample_sequence_starts = sequence_starts[start:end] + [False] * pad_len
        sample_loss_weights = target_loss_weights[start:end] + [0.0] * pad_len
        sample_role_ids = target_role_ids[start:end] + [SFT_ROLE_CONTROL] * pad_len

        input_start = start
        input_end = min(start + block_size, start + valid_len - 1)
        span_starts: list[int] = []
        span_ends: list[int] = []
        for _seq_id, span_start, span_end in global_spans:
            clipped_start = max(span_start, input_start)
            clipped_end = min(span_end, input_end)
            if clipped_end - clipped_start >= 3:
                span_starts.append(clipped_start - start)
                span_ends.append(clipped_end - start)

        samples.append(
            PackedSFTSample(
                token_ids=sample_token_ids,
                seq_ids=sample_seq_ids,
                position_ids=sample_position_ids,
                sequence_starts=sample_sequence_starts,
                target_loss_weights=sample_loss_weights,
                target_role_ids=sample_role_ids,
                stp_span_starts=span_starts,
                stp_span_ends=span_ends,
            )
        )
    return samples


def _fixed_list(value_type: pa.DataType, block_size: int) -> pa.DataType:
    return pa.list_(value_type, list_size=block_size + 1)


def schema_for(
    *,
    block_size: int,
    tokenizer: ShoujenTokenizer,
    tokenizer_source: str,
    source_path: Path,
    think_loss_weight: float,
    think_boundary_loss_weight: float,
) -> pa.Schema:
    token_type = pa.uint16() if tokenizer.vocab_size <= 65535 else pa.uint32()
    metadata = {
        b"shoujen_format": b"packed-sft-v1",
        b"block_size": str(block_size).encode("ascii"),
        b"tokenizer": str(tokenizer_source).encode("utf-8"),
        b"source_path": str(source_path).encode("utf-8"),
        b"think_loss_weight": str(think_loss_weight).encode("ascii"),
        b"think_boundary_loss_weight": str(think_boundary_loss_weight).encode("ascii"),
    }
    return pa.schema(
        [
            ("token_ids", _fixed_list(token_type, block_size)),
            ("seq_ids", _fixed_list(pa.int32(), block_size)),
            ("position_ids", _fixed_list(pa.uint32(), block_size)),
            ("sequence_starts", _fixed_list(pa.bool_(), block_size)),
            ("target_loss_weights", _fixed_list(pa.float32(), block_size)),
            ("target_role_ids", _fixed_list(pa.int8(), block_size)),
            ("stp_span_starts", pa.list_(pa.uint32())),
            ("stp_span_ends", pa.list_(pa.uint32())),
        ],
        metadata=metadata,
    )


def write_samples(
    samples: Sequence[PackedSFTSample],
    out_path: Path,
    schema: pa.Schema,
    *,
    write_batch_size: int,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = pq.ParquetWriter(str(tmp_path), schema, compression="zstd")
    try:
        for start in range(0, len(samples), write_batch_size):
            batch = samples[start : start + write_batch_size]
            table = pa.Table.from_arrays(
                [
                    pa.array([item.token_ids for item in batch], type=schema.field("token_ids").type),
                    pa.array([item.seq_ids for item in batch], type=schema.field("seq_ids").type),
                    pa.array(
                        [item.position_ids for item in batch],
                        type=schema.field("position_ids").type,
                    ),
                    pa.array(
                        [item.sequence_starts for item in batch],
                        type=schema.field("sequence_starts").type,
                    ),
                    pa.array(
                        [item.target_loss_weights for item in batch],
                        type=schema.field("target_loss_weights").type,
                    ),
                    pa.array(
                        [item.target_role_ids for item in batch],
                        type=schema.field("target_role_ids").type,
                    ),
                    pa.array(
                        [item.stp_span_starts for item in batch],
                        type=schema.field("stp_span_starts").type,
                    ),
                    pa.array(
                        [item.stp_span_ends for item in batch],
                        type=schema.field("stp_span_ends").type,
                    ),
                ],
                schema=schema,
            )
            writer.write_table(table)
    except Exception:
        writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    else:
        writer.close()
        tmp_path.replace(out_path)


def build_split(
    records: Sequence[dict],
    tokenizer: ShoujenTokenizer,
    *,
    block_size: int,
    think_loss_weight: float,
    think_boundary_loss_weight: float,
) -> list[PackedSFTSample]:
    encoded = [
        encode_sft_messages(
            record["messages"],
            tokenizer,
            think_loss_weight=think_loss_weight,
            think_boundary_loss_weight=think_boundary_loss_weight,
        )
        for record in records
    ]
    return pack_encoded_records(encoded, block_size=block_size, pad_id=tokenizer.pad_id)


def split_records(
    records: list[dict],
    *,
    val_ratio: float,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if val_ratio == 0.0:
        return records, []
    val_count = max(1, int(round(len(records) * val_ratio)))
    val_count = min(val_count, max(0, len(records) - 1))
    return records[val_count:], records[:val_count]


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.write_batch_size <= 0:
        raise SystemExit("--write-batch-size must be positive")
    if not args.input.exists():
        raise SystemExit(f"missing SFT JSONL: {args.input}")

    tokenizer = ShoujenTokenizer.load(args.tokenizer)
    records = load_jsonl_records(args.input, max_records=args.max_records)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    train_records, val_records = split_records(records, val_ratio=args.val_ratio)

    schema = schema_for(
        block_size=args.block_size,
        tokenizer=tokenizer,
        tokenizer_source=args.tokenizer,
        source_path=args.input,
        think_loss_weight=args.think_loss_weight,
        think_boundary_loss_weight=args.think_boundary_loss_weight,
    )
    train_samples = build_split(
        train_records,
        tokenizer,
        block_size=args.block_size,
        think_loss_weight=args.think_loss_weight,
        think_boundary_loss_weight=args.think_boundary_loss_weight,
    )
    train_path = args.out_dir / args.train_name
    write_samples(
        train_samples,
        train_path,
        schema,
        write_batch_size=args.write_batch_size,
        overwrite=args.overwrite,
    )
    print(
        f"train: records={len(train_records)} samples={len(train_samples)} -> {train_path}",
        flush=True,
    )

    if val_records:
        val_samples = build_split(
            val_records,
            tokenizer,
            block_size=args.block_size,
            think_loss_weight=args.think_loss_weight,
            think_boundary_loss_weight=args.think_boundary_loss_weight,
        )
        val_path = args.out_dir / args.val_name
        write_samples(
            val_samples,
            val_path,
            schema,
            write_batch_size=args.write_batch_size,
            overwrite=args.overwrite,
        )
        print(
            f"val: records={len(val_records)} samples={len(val_samples)} -> {val_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
