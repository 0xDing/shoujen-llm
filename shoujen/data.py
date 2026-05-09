"""Data loaders for pretraining and SFT.

Pretraining input format:
    Either a single jsonl file (each line {"text": "..."}) or a directory of
    .txt files. Documents are tokenized, joined with <eos>, and packed into
    fixed-length chunks of (block_size+1) tokens.

SFT input format:
    A jsonl file where each line is {"messages": [{"role": ..., "content": ...}, ...]}.
    `tokenizer.encode_chat` produces input_ids and assistant_mask. Sequences are
    truncated/padded to block_size+1.
    Offline packed SFT parquet uses tokenized `block_size + 1` windows with
    per-target loss weights and assistant-answer spans for STP.

Each batch item exposes:
    input_ids, labels, loss_mask  (all length block_size)
where labels are shifted by 1 (predict t+1 from t), loss_mask gates the LM
loss (1 over assistant tokens for SFT, 1 everywhere for pretraining).

Semantic Tube Prediction is intentionally not emitted by packed pretraining
batches. Packed SFT batches expose message-local assistant-answer spans so STP
never samples across unrelated text or scratch-space thinking blocks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from shoujen.tokenizer import ShoujenTokenizer

PREPACKED_COLUMNS = ("token_ids", "seq_ids", "position_ids", "sequence_starts")
SFT_PACKED_COLUMNS = (
    "token_ids",
    "seq_ids",
    "position_ids",
    "sequence_starts",
    "target_loss_weights",
    "target_role_ids",
    "stp_span_starts",
    "stp_span_ends",
)

SFT_ROLE_CONTROL = 0
SFT_ROLE_SYSTEM = 1
SFT_ROLE_USER = 2
SFT_ROLE_THINK = 3
SFT_ROLE_ASSISTANT = 4
SFT_ROLE_THINK_BOUNDARY = 5


def _iter_text_documents(path: Path) -> Iterator[str]:
    if path.is_dir():
        for f in sorted(path.rglob("*.txt")):
            yield f.read_text(encoding="utf-8", errors="replace")
    else:
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content")
                    if text:
                        yield text
        elif suffix == ".txt":
            yield path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unknown corpus format: {path}")


class PackedPretrainDataset(Dataset):
    """Pre-tokenizes a corpus into a single uint32 array, then yields packed
    chunks of `block_size + 1` tokens. LM loss is computed everywhere.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        tokenizer: ShoujenTokenizer,
        block_size: int,
        cache_path: str | Path | None = None,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        cache_path = Path(cache_path) if cache_path else None
        if cache_path is not None and cache_path.exists():
            self.tokens = np.memmap(cache_path, dtype=np.uint32, mode="r")
        else:
            buf: list[int] = []
            for doc in _iter_text_documents(Path(corpus_path)):
                buf.extend(tokenizer.encode(doc, add_eos=True))
            arr = np.asarray(buf, dtype=np.uint32)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                arr.tofile(cache_path)
                self.tokens = np.memmap(cache_path, dtype=np.uint32, mode="r")
            else:
                self.tokens = arr
        if len(self.tokens) < block_size + 1:
            raise ValueError(
                f"Corpus too small: {len(self.tokens)} tokens, need at least {block_size + 1}"
            )
        self.n_chunks = (len(self.tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.block_size
        chunk = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])
        T = ids.shape[0]
        loss_mask = torch.ones(T, dtype=torch.float32)
        return {
            "input_ids": ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def _make_packed_pretrain_sample(
    token_ids: Sequence[int],
    seq_ids: Sequence[int],
    position_ids: Sequence[int],
    sequence_starts: Sequence[bool],
    block_size: int,
) -> dict:
    if not (
        len(token_ids)
        == len(seq_ids)
        == len(position_ids)
        == len(sequence_starts)
        == block_size + 1
    ):
        raise ValueError("packed pretraining samples require block_size + 1 aligned tokens")

    ids_full = torch.tensor(token_ids, dtype=torch.long)
    seq_full = torch.tensor(seq_ids, dtype=torch.long)
    pos_full = torch.tensor(position_ids, dtype=torch.long)
    starts_full = torch.tensor(sequence_starts, dtype=torch.bool)

    input_ids = ids_full[:-1]
    labels = ids_full[1:].clone()
    input_seq_ids = seq_full[:-1]
    label_seq_ids = seq_full[1:]

    same_sequence_target = (input_seq_ids == label_seq_ids) & (input_seq_ids >= 0)
    labels = torch.where(same_sequence_target, labels, torch.full_like(labels, -100))

    start_mask = starts_full[:-1].clone()
    start_mask[0] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": same_sequence_target.float(),
        "position_ids": pos_full[:-1],
        "sequence_start_mask": start_mask,
        "seq_ids": input_seq_ids,
    }


def build_packed_causal_mask(seq_ids: torch.Tensor) -> torch.Tensor:
    """Return a bool attention mask for packed sequences.

    Shape is `(B, 1, T, T)`. True entries are visible keys. Tokens may attend
    only to earlier tokens that share the same packed document id.
    """
    if seq_ids.ndim != 2:
        raise ValueError(f"seq_ids must have shape (B, T), got {tuple(seq_ids.shape)}")
    _, seq_len = seq_ids.shape
    positions = torch.arange(seq_len, device=seq_ids.device)
    causal = positions[None, :, None] >= positions[None, None, :]
    same_seq = seq_ids[:, :, None] == seq_ids[:, None, :]
    valid = seq_ids >= 0
    allowed = causal & same_seq & valid[:, :, None] & valid[:, None, :]
    return allowed[:, None, :, :]


def packed_collate(batch: list[dict]) -> dict:
    out = collate(batch)
    out["attention_mask"] = build_packed_causal_mask(out["seq_ids"])
    return out


def is_prepacked_parquet(path: str | Path) -> bool:
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(str(path)).schema_arrow.names)
    except Exception:
        return False
    return set(PREPACKED_COLUMNS).issubset(names)


def is_packed_sft_parquet(path: str | Path) -> bool:
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(str(path)).schema_arrow.names)
    except Exception:
        return False
    return set(SFT_PACKED_COLUMNS).issubset(names)


def prepacked_parquet_metadata(path: str | Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(str(path)).schema_arrow.metadata or {}
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in metadata.items()
    }


class PrepackedParquetPretrainDataset(IterableDataset):
    """Read offline tokenized and packed LM blocks from parquet.

    Expected columns are full-length `block_size + 1` arrays:
    `token_ids`, `seq_ids`, `position_ids`, and `sequence_starts`. The dataset
    derives shifted labels and the per-token loss mask at read time, matching
    `PackedParquetPretrainDataset` without per-worker tokenization.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        block_size: int,
        *,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1024,
        seed: int = 1337,
        repeat: bool = False,
        read_batch_size: int = 1024,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.parquet_path = Path(parquet_path)
        self.block_size = block_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.seed = seed
        self.repeat = repeat
        self.read_batch_size = read_batch_size
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.rank = rank
        self.world_size = world_size
        metadata = prepacked_parquet_metadata(self.parquet_path)
        stored_block_size = metadata.get("block_size")
        if stored_block_size is not None and int(stored_block_size) != block_size:
            raise ValueError(
                f"{self.parquet_path} was packed with block_size={stored_block_size}; "
                f"training requested block_size={block_size}"
            )

    def _shard_info(self) -> tuple[int, int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_count = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id
        return shard_index, shard_count, worker_id

    def _iter_records(self):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(self.parquet_path))
        shard_index, shard_count, worker_id = self._shard_info()
        record_idx = 0

        for batch in pf.iter_batches(
            batch_size=self.read_batch_size,
            row_groups=list(range(pf.metadata.num_row_groups)),
            columns=["token_ids", "seq_ids", "position_ids", "sequence_starts"],
        ):
            columns = [batch.column(name).to_pylist() for name in PREPACKED_COLUMNS]
            cols = dict(zip(PREPACKED_COLUMNS, columns))
            for token_ids, seq_ids, position_ids, sequence_starts in zip(
                cols["token_ids"],
                cols["seq_ids"],
                cols["position_ids"],
                cols["sequence_starts"],
            ):
                if record_idx % shard_count == shard_index:
                    yield token_ids, seq_ids, position_ids, sequence_starts, worker_id
                record_idx += 1

    def _sample_from_record(self, record: tuple) -> dict:
        token_ids, seq_ids, position_ids, sequence_starts, _worker_id = record
        expected_len = self.block_size + 1
        if not (
            len(token_ids)
            == len(seq_ids)
            == len(position_ids)
            == len(sequence_starts)
            == expected_len
        ):
            raise ValueError(
                f"prepacked sample length mismatch in {self.parquet_path}: "
                f"expected {expected_len}"
            )
        return _make_packed_pretrain_sample(
            token_ids,
            seq_ids,
            position_ids,
            sequence_starts,
            self.block_size,
        )

    def __iter__(self):
        epoch = 0
        while True:
            shard_index, _shard_count, _worker_id = self._shard_info()
            rng = random.Random(self.seed + epoch * 1009 + shard_index)
            if self.shuffle:
                buffer: list[tuple] = []
                for record in self._iter_records():
                    buffer.append(record)
                    if len(buffer) >= self.shuffle_buffer_size:
                        idx = rng.randrange(len(buffer))
                        yield self._sample_from_record(buffer.pop(idx))
                rng.shuffle(buffer)
                for record in buffer:
                    yield self._sample_from_record(record)
            else:
                for record in self._iter_records():
                    yield self._sample_from_record(record)

            if not self.repeat:
                break
            epoch += 1


def _make_packed_sft_sample(
    token_ids: Sequence[int],
    seq_ids: Sequence[int],
    position_ids: Sequence[int],
    sequence_starts: Sequence[bool],
    target_loss_weights: Sequence[float],
    target_role_ids: Sequence[int],
    stp_span_starts: Sequence[int],
    stp_span_ends: Sequence[int],
    block_size: int,
) -> dict:
    expected_len = block_size + 1
    if not (
        len(token_ids)
        == len(seq_ids)
        == len(position_ids)
        == len(sequence_starts)
        == len(target_loss_weights)
        == len(target_role_ids)
        == expected_len
    ):
        raise ValueError("packed SFT samples require block_size + 1 aligned token fields")
    if len(stp_span_starts) != len(stp_span_ends):
        raise ValueError("stp_span_starts and stp_span_ends must have the same length")

    ids_full = torch.tensor(token_ids, dtype=torch.long)
    seq_full = torch.tensor(seq_ids, dtype=torch.long)
    pos_full = torch.tensor(position_ids, dtype=torch.long)
    starts_full = torch.tensor(sequence_starts, dtype=torch.bool)
    weights_full = torch.tensor(target_loss_weights, dtype=torch.float32)
    roles_full = torch.tensor(target_role_ids, dtype=torch.long)

    input_seq_ids = seq_full[:-1]
    label_seq_ids = seq_full[1:]
    same_sequence_target = (input_seq_ids == label_seq_ids) & (input_seq_ids >= 0)

    labels = ids_full[1:].clone()
    labels = torch.where(same_sequence_target, labels, torch.full_like(labels, -100))
    loss_mask = weights_full[1:] * same_sequence_target.float()

    start_mask = starts_full[:-1].clone()
    start_mask[0] = True

    spans: list[tuple[int, int]] = []
    for start, end in zip(stp_span_starts, stp_span_ends):
        start = int(start)
        end = int(end)
        if end - start >= 3 and 0 <= start < end <= block_size:
            spans.append((start, end))

    return {
        "input_ids": ids_full[:-1],
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": pos_full[:-1],
        "sequence_start_mask": start_mask,
        "seq_ids": input_seq_ids,
        "target_role_ids": roles_full[1:],
        "stp_spans": (
            torch.tensor(spans, dtype=torch.long)
            if spans
            else torch.empty((0, 2), dtype=torch.long)
        ),
    }


class PrepackedParquetSFTDataset(IterableDataset):
    """Read offline tokenized and packed SFT blocks from parquet."""

    def __init__(
        self,
        parquet_path: str | Path,
        block_size: int,
        *,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1024,
        seed: int = 1337,
        repeat: bool = False,
        read_batch_size: int = 1024,
    ):
        self.parquet_path = Path(parquet_path)
        self.block_size = block_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.seed = seed
        self.repeat = repeat
        self.read_batch_size = read_batch_size
        metadata = prepacked_parquet_metadata(self.parquet_path)
        stored_block_size = metadata.get("block_size")
        if stored_block_size is not None and int(stored_block_size) != block_size:
            raise ValueError(
                f"{self.parquet_path} was packed with block_size={stored_block_size}; "
                f"training requested block_size={block_size}"
            )
        if not is_packed_sft_parquet(self.parquet_path):
            raise ValueError(f"{self.parquet_path} is not a packed SFT parquet")

    def _shard_info(self) -> tuple[int, int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        return worker_id, num_workers, worker_id

    def _iter_records(self):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(self.parquet_path))
        shard_index, shard_count, worker_id = self._shard_info()
        record_idx = 0

        for batch in pf.iter_batches(
            batch_size=self.read_batch_size,
            row_groups=list(range(pf.metadata.num_row_groups)),
            columns=list(SFT_PACKED_COLUMNS),
        ):
            cols = {name: batch.column(name).to_pylist() for name in SFT_PACKED_COLUMNS}
            for values in zip(*(cols[name] for name in SFT_PACKED_COLUMNS)):
                if record_idx % shard_count == shard_index:
                    yield (*values, worker_id)
                record_idx += 1

    def _sample_from_record(self, record: tuple) -> dict:
        (
            token_ids,
            seq_ids,
            position_ids,
            sequence_starts,
            target_loss_weights,
            target_role_ids,
            stp_span_starts,
            stp_span_ends,
            _worker_id,
        ) = record
        return _make_packed_sft_sample(
            token_ids,
            seq_ids,
            position_ids,
            sequence_starts,
            target_loss_weights,
            target_role_ids,
            stp_span_starts,
            stp_span_ends,
            self.block_size,
        )

    def __iter__(self):
        epoch = 0
        while True:
            shard_index, _shard_count, _worker_id = self._shard_info()
            rng = random.Random(self.seed + epoch * 1009 + shard_index)
            if self.shuffle:
                buffer: list[tuple] = []
                for record in self._iter_records():
                    buffer.append(record)
                    if len(buffer) >= self.shuffle_buffer_size:
                        idx = rng.randrange(len(buffer))
                        yield self._sample_from_record(buffer.pop(idx))
                rng.shuffle(buffer)
                for record in buffer:
                    yield self._sample_from_record(record)
            else:
                for record in self._iter_records():
                    yield self._sample_from_record(record)

            if not self.repeat:
                break
            epoch += 1


def packed_sft_collate(batch: list[dict]) -> dict:
    fixed_keys = [key for key in batch[0] if key != "stp_spans"]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in fixed_keys}
    out["attention_mask"] = build_packed_causal_mask(out["seq_ids"])

    max_spans = max((item["stp_spans"].shape[0] for item in batch), default=0)
    spans = torch.zeros((len(batch), max_spans, 2), dtype=torch.long)
    span_mask = torch.zeros((len(batch), max_spans), dtype=torch.bool)
    for row_idx, item in enumerate(batch):
        item_spans = item["stp_spans"]
        if item_spans.numel() == 0:
            continue
        count = item_spans.shape[0]
        spans[row_idx, :count] = item_spans
        span_mask[row_idx, :count] = True
    out["stp_spans"] = spans
    out["stp_span_mask"] = span_mask
    return out


class PackedParquetPretrainDataset(IterableDataset):
    """Stream parquet text rows and emit fixed-size packed LM blocks.

    Each parquet row is treated as an independent document. Documents are
    tokenized with an EOS, streamed in shuffled order when requested, and packed
    into `block_size + 1` token windows. The emitted tensors include document
    ids, reset masks, and per-document position ids so the model can avoid
    cross-document attention and recurrent-state leakage.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: ShoujenTokenizer,
        block_size: int,
        *,
        text_column: str = "text",
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 1337,
        repeat: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.parquet_path = Path(parquet_path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.repeat = repeat
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.rank = rank
        self.world_size = world_size

    def _shard_info(self) -> tuple[int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_count = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id
        return shard_index, shard_count

    def _stream_rows(self, epoch: int):
        from datasets import load_dataset

        stream = load_dataset(
            "parquet",
            data_files=str(self.parquet_path),
            split="train",
            streaming=True,
        )
        shard_index, shard_count = self._shard_info()
        if shard_count > 1:
            stream = stream.shard(num_shards=shard_count, index=shard_index)
        if self.shuffle:
            stream = stream.shuffle(
                buffer_size=self.shuffle_buffer_size,
                seed=self.seed + epoch + shard_index,
            )
        return stream

    def __iter__(self):
        epoch = 0
        while True:
            tokens: list[int] = []
            seq_ids: list[int] = []
            position_ids: list[int] = []
            sequence_starts: list[bool] = []
            next_seq_id = 0

            for row in self._stream_rows(epoch):
                text = row.get(self.text_column)
                if not text:
                    continue
                doc_ids = self.tokenizer.encode(str(text), add_eos=True)
                if len(doc_ids) < 2:
                    continue

                seq_id = next_seq_id
                next_seq_id += 1
                for pos, token_id in enumerate(doc_ids):
                    tokens.append(int(token_id))
                    seq_ids.append(seq_id)
                    position_ids.append(pos)
                    sequence_starts.append(pos == 0)

                    while len(tokens) >= self.block_size + 1:
                        yield _make_packed_pretrain_sample(
                            tokens[: self.block_size + 1],
                            seq_ids[: self.block_size + 1],
                            position_ids[: self.block_size + 1],
                            sequence_starts[: self.block_size + 1],
                            self.block_size,
                        )
                        del tokens[: self.block_size]
                        del seq_ids[: self.block_size]
                        del position_ids[: self.block_size]
                        del sequence_starts[: self.block_size]

            if not self.repeat:
                break
            epoch += 1


class SFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: ShoujenTokenizer,
        block_size: int,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples: list[tuple[list[int], list[int]]] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj["messages"]
                ids, mask = tokenizer.encode_chat(msgs)
                ids.append(tokenizer.eos_id)
                mask.append(1)  # learn to emit EOS
                self.examples.append((ids, mask))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ids_full, mask_full = self.examples[idx]
        T = self.block_size
        ids_arr = ids_full[: T + 1]
        mask_arr = mask_full[: T + 1]
        valid_len = len(ids_arr)
        if len(ids_arr) < T + 1:
            pad = T + 1 - len(ids_arr)
            ids_arr = ids_arr + [self.tokenizer.pad_id] * pad
            mask_arr = mask_arr + [0] * pad
        ids = torch.tensor(ids_arr[:-1], dtype=torch.long)
        labels = torch.tensor(ids_arr[1:], dtype=torch.long)
        # If pad_id falls back to eos_id, ignore by position rather than token id.
        if valid_len < T + 1:
            labels[valid_len - 1 :] = -100
        loss_mask = torch.tensor(mask_arr[1:], dtype=torch.float32)  # supervise t+1 if it is assistant
        return {
            "input_ids": ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
