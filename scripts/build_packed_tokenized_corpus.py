"""Build offline tokenized and packed pretraining parquet shards.

This removes tokenizer and document-packing work from training workers. The
output shards keep full `block_size + 1` token windows plus document ids,
position ids, and sequence-start flags. `train_staged_packed.py` can read these
files directly with `--data-dir data/processed-packed --data-format packed`.

Usage:
    uv run python scripts/build_packed_tokenized_corpus.py \
      --source-dir data/processed-clean \
      --out-dir data/processed-packed \
      --block-size 2048
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer  # noqa: E402

DEFAULT_FILES = [
    "s-init.parquet",
    "s0-train.parquet",
    "s0-val.parquet",
    "s1.parquet",
    "s2.parquet",
]

_WORKER_TOKENIZER: ShoujenTokenizer | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--vocab",
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="Hugging Face tokenizer id/URL or legacy local vocab.json",
    )
    ap.add_argument("--source-dir", type=Path, default=Path("data/processed-clean"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed-packed"))
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--read-batch-size", type=int, default=1024)
    ap.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=1024,
        help="Documents per tokenizer call / worker task.",
    )
    ap.add_argument(
        "--tokenizer-workers",
        type=int,
        default=0,
        help="Tokenizer worker processes. 0 uses a conservative CPU-based default; 1 disables multiprocessing.",
    )
    ap.add_argument("--write-batch-size", type=int, default=512)
    ap.add_argument("--limit-docs", type=int, default=None, help="smoke-test cap per input file")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke-test cap per output file")
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def resolve_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def effective_tokenizer_workers(requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, min(os.cpu_count() or 1, 8))


def init_tokenizer_worker(tokenizer_source: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = ShoujenTokenizer.load(tokenizer_source)


def tokenize_text_batch(texts: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return _WORKER_TOKENIZER.encode_batch(texts, add_eos=True)


def tokenize_text_batch_local(tokenizer: ShoujenTokenizer, texts: list[str]) -> list[list[int]]:
    return tokenizer.encode_batch(texts, add_eos=True)


def fixed_list_type(value_type: pa.DataType, block_size: int) -> pa.DataType:
    return pa.list_(value_type, list_size=block_size + 1)


def schema_for(
    block_size: int,
    tokenizer: ShoujenTokenizer,
    tokenizer_source: str,
    source_path: Path,
) -> pa.Schema:
    token_type = pa.uint16() if tokenizer.vocab_size <= 65535 else pa.uint32()
    seq_type = pa.uint16() if block_size + 1 <= 65535 else pa.uint32()
    metadata = {
        b"shoujen_format": b"packed-tokenized-v1",
        b"block_size": str(block_size).encode("ascii"),
        b"tokenizer": str(tokenizer_source).encode("utf-8"),
        b"source_path": str(source_path).encode("utf-8"),
    }
    return pa.schema(
        [
            ("token_ids", fixed_list_type(token_type, block_size)),
            ("seq_ids", fixed_list_type(seq_type, block_size)),
            ("position_ids", fixed_list_type(pa.uint32(), block_size)),
            ("sequence_starts", fixed_list_type(pa.bool_(), block_size)),
        ],
        metadata=metadata,
    )


def normalize_seq_ids(seq_ids: list[int]) -> list[int]:
    remap: dict[int, int] = {}
    out: list[int] = []
    for seq_id in seq_ids:
        mapped = remap.get(seq_id)
        if mapped is None:
            mapped = len(remap)
            remap[seq_id] = mapped
        out.append(mapped)
    return out


class PackedWriter:
    def __init__(
        self,
        out_path: Path,
        schema: pa.Schema,
        *,
        block_size: int,
        write_batch_size: int,
        overwrite: bool,
    ) -> None:
        self.out_path = out_path
        self.tmp_path = out_path.with_name(out_path.name + ".tmp")
        self.schema = schema
        self.block_size = block_size
        self.write_batch_size = write_batch_size
        self.count = 0
        self._token_ids: list[list[int]] = []
        self._seq_ids: list[list[int]] = []
        self._position_ids: list[list[int]] = []
        self._sequence_starts: list[list[bool]] = []

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self._writer = pq.ParquetWriter(str(self.tmp_path), schema, compression="zstd")

    def add(
        self,
        token_ids: list[int],
        seq_ids: list[int],
        position_ids: list[int],
        sequence_starts: list[bool],
    ) -> None:
        expected_len = self.block_size + 1
        if not (
            len(token_ids)
            == len(seq_ids)
            == len(position_ids)
            == len(sequence_starts)
            == expected_len
        ):
            raise ValueError(f"packed samples must have length {expected_len}")
        self._token_ids.append(token_ids)
        self._seq_ids.append(normalize_seq_ids(seq_ids))
        self._position_ids.append(position_ids)
        self._sequence_starts.append(sequence_starts)
        self.count += 1
        if len(self._token_ids) >= self.write_batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._token_ids:
            return
        arrays = [
            pa.array(self._token_ids, type=self.schema.field("token_ids").type),
            pa.array(self._seq_ids, type=self.schema.field("seq_ids").type),
            pa.array(self._position_ids, type=self.schema.field("position_ids").type),
            pa.array(self._sequence_starts, type=self.schema.field("sequence_starts").type),
        ]
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))
        self._token_ids.clear()
        self._seq_ids.clear()
        self._position_ids.clear()
        self._sequence_starts.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._writer.close()
        self.tmp_path.replace(self.out_path)

    def abort(self) -> None:
        try:
            self._writer.close()
        finally:
            if self.tmp_path.exists():
                self.tmp_path.unlink()


def emit_available_samples(
    writer: PackedWriter,
    tokens: list[int],
    seq_ids: list[int],
    position_ids: list[int],
    sequence_starts: list[bool],
    *,
    block_size: int,
    max_samples: int | None,
) -> bool:
    sample_len = block_size + 1
    while len(tokens) >= sample_len:
        writer.add(
            tokens[:sample_len],
            seq_ids[:sample_len],
            position_ids[:sample_len],
            sequence_starts[:sample_len],
        )
        if max_samples is not None and writer.count >= max_samples:
            return True
        del tokens[:block_size]
        del seq_ids[:block_size]
        del position_ids[:block_size]
        del sequence_starts[:block_size]
    return False


def process_file(
    src_path: Path,
    out_path: Path,
    tokenizer: ShoujenTokenizer,
    args: argparse.Namespace,
    *,
    tokenizer_workers: int,
    executor: ProcessPoolExecutor | None,
) -> dict[str, Any]:
    pf = pq.ParquetFile(str(src_path))
    names = set(pf.schema_arrow.names)
    if args.text_column not in names:
        raise ValueError(f"{src_path} does not contain column {args.text_column!r}")

    writer = PackedWriter(
        out_path,
        schema_for(args.block_size, tokenizer, args.vocab, src_path),
        block_size=args.block_size,
        write_batch_size=args.write_batch_size,
        overwrite=args.overwrite,
    )

    tokens: list[int] = []
    seq_ids: list[int] = []
    position_ids: list[int] = []
    sequence_starts: list[bool] = []
    next_seq_id = 0
    docs_seen = 0
    docs_used = 0
    docs_skipped = 0
    stop = False
    pending: deque[Future[list[list[int]]]] = deque()
    max_pending_batches = max(1, tokenizer_workers * 2)

    def consume_tokenized_batch(encoded_batch: list[list[int]]) -> None:
        nonlocal docs_used, docs_skipped, stop, next_seq_id

        for doc_ids in encoded_batch:
            if len(doc_ids) < 2:
                docs_skipped += 1
                continue
            seq_id = next_seq_id
            next_seq_id += 1
            for pos, token_id in enumerate(doc_ids):
                tokens.append(int(token_id))
                seq_ids.append(seq_id)
                position_ids.append(pos)
                sequence_starts.append(pos == 0)
            docs_used += 1
            stop = emit_available_samples(
                writer,
                tokens,
                seq_ids,
                position_ids,
                sequence_starts,
                block_size=args.block_size,
                max_samples=args.max_samples,
            )
            if stop:
                break

    def drain_pending(*, force: bool = False) -> None:
        while pending and (force or len(pending) >= max_pending_batches):
            consume_tokenized_batch(pending.popleft().result())
            if stop:
                while pending:
                    pending.popleft().cancel()
                break

    def submit_or_consume(texts_batch: list[str], executor: ProcessPoolExecutor | None) -> None:
        if not texts_batch:
            return
        if executor is None:
            consume_tokenized_batch(tokenize_text_batch_local(tokenizer, texts_batch))
            return
        pending.append(executor.submit(tokenize_text_batch, texts_batch))
        drain_pending()

    def process_input(executor: ProcessPoolExecutor | None) -> None:
        nonlocal docs_seen, docs_skipped, stop

        texts_batch: list[str] = []
        with tqdm(total=pf.metadata.num_rows, desc=src_path.name, unit="doc") as pbar:
            for batch in pf.iter_batches(batch_size=args.read_batch_size, columns=[args.text_column]):
                for text in batch.column(args.text_column).to_pylist():
                    if args.limit_docs is not None and docs_seen >= args.limit_docs:
                        stop = True
                        break
                    pbar.update(1)
                    docs_seen += 1
                    if not text:
                        docs_skipped += 1
                        continue
                    texts_batch.append(str(text))
                    if len(texts_batch) >= args.tokenize_batch_size:
                        submit_or_consume(texts_batch, executor)
                        texts_batch = []
                        if stop:
                            break
                if stop:
                    break
            if texts_batch and not stop:
                submit_or_consume(texts_batch, executor)
            drain_pending(force=True)

    try:
        process_input(executor)
    except Exception:
        writer.abort()
        raise
    else:
        writer.close()

    return {
        "docs_seen": docs_seen,
        "docs_used": docs_used,
        "docs_skipped": docs_skipped,
        "samples": writer.count,
    }


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise SystemExit("--block-size must be positive")
    if args.read_batch_size <= 0 or args.write_batch_size <= 0 or args.tokenize_batch_size <= 0:
        raise SystemExit("--read-batch-size, --write-batch-size and --tokenize-batch-size must be positive")

    tokenizer = ShoujenTokenizer.load(args.vocab)
    tokenizer_workers = effective_tokenizer_workers(args.tokenizer_workers)
    print(
        f"tokenizer={args.vocab} vocab_size={tokenizer.vocab_size} block_size={args.block_size} "
        f"tokenizer_workers={tokenizer_workers} tokenize_batch_size={args.tokenize_batch_size}",
        flush=True,
    )
    executor_context = (
        nullcontext(None)
        if tokenizer_workers <= 1
        else ProcessPoolExecutor(
            max_workers=tokenizer_workers,
            initializer=init_tokenizer_worker,
            initargs=(args.vocab,),
        )
    )
    with executor_context as executor:
        for item in args.files:
            src_path = resolve_path(args.source_dir, item)
            out_path = resolve_path(args.out_dir, item)
            if not src_path.exists():
                raise SystemExit(f"missing input parquet: {src_path}")
            stats = process_file(
                src_path,
                out_path,
                tokenizer,
                args,
                tokenizer_workers=tokenizer_workers,
                executor=executor,
            )
            print(
                f"{src_path.name}: docs_seen={stats['docs_seen']} docs_used={stats['docs_used']} "
                f"docs_skipped={stats['docs_skipped']} samples={stats['samples']} -> {out_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
