"""Build the s1 pretraining corpus parquet.

Walks ``data/train/s1/`` (flat), reads each ``.txt`` file as one document,
splits into 2000-character chunks (each stripped of leading/trailing
whitespace), and writes a parquet with columns ``text`` and ``source`` to
``data/processed/s1.parquet``.

Usage:
    uv run python scripts/build_corpus_s1.py [--limit N] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SRC_DIR = Path("data/train/s1")
DEFAULT_OUT = Path("data/processed/s1.parquet")

CHUNK_SIZE = 2000
BATCH_FLUSH = 10000


def emit_from_buffer(buffer: str, source: str, flush: bool, sink) -> str:
    while len(buffer) >= CHUNK_SIZE:
        s = buffer[:CHUNK_SIZE].strip()
        if s:
            sink(s, source)
        buffer = buffer[CHUNK_SIZE:]
    if flush:
        s = buffer.strip()
        if s:
            sink(s, source)
        return ""
    return buffer


def build(out_path: Path, limit: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in SRC_DIR.iterdir()
                   if p.is_file() and p.suffix == ".txt")
    if limit is not None:
        files = files[:limit]

    schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    writer = pq.ParquetWriter(out_path, schema, compression="snappy")
    batch_texts: list[str] = []
    batch_sources: list[str] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        writer.write_table(
            pa.table({"text": batch_texts, "source": batch_sources}, schema=schema)
        )
        batch_texts.clear()
        batch_sources.clear()

    def sink(t: str, s: str) -> None:
        batch_texts.append(t)
        batch_sources.append(s)
        if len(batch_texts) >= BATCH_FLUSH:
            flush_batch()

    def read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="gb18030")

    try:
        for path in tqdm(files, desc="s1", unit="file"):
            try:
                buffer = read_text(path)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: failed to parse {path}: {exc}", file=sys.stderr)
                continue
            emit_from_buffer(buffer, os.path.basename(path), True, sink)
        flush_batch()
    finally:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap files processed at N (for smoke testing)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output parquet path")
    args = ap.parse_args()
    build(args.out, args.limit)


if __name__ == "__main__":
    main()
