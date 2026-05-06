"""Build the s0 pretraining corpus parquet.

Walks ``data/train/s0/`` (flat), extracts text from every file, splits the
running buffer into 2000-character chunks (each stripped of leading/trailing
whitespace), and writes a parquet with columns ``text`` and ``source`` to
``data/processed/s0.parquet``.

The directory contains:
- 1,866 ``.txt`` files of classical Chinese literature, philosophy, poetry.
- 1 ``.parquet`` file (``coct-en-zh-tw-translations-twp-300k.parquet``) with
  ``en``/``ch`` parallel translation data; rotating templates render each row.

Usage:
    uv run python scripts/build_corpus_s0.py [--limit N] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SRC_DIR = Path("data/train/s0")
DEFAULT_OUT = Path("data/processed/s0.parquet")

CHUNK_SIZE = 2000
PARQUET_BATCH_SIZE = 2048
BATCH_FLUSH = 10000

TRANSLATION_TEMPLATES = [
    "中文：{zh}\n英文：{en}",
    "{zh}\n翻译为英文是：\n{en}",
    "英文：{en}\n\n中文：{zh}",
    "{en}\n该英文翻译为中文是：\n{zh}",
    "请把下面这句中文翻译成英文：\n{zh}\n答：{en}",
]


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


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030")


def process_txt(path: Path, sink) -> None:
    text = read_text(path)
    source = path.name
    buffer = text + "\n\n"
    buffer = emit_from_buffer(buffer, source, False, sink)
    emit_from_buffer(buffer, source, True, sink)


def process_parquet(path: Path, sink, limit: int | None) -> None:
    source = path.name
    pf = pq.ParquetFile(path)
    buffer = ""
    idx = 0
    for batch in pf.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=["en", "ch"]):
        ens = batch.column("en").to_pylist()
        chs = batch.column("ch").to_pylist()
        for en_value, ch_value in zip(ens, chs):
            if limit is not None and idx >= limit:
                emit_from_buffer(buffer, source, True, sink)
                return
            if en_value is None or ch_value is None:
                continue
            tmpl = TRANSLATION_TEMPLATES[idx % len(TRANSLATION_TEMPLATES)]
            record_text = tmpl.format(zh=ch_value, en=en_value)
            buffer += record_text + "\n\n"
            buffer = emit_from_buffer(buffer, source, False, sink)
            idx += 1
    emit_from_buffer(buffer, source, True, sink)


def build(out_path: Path, limit: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in SRC_DIR.iterdir() if p.is_file() and p.name != ".DS_Store"
    )

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

    try:
        for path in tqdm(files, desc="s0", unit="file"):
            try:
                if path.suffix == ".parquet":
                    process_parquet(path, sink, limit)
                elif path.suffix == ".txt":
                    process_txt(path, sink)
                else:
                    print(f"skipping unknown file type: {path}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: failed to parse {path}: {exc}", file=sys.stderr)
        flush_batch()
    finally:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap records per source file")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output parquet path")
    args = ap.parse_args()
    build(args.out, args.limit)


if __name__ == "__main__":
    main()
