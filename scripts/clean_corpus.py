"""Clean processed-mid parquet shards to data/processed-clean/.

Multi-process: pool cleans batches; main owns dedup + parquet writes.

Usage: uv run python scripts/clean_corpus.py [--workers N]
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SRC_DIR = Path("data/processed-mid")
DST_DIR = Path("data/processed-clean")
FILES = ["s-init.parquet", "s0.parquet", "s1.parquet", "s2.parquet"]

MIN_LEN = 20
MAX_NONTEXT_RATIO = 0.35
IPC_BATCH = 1024
WRITE_BUFFER = 10000
TABLE_DELIM_LIKE = re.compile(r"^[\s|:\-+]+$")
EMPTY_TABLE_ROW = re.compile(r"^\|(\s*\|)+\s*$")
GUTENBERG_LINENUM = re.compile(r"^\s*\d{1,5}(?:[ \t]{2,}|\.\s+)")
DISPLAYSTYLE = re.compile(r"\{\\displaystyle\s+(.+?)\}", re.DOTALL)
MULTI_SPACE = re.compile(r"[ \t]{2,}")
PIPE_PAD = re.compile(r"\|[ \t]+")
PIPE_PAD_R = re.compile(r"[ \t]+\|")
CJK = re.compile(r"[㐀-鿿豈-﫿]")
CJK_SOFTWRAP = re.compile(
    r"([㐀-鿿豈-﫿，。；：、！？）」』])\n"
    r"([㐀-鿿豈-﫿（「『])"
)

BOOK_META_PREFIX = ("【书名】", "【作者】", "【类别】", "【状态】", "【更新】", "【本册章节】", "【简介】")
BOOK_MARKERS = {"---开始阅读---", "---结束阅读---"}
CHAPTER_TAG = re.compile(r"^\[\d+\][一二三四五六七八九十百千〇零０-９0-9]+[ 　]")

TEMPLATE_FRAGMENT = re.compile(
    r"^(句中文翻译成英文：|答：|翻译为英文是：|该英文翻译为中文是：|英文：|中文：)\s*$"
)


def normalize_unicode(s: str) -> str:
    s = s.replace("　", " ").replace("\xa0", " ").replace("﻿", "")
    s = "".join(c for c in s if c == "\n" or c == "\t" or unicodedata.category(c)[0] != "C")
    return s


def strip_displaystyle(s: str) -> str:
    return DISPLAYSTYLE.sub(r"\1", s)


def collapse_table_padding(s: str) -> str:
    s = PIPE_PAD.sub("| ", s)
    s = PIPE_PAD_R.sub(" |", s)
    s = MULTI_SPACE.sub(" ", s)
    return s


def fix_cjk_softwrap(s: str) -> str:
    return CJK_SOFTWRAP.sub(r"\1\2", s)


def line_is_noise(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if t in BOOK_MARKERS:
        return True
    if any(t.startswith(p) for p in BOOK_META_PREFIX):
        return True
    if CHAPTER_TAG.match(t):
        return True
    if EMPTY_TABLE_ROW.match(t):
        return True
    if TABLE_DELIM_LIKE.match(t) and ("|" in t or "-" in t) and len(t) > 4:
        return True
    if GUTENBERG_LINENUM.match(line) and len(line) - len(line.lstrip()) < 8:
        return True
    if TEMPLATE_FRAGMENT.match(t):
        return True
    return False


def clean_chunk(text: str, source: str) -> str | None:
    text = normalize_unicode(text)
    text = strip_displaystyle(text)
    text = fix_cjk_softwrap(text)

    out_lines = []
    for line in text.split("\n"):
        if line_is_noise(line):
            continue
        line = collapse_table_padding(line.rstrip())
        out_lines.append(line)

    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < MIN_LEN:
        return None

    angle_brackets = text.count("〈") + text.count("〉")
    if angle_brackets / max(len(text), 1) > 0.05:
        return None
    pipes = text.count("|")
    if pipes / max(len(text), 1) > 0.08:
        return None

    letters = sum(1 for c in text if c.isalnum() or CJK.match(c))
    if letters / max(len(text), 1) < (1 - MAX_NONTEXT_RATIO):
        return None

    return text


def chunk_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def clean_batch(payload: tuple[list[str], list[str | None]]) -> tuple[int, list[tuple[str, str | None, bytes]]]:
    texts, sources = payload
    out: list[tuple[str, str | None, bytes]] = []
    for t, s in zip(texts, sources):
        if not t:
            continue
        cleaned = clean_chunk(t, s or "")
        if cleaned is None:
            continue
        out.append((cleaned, s, chunk_hash(cleaned)))
    return len(texts), out


def iter_input_batches(pf: pq.ParquetFile):
    for batch in pf.iter_batches(batch_size=IPC_BATCH, columns=["text", "source"]):
        yield (batch.column("text").to_pylist(), batch.column("source").to_pylist())


def clean_file(src: Path, dst: Path, seen: set[bytes], pool: Pool) -> tuple[int, int]:
    pf = pq.ParquetFile(str(src))
    schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    writer = pq.ParquetWriter(str(dst), schema, compression="snappy")
    kept = dropped = 0
    buf_t: list[str] = []
    buf_s: list[str | None] = []
    total_rows = pf.metadata.num_rows

    try:
        with tqdm(total=total_rows, desc=src.name, unit="row", smoothing=0.05) as pbar:
            for n_in, results in pool.imap_unordered(
                clean_batch, iter_input_batches(pf), chunksize=2
            ):
                pbar.update(n_in)
                dropped += n_in - len(results)
                for cleaned, source, h in results:
                    if h in seen:
                        dropped += 1
                        continue
                    seen.add(h)
                    buf_t.append(cleaned)
                    buf_s.append(source)
                    kept += 1
                    if len(buf_t) >= WRITE_BUFFER:
                        writer.write_table(
                            pa.table({"text": buf_t, "source": buf_s}, schema=schema)
                        )
                        buf_t.clear(); buf_s.clear()
            if buf_t:
                writer.write_table(
                    pa.table({"text": buf_t, "source": buf_s}, schema=schema)
                )
    finally:
        writer.close()
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    ap.add_argument("--workers", type=int, default=default_workers)
    args = ap.parse_args()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    print(f"using {args.workers} workers")
    with Pool(args.workers) as pool:
        for name in FILES:
            src = SRC_DIR / name
            dst = DST_DIR / name
            if not src.exists():
                print(f"skip {src}", file=sys.stderr)
                continue
            kept, dropped = clean_file(src, dst, seen, pool)
            total = kept + dropped
            rate = dropped / total if total else 0
            print(f"{name}: kept={kept} dropped={dropped} drop_rate={rate:.1%}")


if __name__ == "__main__":
    main()
