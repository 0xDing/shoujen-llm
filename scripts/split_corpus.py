"""Split a cleaned parquet shard into train/val parquet files.

Usage: uv run python scripts/split_corpus.py data/processed-clean/s0.parquet [--val-ratio 0.01] [--seed 42]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    table = pq.read_table(args.src)
    n = table.num_rows

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, round(n * args.val_ratio))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_table = table.take(pa.array(train_idx))
    val_table = table.take(pa.array(val_idx))

    stem = args.src.stem
    out_dir = args.src.parent
    train_path = out_dir / f"{stem}-train.parquet"
    val_path = out_dir / f"{stem}-val.parquet"

    pq.write_table(train_table, train_path, compression="snappy")
    pq.write_table(val_table, val_path, compression="snappy")

    print(f"total: {n}")
    print(f"train: {train_table.num_rows} -> {train_path}")
    print(f"val:   {val_table.num_rows} -> {val_path} ({val_table.num_rows / n:.2%})")


if __name__ == "__main__":
    main()
