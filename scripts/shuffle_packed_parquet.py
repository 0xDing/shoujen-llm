"""Deterministically shuffle offline packed-tokenized parquet shards.

This operates on rows of `data/processed-packed/*.parquet`. Each row is one
prepacked LM sample, so shuffling here breaks long source-order runs without
re-tokenizing or changing sample contents.

Usage:
    uv run --no-sync python scripts/shuffle_packed_parquet.py \
        --source-dir data/processed-packed \
        --out-dir data/processed-packed-shuffled \
        --overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from shoujen.data import PREPACKED_COLUMNS

SHUFFLE_KEY = "_shuffle_key"
UINT64_MASK = (1 << 64) - 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path("data/processed-packed"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed-packed-shuffled"))
    ap.add_argument("--files", nargs="+", default=None, help="Parquet files to shuffle; defaults to all *.parquet")
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--buckets", type=int, default=128)
    ap.add_argument("--read-batch-size", type=int, default=2048)
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def stable_u64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    z = value
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (z ^ (z >> 31)) & UINT64_MASK


def bucket_for_key(key: int, buckets: int) -> int:
    return min(buckets - 1, (key * buckets) >> 64)


def source_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        return [args.source_dir / name for name in args.files]
    return sorted(args.source_dir.glob("*.parquet"))


def require_prepacked_schema(path: Path, schema: pa.Schema) -> None:
    names = set(schema.names)
    missing = [name for name in PREPACKED_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"{path} is missing prepacked columns: {missing}")


def remove_shuffle_key(table: pa.Table, schema: pa.Schema) -> pa.Table:
    table = table.drop([SHUFFLE_KEY])
    return table.cast(schema)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        if not path.name.endswith(".shuffle_tmp"):
            raise ValueError(f"refusing to remove unexpected temp path: {path}")
        shutil.rmtree(path)


def shuffle_file(
    src_path: Path,
    out_path: Path,
    *,
    seed: int,
    buckets: int,
    read_batch_size: int,
    compression: str,
    overwrite: bool,
) -> None:
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    pf = pq.ParquetFile(src_path)
    schema = pf.schema_arrow
    require_prepacked_schema(src_path, schema)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.with_name(out_path.name + ".shuffle_tmp")
    tmp_out = out_path.with_name(out_path.name + ".tmp")
    safe_rmtree(tmp_dir)
    if tmp_out.exists():
        tmp_out.unlink()
    tmp_dir.mkdir(parents=True)

    bucket_schema = schema.append(pa.field(SHUFFLE_KEY, pa.uint64()))
    bucket_writers: dict[int, pq.ParquetWriter] = {}
    row_index = 0
    file_seed = (seed ^ stable_u64(src_path.name)) & UINT64_MASK

    try:
        with tqdm(total=pf.metadata.num_rows, desc=f"bucket {src_path.name}", unit="row") as pbar:
            for batch in pf.iter_batches(batch_size=read_batch_size, columns=list(PREPACKED_COLUMNS)):
                n = batch.num_rows
                keys = [splitmix64(file_seed + row_index + offset) for offset in range(n)]
                by_bucket: dict[int, list[int]] = {}
                for idx, key in enumerate(keys):
                    by_bucket.setdefault(bucket_for_key(key, buckets), []).append(idx)

                table = pa.Table.from_batches([batch]).append_column(
                    SHUFFLE_KEY,
                    pa.array(keys, type=pa.uint64()),
                )
                for bucket, indices in by_bucket.items():
                    writer = bucket_writers.get(bucket)
                    if writer is None:
                        bucket_path = tmp_dir / f"bucket-{bucket:05d}.parquet"
                        writer = pq.ParquetWriter(str(bucket_path), bucket_schema, compression=compression)
                        bucket_writers[bucket] = writer
                    writer.write_table(table.take(pa.array(indices, type=pa.int64())))

                row_index += n
                pbar.update(n)
        for writer in bucket_writers.values():
            writer.close()
        bucket_writers.clear()

        final_writer = pq.ParquetWriter(str(tmp_out), schema, compression=compression)
        try:
            for bucket in tqdm(range(buckets), desc=f"write {src_path.name}", unit="bucket"):
                bucket_path = tmp_dir / f"bucket-{bucket:05d}.parquet"
                if not bucket_path.exists():
                    continue
                table = pq.read_table(bucket_path)
                table = table.sort_by([(SHUFFLE_KEY, "ascending")])
                final_writer.write_table(remove_shuffle_key(table, schema))
        finally:
            final_writer.close()

        out_pf = pq.ParquetFile(tmp_out)
        if out_pf.metadata.num_rows != pf.metadata.num_rows:
            raise RuntimeError(
                f"row count mismatch for {src_path.name}: "
                f"{out_pf.metadata.num_rows} != {pf.metadata.num_rows}"
            )
        tmp_out.replace(out_path)
    finally:
        for writer in bucket_writers.values():
            writer.close()
        safe_rmtree(tmp_dir)
        if tmp_out.exists():
            tmp_out.unlink()


def main() -> None:
    args = parse_args()
    if args.buckets <= 0:
        raise SystemExit("--buckets must be positive")
    if args.read_batch_size <= 0:
        raise SystemExit("--read-batch-size must be positive")

    files = source_files(args)
    if not files:
        raise SystemExit(f"no parquet files found in {args.source_dir}")

    print(
        f"shuffle {len(files)} files seed={args.seed} buckets={args.buckets} "
        f"source={args.source_dir} out={args.out_dir}",
        flush=True,
    )
    for src_path in files:
        out_path = args.out_dir / src_path.name
        shuffle_file(
            src_path,
            out_path,
            seed=args.seed,
            buckets=args.buckets,
            read_batch_size=args.read_batch_size,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        print(f"{src_path.name} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
