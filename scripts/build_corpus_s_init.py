from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CHUNK_SIZE = 2000
BATCH_FLUSH = 10000
JSON_ARRAY_READ_SIZE = 1024 * 1024
DATA_ROOT = Path("data/train/s-init")
DEFAULT_OUT = Path("data/processed/s-init.parquet")

TRANSLATION_TEMPLATES = [
    "中文：{zh}\n英文：{en}",
    "{zh}\n翻译为英文是：\n{en}",
    "英文：{en}\n\n中文：{zh}",
    "{en}\n该英文翻译为中文是：\n{zh}",
    "请把下面这句中文翻译成英文：\n{zh}\n答：{en}",
]


def translation_text(zh: str, en: str, idx: int) -> str:
    return TRANSLATION_TEMPLATES[idx % len(TRANSLATION_TEMPLATES)].format(zh=zh, en=en)


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


def feed(buffer: str, record_text: str, source: str, sink) -> str:
    if not record_text:
        return buffer
    buffer += record_text + "\n\n"
    return emit_from_buffer(buffer, source, False, sink)


def process_parquet(path: Path, sink, limit: int | None) -> None:
    source = path.name
    pf = pq.ParquetFile(str(path))
    names = set(pf.schema_arrow.names)

    if "TEXT" in names:
        text_col = "TEXT"
    elif "og_full_text" in names and "translated_text" in names:
        text_col = None
    elif "text" in names:
        text_col = "text"
    else:
        print(f"warn: {source} has unknown parquet schema {sorted(names)[:8]}", file=sys.stderr)
        return

    cols = [text_col] if text_col else ["og_full_text", "translated_text"]
    buffer = ""
    seen = 0
    done = False
    for batch in pf.iter_batches(batch_size=1024, columns=cols):
        if text_col is None:
            zh_col = batch.column("og_full_text").to_pylist()
            en_col = batch.column("translated_text").to_pylist()
            for zh, en in zip(zh_col, en_col):
                if limit is not None and seen >= limit:
                    done = True
                    break
                zh = zh or ""
                en = en or ""
                if not zh and not en:
                    continue
                text = translation_text(zh, en, seen) if (zh and en) else (zh or en)
                buffer = feed(buffer, text, source, sink)
                seen += 1
        else:
            for val in batch.column(text_col).to_pylist():
                if limit is not None and seen >= limit:
                    done = True
                    break
                if not val:
                    continue
                buffer = feed(buffer, val, source, sink)
                seen += 1
        if done:
            break

    emit_from_buffer(buffer, source, True, sink)


def detect_jsonl_schema(path: Path) -> str | None:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return None
            if not isinstance(rec, dict):
                return None
            keys = rec.keys()
            if "zh_text" in keys:
                return "translation_blockwise"
            if "段落" in keys and "ID" not in keys:
                return "segments"
            if "回复" in keys:
                return "forum"
            if "instruction" in keys:
                return "instruction"
            if "问" in keys:
                return "wikihow"
            return None
    return None


def process_jsonl(path: Path, sink, limit: int | None) -> None:
    source = path.name
    schema = detect_jsonl_schema(path)
    if schema is None:
        print(f"warn: {source} unrecognised jsonl schema", file=sys.stderr)
        return

    buffer = ""
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if limit is not None and seen >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue

            text = ""
            if schema == "translation_blockwise":
                zh = (rec.get("zh_text") or "").strip()
                en = (rec.get("en_text") or "").strip()
                if zh and en:
                    text = translation_text(zh, en, seen)
                elif zh:
                    text = zh
                else:
                    continue
            elif schema == "segments":
                paras = rec.get("段落") or []
                parts = [p.get("内容", "") for p in paras if isinstance(p, dict)]
                text = "\n".join(p for p in parts if p)
            elif schema == "forum":
                topic = rec.get("主题") or ""
                replies = rec.get("回复") or []
                reply_texts = [r.get("回复", "") for r in replies if isinstance(r, dict)]
                text = topic + "\n\n" + "\n".join(reply_texts)
            elif schema == "instruction":
                instr = rec.get("instruction") or ""
                inp = rec.get("input") or ""
                out = rec.get("output") or ""
                text = instr + (("\n" + inp) if inp else "") + "\n" + out
            elif schema == "wikihow":
                q = rec.get("问") or ""
                a = rec.get("答") or ""
                text = q + "\n" + a

            buffer = feed(buffer, text, source, sink)
            seen += 1

    emit_from_buffer(buffer, source, True, sink)


def iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False

    def fill(f) -> None:
        nonlocal buffer, eof
        chunk = f.read(JSON_ARRAY_READ_SIZE)
        if chunk:
            buffer += chunk
        else:
            eof = True

    with open(path, encoding="utf-8") as f:
        fill(f)
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer) or eof:
                break
            buffer = ""
            pos = 0
            fill(f)
        if pos >= len(buffer) or buffer[pos] != "[":
            raise ValueError("top-level is not a JSON array")
        pos += 1

        while True:
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    break
                buffer = ""
                pos = 0
                fill(f)
            if pos >= len(buffer):
                return
            if buffer[pos] == "]":
                return
            if buffer[pos] == ",":
                pos += 1
                continue

            while True:
                try:
                    rec, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    yield rec
                    if pos > JSON_ARRAY_READ_SIZE:
                        buffer = buffer[pos:]
                        pos = 0
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    if pos > 0:
                        buffer = buffer[pos:]
                        pos = 0
                    fill(f)


def json_array_record(rec: dict, idx: int, default_source: str) -> tuple[str, str]:
    record_source = str(rec.get("source") or default_source)
    if "completion" in rec:
        return str(rec.get("completion") or ""), record_source

    if any(key in rec for key in ("query", "thought", "answer")):
        q = rec.get("query") or ""
        t = rec.get("thought") or ""
        a = rec.get("answer") or ""
        return f"{q}\n{t}\n{a}", record_source

    if "text" in rec:
        return str(rec.get("text") or ""), record_source

    if "content" in rec:
        return str(rec.get("content") or ""), record_source

    return "", record_source


def process_json_array(path: Path, sink, limit: int | None) -> None:
    source = path.name

    buffer = ""
    seen = 0
    last_source = source
    try:
        records = iter_json_array(path)
        for rec in records:
            if limit is not None and seen >= limit:
                break
            if not isinstance(rec, dict):
                continue
            text, record_source = json_array_record(rec, seen, source)
            if not text:
                continue
            buffer = feed(buffer, text, record_source, sink)
            last_source = record_source
            seen += 1
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"warn: {source} failed to parse: {e}", file=sys.stderr)
        return

    emit_from_buffer(buffer, last_source, True, sink)


def process_text(path: Path, sink, limit: int | None) -> None:
    source = path.name
    buffer = ""
    seen = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if limit is not None and seen >= limit:
                    break
                buffer = feed(buffer, line.rstrip("\n"), source, sink)
                seen += 1
    except (OSError, UnicodeDecodeError) as e:
        print(f"warn: {source} failed to read: {e}", file=sys.stderr)
        return

    emit_from_buffer(buffer, source, True, sink)


def process_file(path: Path, sink, limit: int | None) -> int:
    before = sink.count
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            process_parquet(path, sink, limit)
        elif suffix == ".jsonl":
            process_jsonl(path, sink, limit)
        elif suffix == ".json":
            process_json_array(path, sink, limit)
        else:
            process_text(path, sink, limit)
    except Exception as e:
        print(f"warn: {path.name} failed: {e}", file=sys.stderr)
    return sink.count - before


class Sink:
    def __init__(self, writer: pq.ParquetWriter, schema: pa.Schema) -> None:
        self._writer = writer
        self._schema = schema
        self._texts: list[str] = []
        self._sources: list[str] = []
        self.count = 0

    def __call__(self, text: str, source: str) -> None:
        self._texts.append(text)
        self._sources.append(source)
        self.count += 1
        if len(self._texts) >= BATCH_FLUSH:
            self.flush()

    def flush(self) -> None:
        if not self._texts:
            return
        self._writer.write_table(
            pa.table({"text": self._texts, "source": self._sources}, schema=self._schema)
        )
        self._texts.clear()
        self._sources.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not DATA_ROOT.exists():
        print(f"error: {DATA_ROOT} does not exist", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in DATA_ROOT.rglob("*") if p.is_file() and p.name != ".DS_Store")

    schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    writer = pq.ParquetWriter(str(args.out), schema, compression="snappy")
    sink = Sink(writer, schema)

    try:
        for path in tqdm(files, desc="s-init"):
            produced = process_file(path, sink, args.limit)
            if produced == 0:
                print(f"warn: {path.name} produced 0 chunks", file=sys.stderr)
        sink.flush()
    finally:
        writer.close()

    print(f"wrote {sink.count} chunks to {args.out}")


if __name__ == "__main__":
    main()
