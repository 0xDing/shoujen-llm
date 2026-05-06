"""Build the Shoujen tokenizer vocabulary from external character lists.

Default sources for CJK come from https://github.com/zispace/hanzi-chars per the
project README. The script downloads the relevant text files, unions all
characters, adds the built-in coverage (ASCII / Greek / Cyrillic / kana /
punctuation / fullwidth), and writes a vocab JSON file.

Usage:
    uv run scripts/build_tokenizer.py --output data/vocab.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.tokenizer import ShoujenTokenizer, default_charset  # noqa: E402

HANZI_BASE = "https://raw.githubusercontent.com/zispace/hanzi-chars/main"
HANZI_FILES = [
    "data-charlist/《通用规范汉字表》（2013年）一级字.txt",
    "data-charlist/《通用规范汉字表》（2013年）二级字.txt",
    "data-charlist/《通用规范汉字表》（2013年）三级字.txt",
    "data-charlist/香港《常用字表》.txt",
    "data-charlist/臺灣《常用國字表》（1982年）.txt",
    "data-charlist/臺灣《次常用國字表》（1982年）.txt",
    "data-charlist/日本《常用漢字表》（2010年）.txt",
    "data-charlist/日本《学年別漢字配当表》（2017年）.txt",
    "data-charlist/韩国《漢文教育用基礎漢字》（2000年版）.txt",
]


def parse_hanzi_chars(text: str) -> set[str]:
    """Parse a hanzi-chars text file.

    Repo files are UTF-8 text where comments start with "#"; every data line
    contributes the first character, with optional variants after it.
    """
    chars: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        chars.add(line[0])
    return chars


def fetch_chars_online(timeout: float = 30.0) -> set[str]:
    import requests

    chars: set[str] = set()
    for rel in HANZI_FILES:
        url = f"{HANZI_BASE}/{quote(rel, safe='/')}"
        print(f"fetching {url}", flush=True)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        r.encoding = "utf-8"
        file_chars = parse_hanzi_chars(r.text)
        print(f"  parsed {len(file_chars)} characters", flush=True)
        chars |= file_chars
    return chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Output vocab.json path")
    args = ap.parse_args()

    chars = set(default_charset())
    cjk = fetch_chars_online()
    print(f"fetched {len(cjk)} CJK characters")
    chars |= cjk

    tok = ShoujenTokenizer.from_charset(chars, pad_to_multiple=128)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)
    print(f"wrote vocab to {out} ({tok.vocab_size} tokens)")


if __name__ == "__main__":
    main()
