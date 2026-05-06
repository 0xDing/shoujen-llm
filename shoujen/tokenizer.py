"""Character-level tokenizer with UTF-8 byte fallback.

Layout of the vocabulary:
    0..3        Special tokens: <pad>, <eos>, <im_start>, <im_end>
    4..259      Byte fallback tokens: <byte_00>..<byte_FF>
    260..       Character tokens, one id per Unicode codepoint

Text is normalized in both `from_charset` and `encode` via a two-step pipeline:
    1. Strip control / bidi / zero-width noise (Nmt-inspired, but conservative
       -- keeps \\t \\n \\r and ZWNJ/ZWJ).
    2. NFKC, so fullwidth/halfwidth, ligatures, circled digits, etc. fold to
       canonical forms.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

SPECIAL_TOKENS = ["<pad>", "<eos>", "<im_start>", "<im_end>"]
PAD_ID, EOS_ID, IM_START_ID, IM_END_ID = 0, 1, 2, 3
BYTE_OFFSET = len(SPECIAL_TOKENS)
N_BYTES = 256
CHAR_OFFSET = BYTE_OFFSET + N_BYTES

NORMALIZATION = "NFKC"

# Strip noise that NFKC leaves alone. Modeled on HF tokenizers' Nmt normalizer
# but tailored for char-level CJK + code: full Nmt folds \t \n \r and ZWJ/ZWNJ
# to space, which destroys code/text structure and emoji sequences. Here we
# only drop chars that carry no signal: C0 controls except \t \n \r, DEL + C1
# controls, ZWSP, LRM/RLM, line/paragraph separators, BOM, U+FFFD replacement.
# Kept on purpose: ZWNJ (U+200C), ZWJ (U+200D).
_NOISE_RE = re.compile(
    "["
    "\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F"
    "\u200B\u200E\u200F"  # ZWSP, LRM, RLM (keeps ZWNJ \u200C, ZWJ \u200D)
    "\u2028\u2029"          # LSEP, PSEP
    "\uFEFF\uFFFD"          # BOM, replacement
    "]"
)


def normalize(text: str) -> str:
    return unicodedata.normalize(NORMALIZATION, _NOISE_RE.sub("", text))


def _byte_token(b: int) -> str:
    return f"<byte_{b:02X}>"


def _is_byte_token(tok: str) -> bool:
    return len(tok) == 8 and tok.startswith("<byte_") and tok.endswith(">")


def _byte_token_value(tok: str) -> int:
    return int(tok[6:8], 16)


class ShoujenTokenizer:
    def __init__(self, tokens: Sequence[str]):
        if list(tokens[:BYTE_OFFSET]) != SPECIAL_TOKENS:
            raise ValueError("Vocab must start with the special tokens.")
        for i in range(N_BYTES):
            expected = _byte_token(i)
            if tokens[BYTE_OFFSET + i] != expected:
                raise ValueError(
                    f"Vocab id {BYTE_OFFSET + i} must be {expected}, got {tokens[BYTE_OFFSET + i]}"
                )
        self._tokens: list[str] = list(tokens)
        self._id_of: dict[str, int] = {t: i for i, t in enumerate(self._tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def pad_id(self) -> int: return PAD_ID
    @property
    def eos_id(self) -> int: return EOS_ID
    @property
    def im_start_id(self) -> int: return IM_START_ID
    @property
    def im_end_id(self) -> int: return IM_END_ID

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for ch in normalize(text):
            tid = self._id_of.get(ch)
            if tid is not None:
                ids.append(tid)
            else:
                for b in ch.encode("utf-8"):
                    ids.append(BYTE_OFFSET + b)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out: list[str] = []
        byte_buf: list[int] = []

        def flush_bytes():
            if byte_buf:
                out.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in ids:
            if tid < 0 or tid >= len(self._tokens):
                flush_bytes()
                continue
            tok = self._tokens[tid]
            if tid < BYTE_OFFSET:
                flush_bytes()
                if not skip_special:
                    out.append(tok)
                continue
            if tid < CHAR_OFFSET:
                byte_buf.append(_byte_token_value(tok))
                continue
            flush_bytes()
            out.append(tok)
        flush_bytes()
        return "".join(out)

    def encode_chat(
        self,
        messages: Sequence[dict],
        add_generation_prompt: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Encode a chat. Returns (input_ids, assistant_mask).

        Each message: {"role": "user"|"assistant"|"system", "content": str}.
        Format per turn:
            <im_start>{role}\n{content}<im_end>\n
        assistant_mask is 1 over the assistant `content` and the closing <im_end>;
        0 elsewhere. This makes the SFT loss target only assistant tokens.
        """
        ids: list[int] = []
        mask: list[int] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            ids.append(IM_START_ID); mask.append(0)
            for tid in self.encode(role + "\n"):
                ids.append(tid); mask.append(0)
            content_ids = self.encode(content)
            is_assistant = role == "assistant"
            for tid in content_ids:
                ids.append(tid)
                mask.append(1 if is_assistant else 0)
            ids.append(IM_END_ID)
            mask.append(1 if is_assistant else 0)
            for tid in self.encode("\n"):
                ids.append(tid); mask.append(0)
        if add_generation_prompt:
            ids.append(IM_START_ID); mask.append(0)
            for tid in self.encode("assistant\n"):
                ids.append(tid); mask.append(0)
        return ids, mask

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self._tokens, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "ShoujenTokenizer":
        tokens = json.loads(Path(path).read_text())
        return cls(tokens)

    @classmethod
    def from_charset(
        cls,
        charset: Iterable[str],
        pad_to_multiple: int = 1,
    ) -> "ShoujenTokenizer":
        seen: set[str] = set()
        chars: list[str] = []
        for ch in charset:
            for c in normalize(ch):
                if c in seen:
                    continue
                seen.add(c)
                chars.append(c)
        chars.sort()
        tokens: list[str] = list(SPECIAL_TOKENS)
        tokens.extend(_byte_token(b) for b in range(N_BYTES))
        tokens.extend(chars)
        if pad_to_multiple > 1:
            pad = (-len(tokens)) % pad_to_multiple
            tokens.extend(f"<unused_{i}>" for i in range(pad))
        return cls(tokens)


def default_charset() -> list[str]:
    """A built-in fallback character coverage when external lists are unavailable.

    Covers what the README requires *except* the heavy CJK lists. Use
    scripts/build_tokenizer.py to produce a full vocab including CJK.
    """
    chars: set[str] = set()
    for cp in range(0x20, 0x7F):  # printable ASCII
        chars.add(chr(cp))
    chars.update(["\t", "\n"])
    for cp in range(0x0370, 0x0400):  # Greek
        chars.add(chr(cp))
    for cp in range(0x0400, 0x0500):  # Cyrillic
        chars.add(chr(cp))
    for cp in range(0x3040, 0x30A0):  # Hiragana
        chars.add(chr(cp))
    for cp in range(0x30A0, 0x3100):  # Katakana
        chars.add(chr(cp))
    for cp in range(0x3000, 0x3040):  # CJK Symbols and Punctuation
        chars.add(chr(cp))
    for cp in range(0x2000, 0x2070):  # General Punctuation
        chars.add(chr(cp))
    for cp in range(0xFF00, 0xFFF0):  # Halfwidth and Fullwidth Forms
        chars.add(chr(cp))
    return sorted(chars)
