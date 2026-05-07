"""Tokenizer compatibility wrapper.

The default tokenizer is the Hugging Face tokenizer at
`AgentBull/CJK-Tokenizer`. It extends the LLaMA tokenizer with single-token CJK
characters while keeping LLaMA's normal subword behavior for other scripts.

The legacy character-level tokenizer is still supported for old local
`vocab.json` files and `scripts/build_tokenizer.py`, but new training and
inference paths should call `ShoujenTokenizer.load()` without a local vocab.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_ID = "AgentBull/CJK-Tokenizer"
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
PROJECT_ADDITIONAL_SPECIAL_TOKENS = [IM_START_TOKEN, IM_END_TOKEN]

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


def _normalize_hf_tokenizer_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "huggingface.co":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return source


class ShoujenTokenizer:
    def __init__(self, tokenizer_or_tokens: PreTrainedTokenizerBase | Sequence[str]):
        self._hf_tokenizer: PreTrainedTokenizerBase | None = None
        self._tokens: list[str] | None = None
        self._id_of: dict[str, int] | None = None

        if isinstance(tokenizer_or_tokens, PreTrainedTokenizerBase):
            self._hf_tokenizer = tokenizer_or_tokens
            self._ensure_project_tokens()
            return

        tokens = tokenizer_or_tokens
        if list(tokens[:BYTE_OFFSET]) != SPECIAL_TOKENS:
            raise ValueError("Vocab must start with the special tokens.")
        for i in range(N_BYTES):
            expected = _byte_token(i)
            if tokens[BYTE_OFFSET + i] != expected:
                raise ValueError(
                    f"Vocab id {BYTE_OFFSET + i} must be {expected}, got {tokens[BYTE_OFFSET + i]}"
                )
        self._tokens = list(tokens)
        self._id_of = {t: i for i, t in enumerate(self._tokens)}

    @property
    def is_hf(self) -> bool:
        return self._hf_tokenizer is not None

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._hf_tokenizer is None:
            raise TypeError("This tokenizer was loaded from a legacy local vocab.json")
        return self._hf_tokenizer

    def _ensure_project_tokens(self) -> None:
        tok = self.hf_tokenizer
        missing_chat_tokens = [
            token
            for token in PROJECT_ADDITIONAL_SPECIAL_TOKENS
            if tok.convert_tokens_to_ids(token) == tok.unk_token_id
        ]
        if missing_chat_tokens:
            try:
                tok.add_special_tokens(
                    {"additional_special_tokens": missing_chat_tokens},
                    replace_additional_special_tokens=False,
                )
            except TypeError:
                existing = list(getattr(tok, "additional_special_tokens", []) or [])
                tok.add_special_tokens(
                    {"additional_special_tokens": existing + missing_chat_tokens}
                )

        missing = [
            token
            for token in PROJECT_ADDITIONAL_SPECIAL_TOKENS
            if tok.convert_tokens_to_ids(token) == tok.unk_token_id
        ]
        if missing:
            raise ValueError(f"Tokenizer is missing required special tokens: {missing}")

    @property
    def vocab_size(self) -> int:
        if self._hf_tokenizer is not None:
            return len(self._hf_tokenizer)
        assert self._tokens is not None
        return len(self._tokens)

    @property
    def pad_id(self) -> int:
        if self._hf_tokenizer is not None:
            token_id = self._hf_tokenizer.pad_token_id
            return self.eos_id if token_id is None else int(token_id)
        return PAD_ID

    @property
    def model_pad_id(self) -> int | None:
        if self._hf_tokenizer is not None:
            token_id = self._hf_tokenizer.pad_token_id
            return None if token_id is None else int(token_id)
        return PAD_ID

    @property
    def eos_id(self) -> int:
        if self._hf_tokenizer is not None:
            token_id = self._hf_tokenizer.eos_token_id
            if token_id is None:
                raise ValueError("Tokenizer does not define an eos token")
            return int(token_id)
        return EOS_ID

    @property
    def im_start_id(self) -> int:
        if self._hf_tokenizer is not None:
            return int(self._hf_tokenizer.convert_tokens_to_ids(IM_START_TOKEN))
        return IM_START_ID

    @property
    def im_end_id(self) -> int:
        if self._hf_tokenizer is not None:
            return int(self._hf_tokenizer.convert_tokens_to_ids(IM_END_TOKEN))
        return IM_END_ID

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        if self._hf_tokenizer is not None:
            ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
            if add_eos:
                ids.append(self.eos_id)
            return [int(tid) for tid in ids]

        assert self._id_of is not None
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

    def encode_batch(self, texts: Sequence[str], add_eos: bool = False) -> list[list[int]]:
        if not texts:
            return []
        if self._hf_tokenizer is not None:
            encoded = self._hf_tokenizer(
                list(texts),
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            eos_id = self.eos_id
            if add_eos:
                return [[int(tid) for tid in ids] + [eos_id] for ids in encoded]
            return [[int(tid) for tid in ids] for ids in encoded]

        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(
                list(ids),
                skip_special_tokens=skip_special,
                clean_up_tokenization_spaces=False,
            )

        assert self._tokens is not None
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
            ids.append(self.im_start_id); mask.append(0)
            for tid in self.encode(role + "\n"):
                ids.append(tid); mask.append(0)
            content_ids = self.encode(content)
            is_assistant = role == "assistant"
            for tid in content_ids:
                ids.append(tid)
                mask.append(1 if is_assistant else 0)
            ids.append(self.im_end_id)
            mask.append(1 if is_assistant else 0)
            for tid in self.encode("\n"):
                ids.append(tid); mask.append(0)
        if add_generation_prompt:
            ids.append(self.im_start_id); mask.append(0)
            for tid in self.encode("assistant\n"):
                ids.append(tid); mask.append(0)
        return ids, mask

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if self._hf_tokenizer is not None:
            path.mkdir(parents=True, exist_ok=True)
            self._hf_tokenizer.save_pretrained(path)
            return
        assert self._tokens is not None
        path.write_text(json.dumps(self._tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path = DEFAULT_TOKENIZER_ID,
        **kwargs,
    ) -> "ShoujenTokenizer":
        source = _normalize_hf_tokenizer_source(str(model_id_or_path))
        kwargs.setdefault("use_fast", False)
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ShoujenTokenizer":
        if path is None:
            return cls.from_pretrained(DEFAULT_TOKENIZER_ID)

        path_str = str(path)
        path_obj = Path(path_str)
        if path_obj.is_file():
            tokens = json.loads(path_obj.read_text(encoding="utf-8"))
            return cls(tokens)

        return cls.from_pretrained(path_str)

    @classmethod
    def from_vocab_file(cls, path: str | Path) -> "ShoujenTokenizer":
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
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
    """Built-in coverage for the legacy local character tokenizer."""
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
