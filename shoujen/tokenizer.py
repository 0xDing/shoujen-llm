"""Tokenizer wrapper.

The default tokenizer is the Hugging Face tokenizer at
`AgentBull/CJK-Tokenizer`. It extends the LLaMA tokenizer with single-token CJK
characters while keeping LLaMA's normal subword behavior for other scripts.

Training and inference should call `ShoujenTokenizer.load()` or pass another
Hugging Face tokenizer id/URL/saved tokenizer directory. Legacy local
`vocab.json` character tokenizers are no longer supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_ID = "AgentBull/CJK-Tokenizer"
IM_START_TOKEN = "<im_start>"
IM_END_TOKEN = "<im_end>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
SPECIAL_TOKENS = [IM_START_TOKEN, IM_END_TOKEN, THINK_START_TOKEN, THINK_END_TOKEN]
PROJECT_ADDITIONAL_SPECIAL_TOKENS = SPECIAL_TOKENS


def _normalize_hf_tokenizer_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "huggingface.co":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return source


def _token_missing(tokenizer: PreTrainedTokenizerBase, token: str) -> bool:
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id is None or (
        tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id
    )


def _special_token_strings(tokenizer: PreTrainedTokenizerBase) -> set[str]:
    tokens: set[str] = set()
    for attr in ("all_special_tokens", "extra_special_tokens", "additional_special_tokens"):
        for token in getattr(tokenizer, attr, []) or []:
            tokens.add(str(token))

    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for value in special_tokens_map.values():
        if isinstance(value, (list, tuple)):
            tokens.update(str(token) for token in value)
        else:
            tokens.add(str(value))
    return tokens


def _extra_special_token_strings(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    tokens: list[str] = []
    for attr in ("extra_special_tokens", "additional_special_tokens"):
        for token in getattr(tokenizer, attr, []) or []:
            token = str(token)
            if token not in tokens:
                tokens.append(token)
    return tokens


class ShoujenTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("ShoujenTokenizer requires a Hugging Face tokenizer instance")
        self._hf_tokenizer = tokenizer
        self._ensure_project_tokens()

    @property
    def is_hf(self) -> bool:
        return True

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._hf_tokenizer

    def _ensure_project_tokens(self) -> None:
        tok = self.hf_tokenizer
        existing_special_tokens = _special_token_strings(tok)
        missing_special_tokens = [
            token
            for token in PROJECT_ADDITIONAL_SPECIAL_TOKENS
            if token not in existing_special_tokens
        ]
        if missing_special_tokens:
            try:
                tok.add_special_tokens(
                    {"additional_special_tokens": missing_special_tokens},
                    replace_additional_special_tokens=False,
                )
            except TypeError:
                existing_extra_tokens = _extra_special_token_strings(tok)
                tok.add_special_tokens(
                    {"additional_special_tokens": existing_extra_tokens + missing_special_tokens}
                )

        missing = [
            token
            for token in PROJECT_ADDITIONAL_SPECIAL_TOKENS
            if _token_missing(tok, token)
        ]
        if missing:
            raise ValueError(f"Tokenizer is missing required special tokens: {missing}")
        non_special = [
            token
            for token in PROJECT_ADDITIONAL_SPECIAL_TOKENS
            if token not in _special_token_strings(tok)
        ]
        if non_special:
            raise ValueError(f"Tokenizer did not register required special tokens: {non_special}")

    @property
    def vocab_size(self) -> int:
        return len(self._hf_tokenizer)

    @property
    def pad_id(self) -> int:
        token_id = self._hf_tokenizer.pad_token_id
        return self.eos_id if token_id is None else int(token_id)

    @property
    def model_pad_id(self) -> int | None:
        token_id = self._hf_tokenizer.pad_token_id
        return None if token_id is None else int(token_id)

    @property
    def eos_id(self) -> int:
        token_id = self._hf_tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("Tokenizer does not define an eos token")
        return int(token_id)

    @property
    def im_start_id(self) -> int:
        return int(self._hf_tokenizer.convert_tokens_to_ids(IM_START_TOKEN))

    @property
    def im_end_id(self) -> int:
        return int(self._hf_tokenizer.convert_tokens_to_ids(IM_END_TOKEN))

    @property
    def think_start_id(self) -> int:
        return int(self._hf_tokenizer.convert_tokens_to_ids(THINK_START_TOKEN))

    @property
    def think_end_id(self) -> int:
        return int(self._hf_tokenizer.convert_tokens_to_ids(THINK_END_TOKEN))

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
        out = [int(tid) for tid in ids]
        if add_eos:
            out.append(self.eos_id)
        return out

    def encode_batch(self, texts: Sequence[str], add_eos: bool = False) -> list[list[int]]:
        if not texts:
            return []
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

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self._hf_tokenizer.decode(
            list(ids),
            skip_special_tokens=skip_special,
            clean_up_tokenization_spaces=False,
        )

    def encode_chat(
        self,
        messages: Sequence[dict],
        add_generation_prompt: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Encode a chat. Returns (input_ids, assistant_mask).

        Each message: {"role": "user"|"assistant"|"system", "content": str}.
        Format per turn:
            <im_start>{role}\n{content}<im_end>\n
        assistant_mask is 1 over the assistant `content` and the closing
        <im_end>; 0 elsewhere. This makes the SFT loss target only assistant
        tokens.
        """
        ids: list[int] = []
        mask: list[int] = []
        for message in messages:
            role = message["role"]
            content = message["content"]

            ids.append(self.im_start_id)
            mask.append(0)
            for tid in self.encode(role + "\n"):
                ids.append(tid)
                mask.append(0)

            content_ids = self.encode(content)
            is_assistant = role == "assistant"
            for tid in content_ids:
                ids.append(tid)
                mask.append(1 if is_assistant else 0)

            ids.append(self.im_end_id)
            mask.append(1 if is_assistant else 0)
            for tid in self.encode("\n"):
                ids.append(tid)
                mask.append(0)

        if add_generation_prompt:
            ids.append(self.im_start_id)
            mask.append(0)
            for tid in self.encode("assistant\n"):
                ids.append(tid)
                mask.append(0)
        return ids, mask

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._hf_tokenizer.save_pretrained(path)

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
    def load(cls, source: str | Path | None = None) -> "ShoujenTokenizer":
        if source is None:
            return cls.from_pretrained(DEFAULT_TOKENIZER_ID)

        path = Path(str(source))
        if path.is_file():
            raise ValueError(
                "Local tokenizer files such as data/vocab.json are no longer supported; "
                "pass a Hugging Face tokenizer id/URL or a saved tokenizer directory."
            )
        return cls.from_pretrained(source)
