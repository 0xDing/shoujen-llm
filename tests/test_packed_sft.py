from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from scripts.build_packed_tokenized_sft import (
    EncodedSFTRecord,
    build_split,
    encode_sft_messages,
    load_jsonl_records,
    pack_encoded_records,
    schema_for,
    split_records,
    write_samples,
)
from shoujen.config import ShoujenConfig
from shoujen.data import (
    SFT_ROLE_ASSISTANT,
    SFT_ROLE_SYSTEM,
    SFT_ROLE_THINK,
    SFT_ROLE_THINK_BOUNDARY,
    SFT_ROLE_USER,
    PrepackedParquetSFTDataset,
    is_packed_sft_parquet,
    packed_sft_collate,
)
from shoujen.losses import SemanticTubePredictionLoss
from shoujen.tokenizer import ShoujenTokenizer


class TinyChatTokenizer:
    pad_id = 0
    eos_id = 1
    im_start_id = 2
    im_end_id = 3
    vocab_size = 256

    def __init__(self) -> None:
        self.ids = {"<think>": 4, "</think>": 5}
        self.next_id = 6

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        idx = 0
        while idx < len(text):
            for special in ("<think>", "</think>"):
                if text.startswith(special, idx):
                    out.append(self.ids[special])
                    idx += len(special)
                    break
            else:
                char = text[idx]
                token_id = self.ids.get(char)
                if token_id is None:
                    token_id = self.next_id
                    self.ids[char] = token_id
                    self.next_id += 1
                out.append(token_id)
                idx += 1
        return out


def test_sft_encoding_masks_roles_and_think_spans() -> None:
    tokenizer = TinyChatTokenizer()
    record = encode_sft_messages(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "<think>abc</think>xyz"},
        ],
        tokenizer,
        think_loss_weight=0.1,
        think_boundary_loss_weight=1.0,
    )

    role_weights: dict[int, set[float]] = {}
    for role_id, weight in zip(record.target_role_ids, record.target_loss_weights):
        role_weights.setdefault(role_id, set()).add(round(weight, 3))

    assert role_weights[SFT_ROLE_SYSTEM] == {0.0}
    assert role_weights[SFT_ROLE_USER] == {0.0}
    assert role_weights[SFT_ROLE_THINK] == {0.1}
    assert role_weights[SFT_ROLE_THINK_BOUNDARY] == {1.0}
    assert role_weights[SFT_ROLE_ASSISTANT] == {1.0}
    assert record.target_loss_weights[record.token_ids.index(tokenizer.im_end_id, 1)] == 0.0
    assert record.target_loss_weights[-1] == 1.0

    assert len(record.stp_spans) == 1
    start, end = record.stp_spans[0]
    assert end - start == 3
    assert record.target_role_ids[start:end] == [SFT_ROLE_ASSISTANT] * 3


def test_packed_sft_parquet_schema_padding_and_attention_mask(tmp_path) -> None:
    block_size = 10
    samples = pack_encoded_records(
        [
            EncodedSFTRecord(
                token_ids=[10, 11, 12, 13, 14],
                target_loss_weights=[0.0, 1.0, 1.0, 1.0, 1.0],
                target_role_ids=[0, SFT_ROLE_ASSISTANT, SFT_ROLE_ASSISTANT, SFT_ROLE_ASSISTANT, 0],
                stp_spans=[(1, 4)],
            ),
            EncodedSFTRecord(
                token_ids=[20, 21, 22, 23, 24],
                target_loss_weights=[0.0, 1.0, 1.0, 1.0, 1.0],
                target_role_ids=[
                    0,
                    SFT_ROLE_ASSISTANT,
                    SFT_ROLE_ASSISTANT,
                    SFT_ROLE_ASSISTANT,
                    SFT_ROLE_ASSISTANT,
                ],
                stp_spans=[(1, 5)],
            ),
        ],
        block_size=block_size,
        pad_id=0,
    )
    path = tmp_path / "sft.parquet"
    write_samples(
        samples,
        path,
        schema_for(
            block_size=block_size,
            tokenizer=TinyChatTokenizer(),
            tokenizer_source="tiny",
            source_path=tmp_path / "sft.jsonl",
            think_loss_weight=0.1,
            think_boundary_loss_weight=1.0,
        ),
        write_batch_size=2,
        overwrite=True,
    )

    assert is_packed_sft_parquet(path)
    schema = pq.ParquetFile(path).schema_arrow
    assert schema.metadata[b"shoujen_format"] == b"packed-sft-v1"

    dataset = PrepackedParquetSFTDataset(path, block_size=block_size, shuffle=False)
    sample = next(iter(dataset))
    assert sample["labels"][4].item() == -100
    assert sample["loss_mask"][4].item() == 0.0
    assert sample["labels"][-1].item() == -100
    assert sample["loss_mask"][-1].item() == 0.0
    assert sample["stp_spans"].tolist() == [[1, 4], [6, 9]]

    batch = packed_sft_collate([sample])
    mask = batch["attention_mask"][0, 0]
    assert mask[4, 5].item() is False
    assert mask[5, 4].item() is False
    assert batch["stp_spans"].shape == (1, 2, 2)
    assert batch["stp_span_mask"].tolist() == [[True, True]]


def test_sft_jsonl_shuffle_is_deterministic(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    rows = [
        {"messages": [{"role": "assistant", "content": value}]}
        for value in ("a", "b", "c", "d")
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    records_a = load_jsonl_records(path)
    records_b = load_jsonl_records(path)
    import random

    random.Random(42).shuffle(records_a)
    random.Random(42).shuffle(records_b)
    assert records_a == records_b
    train, val = split_records(records_a, val_ratio=0.25)
    assert len(train) == 3
    assert len(val) == 1


def test_semantic_tube_prediction_accepts_multi_span_with_mask() -> None:
    hidden = torch.randn(1, 8, 4)
    loss_fn = SemanticTubePredictionLoss(samples_per_sequence=2)
    spans = torch.tensor([[[0, 3], [4, 7], [0, 0]]])
    span_mask = torch.tensor([[True, True, False]])

    multi = loss_fn(hidden, spans, span_mask=span_mask)
    single = loss_fn(hidden, torch.tensor([[0, 3]]))
    empty = loss_fn(hidden, spans, span_mask=torch.zeros_like(span_mask))

    assert torch.isfinite(multi)
    assert torch.isfinite(single)
    assert empty.item() == 0.0


def _save_tiny_tokenizer(path) -> None:
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "system": 2,
        "user": 3,
        "assistant": 4,
        "hello": 5,
        "reason": 6,
        "answer": 7,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)


def test_train_sft_packed_smoke(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tok"
    _save_tiny_tokenizer(tokenizer_dir)
    tokenizer = ShoujenTokenizer.load(tokenizer_dir)
    records = [
        {
            "messages": [
                {"role": "system", "content": "hello"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think> reason </think> answer"},
            ]
        }
    ]
    samples = build_split(
        records,
        tokenizer,
        block_size=16,
        think_loss_weight=0.1,
        think_boundary_loss_weight=1.0,
    )
    parquet_path = tmp_path / "train.parquet"
    write_samples(
        samples,
        parquet_path,
        schema_for(
            block_size=16,
            tokenizer=tokenizer,
            tokenizer_source=str(tokenizer_dir),
            source_path=tmp_path / "sft.jsonl",
            think_loss_weight=0.1,
            think_boundary_loss_weight=1.0,
        ),
        write_batch_size=1,
        overwrite=True,
    )
    config = ShoujenConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        layer_pattern=("attention",),
        pattern_repeats=1,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        intermediate_size=32,
        hidden_size_per_layer_input=0,
        attnres_block_size=1,
        max_seq_len=16,
        attention_window=None,
        pad_token_id=tokenizer.model_pad_id,
        eos_token_id=tokenizer.eos_id,
        im_start_token_id=tokenizer.im_start_id,
        im_end_token_id=tokenizer.im_end_id,
        think_start_token_id=tokenizer.think_start_id,
        think_end_token_id=tokenizer.think_end_id,
    )
    config_path = tmp_path / "config.json"
    config.to_json(config_path)
    out_dir = tmp_path / "run"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_sft_packed.py",
            "--tokenizer",
            str(tokenizer_dir),
            "--train-parquet",
            str(parquet_path),
            "--output",
            str(out_dir),
            "--config",
            str(config_path),
            "--block-size",
            "16",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--save-every",
            "1",
            "--log-every",
            "1",
            "--no-amp",
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stdout + "\n" + result.stderr)
    assert (out_dir / "last.pt").exists()
