import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from shoujen.data import (
    PrepackedParquetPretrainDataset,
    SFTDataset,
    _make_packed_pretrain_sample,
    is_prepacked_parquet,
    packed_collate,
)


def test_packed_sample_masks_cross_document_targets() -> None:
    sample = _make_packed_pretrain_sample(
        token_ids=[10, 11, 1, 20, 21, 1, 30],
        seq_ids=[0, 0, 0, 1, 1, 1, 2],
        position_ids=[0, 1, 2, 0, 1, 2, 0],
        sequence_starts=[True, False, False, True, False, False, True],
        block_size=6,
    )

    assert sample["input_ids"].tolist() == [10, 11, 1, 20, 21, 1]
    assert sample["labels"].tolist() == [11, 1, -100, 21, 1, -100]
    assert sample["loss_mask"].tolist() == [1, 1, 0, 1, 1, 0]
    assert sample["position_ids"].tolist() == [0, 1, 2, 0, 1, 2]
    assert sample["sequence_start_mask"].tolist() == [True, False, False, True, False, False]


def test_packed_collate_builds_document_local_causal_mask() -> None:
    sample = _make_packed_pretrain_sample(
        token_ids=[10, 11, 1, 20, 21, 1, 30],
        seq_ids=[0, 0, 0, 1, 1, 1, 2],
        position_ids=[0, 1, 2, 0, 1, 2, 0],
        sequence_starts=[True, False, False, True, False, False, True],
        block_size=6,
    )
    batch = packed_collate([sample])
    mask = batch["attention_mask"][0, 0]

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_sft_padding_does_not_mask_real_eos_when_pad_falls_back_to_eos(tmp_path) -> None:
    class TinyTokenizer:
        pad_id = 1
        eos_id = 1

        def encode_chat(self, messages):
            assert messages == [{"role": "assistant", "content": "ok"}]
            return [10, 20], [0, 1]

    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "ok"}]}) + "\n",
        encoding="utf-8",
    )

    dataset = SFTDataset(path, TinyTokenizer(), block_size=5)
    sample = dataset[0]

    assert sample["input_ids"].tolist() == [10, 20, 1, 1, 1]
    assert sample["labels"].tolist() == [20, 1, -100, -100, -100]
    assert sample["loss_mask"].tolist() == [1, 1, 0, 0, 0]


def test_prepacked_parquet_dataset_matches_packed_sample(tmp_path) -> None:
    path = tmp_path / "packed.parquet"
    block_size = 6
    schema = pa.schema(
        [
            ("token_ids", pa.list_(pa.uint16(), list_size=block_size + 1)),
            ("seq_ids", pa.list_(pa.uint16(), list_size=block_size + 1)),
            ("position_ids", pa.list_(pa.uint32(), list_size=block_size + 1)),
            ("sequence_starts", pa.list_(pa.bool_(), list_size=block_size + 1)),
        ],
        metadata={b"block_size": str(block_size).encode("ascii")},
    )
    table = pa.Table.from_pydict(
        {
            "token_ids": [[10, 11, 1, 20, 21, 1, 30]],
            "seq_ids": [[0, 0, 0, 1, 1, 1, 2]],
            "position_ids": [[0, 1, 2, 0, 1, 2, 0]],
            "sequence_starts": [[True, False, False, True, False, False, True]],
        },
        schema=schema,
    )
    pq.write_table(table, path)

    assert is_prepacked_parquet(path)
    dataset = PrepackedParquetPretrainDataset(path, block_size=block_size, shuffle=False)
    sample = next(iter(dataset))

    assert sample["input_ids"].tolist() == [10, 11, 1, 20, 21, 1]
    assert sample["labels"].tolist() == [11, 1, -100, 21, 1, -100]
    assert sample["loss_mask"].tolist() == [1, 1, 0, 1, 1, 0]
    assert sample["position_ids"].tolist() == [0, 1, 2, 0, 1, 2]
    assert sample["sequence_start_mask"].tolist() == [True, False, False, True, False, False]

    with pytest.raises(ValueError, match="block_size=6"):
        PrepackedParquetPretrainDataset(path, block_size=5, shuffle=False)
