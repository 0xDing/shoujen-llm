import torch

from shoujen.data import _make_packed_pretrain_sample, packed_collate


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
