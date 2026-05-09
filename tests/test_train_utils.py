import pytest
import torch
import torch.nn as nn

from shoujen.train_utils import (
    filter_optimizer_state_for_param_shapes,
    load_checkpoint,
    warmup_stable_decay_lr,
)


def test_wsd_lr_defaults_to_decay_to_zero() -> None:
    assert warmup_stable_decay_lr(10, warmup=2, max_steps=10, stable_steps=3) == 0.0


def test_wsd_lr_respects_min_ratio() -> None:
    assert warmup_stable_decay_lr(0, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == 0.5
    assert warmup_stable_decay_lr(2, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == 1.0
    assert warmup_stable_decay_lr(7, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == pytest.approx(0.68)
    assert warmup_stable_decay_lr(10, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == 0.2


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, 3)
        self.model.embed_tokens_per_layer = nn.Embedding(vocab_size, 5)
        self.lm_head = nn.Linear(3, vocab_size, bias=False)


def test_load_checkpoint_allows_appended_vocab_rows(tmp_path) -> None:
    old_model = TinyCausalLM(vocab_size=4)
    with torch.no_grad():
        old_model.model.embed_tokens.weight.fill_(1.0)
        old_model.model.embed_tokens_per_layer.weight.fill_(2.0)
        old_model.lm_head.weight.fill_(3.0)
    ckpt_path = tmp_path / "old.pt"
    torch.save({"model": old_model.state_dict(), "config": {}, "step": 7}, ckpt_path)

    new_model = TinyCausalLM(vocab_size=6)
    state = load_checkpoint(ckpt_path, new_model, allow_appended_vocab=True)

    assert state["step"] == 7
    assert len(state["_shoujen_appended_vocab_keys"]) == 3
    assert torch.equal(new_model.model.embed_tokens.weight[:4], old_model.model.embed_tokens.weight)
    assert torch.equal(
        new_model.model.embed_tokens_per_layer.weight[:4],
        old_model.model.embed_tokens_per_layer.weight,
    )
    assert torch.equal(new_model.lm_head.weight[:4], old_model.lm_head.weight)
    assert new_model.model.embed_tokens.weight.shape[0] == 6


def test_filter_optimizer_state_skips_mismatched_param_shapes() -> None:
    old_model = TinyCausalLM(vocab_size=4)
    old_opt = torch.optim.AdamW(old_model.parameters(), lr=0.1)
    old_model.model.embed_tokens.weight.sum().backward()
    old_opt.step()

    new_model = TinyCausalLM(vocab_size=6)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=0.1)
    filtered, skipped = filter_optimizer_state_for_param_shapes(new_opt, old_opt.state_dict())

    assert skipped
    assert len(filtered["state"]) < len(old_opt.state_dict()["state"])
    new_opt.load_state_dict(filtered)
