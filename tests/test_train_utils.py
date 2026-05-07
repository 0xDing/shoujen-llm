import pytest

from shoujen.train_utils import warmup_stable_decay_lr


def test_wsd_lr_defaults_to_decay_to_zero() -> None:
    assert warmup_stable_decay_lr(10, warmup=2, max_steps=10, stable_steps=3) == 0.0


def test_wsd_lr_respects_min_ratio() -> None:
    assert warmup_stable_decay_lr(0, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == 0.5
    assert warmup_stable_decay_lr(2, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == 1.0
    assert warmup_stable_decay_lr(7, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == pytest.approx(0.68)
    assert warmup_stable_decay_lr(10, warmup=2, max_steps=10, stable_steps=3, min_ratio=0.2) == 0.2
