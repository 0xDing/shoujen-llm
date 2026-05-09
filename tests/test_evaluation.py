import json
import math
from types import SimpleNamespace

import pytest
import torch

from shoujen.evaluation import bpb_from_sums, compute_lm_eval_metrics, evaluate_core


def test_compute_lm_eval_metrics_reports_bpb_over_non_special_bytes() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[1, 2]])
    token_bytes = torch.tensor([0, 1, 3])

    loss, active_tokens, bpb_nats, total_bytes = compute_lm_eval_metrics(
        logits,
        labels,
        token_bytes=token_bytes,
    )

    assert loss.item() == pytest.approx(math.log(3.0))
    assert active_tokens.item() == pytest.approx(2.0)
    assert bpb_nats.item() == pytest.approx(2.0 * math.log(3.0))
    assert total_bytes.item() == pytest.approx(4.0)
    assert bpb_from_sums(bpb_nats.item(), total_bytes.item()) == pytest.approx(
        math.log(3.0) / (2.0 * math.log(2.0))
    )


def test_compute_lm_eval_metrics_excludes_zero_byte_targets_from_bpb() -> None:
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[0, 2]])
    token_bytes = torch.tensor([0, 1, 3])

    _, _, bpb_nats, total_bytes = compute_lm_eval_metrics(
        logits,
        labels,
        token_bytes=token_bytes,
    )

    assert bpb_nats.item() == pytest.approx(math.log(3.0))
    assert total_bytes.item() == pytest.approx(3.0)


class TinyTokenizer:
    pad_id = 0

    def __init__(self) -> None:
        chars = ["<pad>", "q", " ", "x", "y"]
        self.ids = {ch: idx for idx, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.ids[ch] for ch in text]


class AlwaysXModel(torch.nn.Module):
    config = SimpleNamespace(max_seq_len=128)

    def __init__(self, x_id: int, vocab_size: int) -> None:
        super().__init__()
        self.x_id = x_id
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        logits = torch.zeros(*input_ids.shape, self.vocab_size, device=input_ids.device)
        logits[..., self.x_id] = 5.0
        return SimpleNamespace(logits=logits)


def test_evaluate_core_reads_nanochat_style_bundle(tmp_path) -> None:
    eval_data = tmp_path / "eval_data"
    eval_data.mkdir()
    (tmp_path / "core.yaml").write_text(
        "\n".join(
            [
                "icl_tasks:",
                "  - label: toy_mc",
                "    icl_task_type: multiple_choice",
                "    dataset_uri: toy.jsonl",
                "    num_fewshot: [0]",
                "    continuation_delimiter: ' '",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval_meta_data.csv").write_text(
        "Eval Task,Random baseline\n" "toy_mc,50\n",
        encoding="utf-8",
    )
    (eval_data / "toy.jsonl").write_text(
        json.dumps({"query": "q", "choices": ["x", "y"], "gold": 0}) + "\n",
        encoding="utf-8",
    )

    tokenizer = TinyTokenizer()
    model = AlwaysXModel(x_id=tokenizer.ids["x"], vocab_size=len(tokenizer.ids))

    metrics = evaluate_core(
        model,
        tokenizer,
        eval_dir=tmp_path,
        device=torch.device("cpu"),
        max_per_task=-1,
    )

    assert metrics.results["toy_mc"] == pytest.approx(1.0)
    assert metrics.centered_results["toy_mc"] == pytest.approx(1.0)
    assert metrics.score == pytest.approx(1.0)
