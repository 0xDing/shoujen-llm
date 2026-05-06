from __future__ import annotations

import types

import torch
import torch.nn as nn

from shoujen.config import ShoujenConfig
from shoujen.model import (
    ShoujenBlockAttentionResidual,
    ShoujenForCausalLM,
    ShoujenModel,
)


def _tiny_config(num_layers: int = 4, hidden_size: int = 16) -> ShoujenConfig:
    head_dim = 4 if hidden_size % 4 == 0 and hidden_size >= 8 else 2
    num_attention_heads = hidden_size // head_dim
    num_kv_heads = 2 if num_attention_heads % 2 == 0 and num_attention_heads > 2 else 1
    return ShoujenConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        layer_pattern=("attention",),
        pattern_repeats=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=hidden_size * 2,
        attnres_block_size=2,
        max_seq_len=16,
        max_position_embeddings=16,
        use_cache=True,
    )


def test_block_attention_residual_matches_rmsnorm_einsum_reference() -> None:
    torch.manual_seed(123)
    cfg = _tiny_config(num_layers=1, hidden_size=6)
    mod = ShoujenBlockAttentionResidual(cfg)
    with torch.no_grad():
        mod.query.copy_(torch.randn(cfg.hidden_size))
        mod.norm.weight.copy_(torch.rand(cfg.hidden_size) + 0.5)

    blocks = [torch.randn(2, 3, cfg.hidden_size), torch.randn(2, 3, cfg.hidden_size)]
    partial = torch.randn(2, 3, cfg.hidden_size)

    out = mod(blocks, partial)

    stacked = torch.stack([*blocks, partial], dim=0)
    keys = mod.norm(stacked)
    logits = torch.einsum("...d,d->...", keys, mod.query)
    weights = torch.softmax(logits.float(), dim=0).to(stacked.dtype)
    ref = (weights[..., None] * stacked).sum(dim=0)

    torch.testing.assert_close(out, ref)


def test_block_attention_residual_returns_single_source_without_extra_work() -> None:
    cfg = _tiny_config(num_layers=1)
    mod = ShoujenBlockAttentionResidual(cfg)
    partial = torch.randn(2, 3, cfg.hidden_size)

    out = mod([], partial)

    assert out.data_ptr() == partial.data_ptr()


def test_model_keeps_token_embedding_as_independent_block_source() -> None:
    cfg = _tiny_config(num_layers=3, hidden_size=4)
    model = ShoujenModel(cfg).eval()
    records: list[dict[str, object]] = []

    class Recorder(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.name = name

        def forward(self, blocks: torch.Tensor, partial_block: torch.Tensor | None) -> torch.Tensor:
            records.append(
                {
                    "name": self.name,
                    "blocks": [block.detach().clone() for block in blocks.unbind(dim=0)],
                    "partial": None if partial_block is None else partial_block.detach().clone(),
                }
            )
            return blocks[0] if partial_block is None else partial_block

    def make_token_mixer(layer_idx: int):
        def apply_token_mixer(self, hidden_states: torch.Tensor, **kwargs):
            return (
                torch.full_like(hidden_states, float(layer_idx + 1)),
                kwargs.get("attn_cache"),
                kwargs.get("rwkv_state"),
                kwargs.get("v_first"),
                None,
            )

        return apply_token_mixer

    def make_mlp(layer_idx: int):
        def apply_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.full_like(hidden_states, float((layer_idx + 1) * 10))

        return apply_mlp

    for layer_idx, layer in enumerate(model.layers):
        layer.attn_res = Recorder(f"layer{layer_idx}.attn")
        layer.mlp_res = Recorder(f"layer{layer_idx}.mlp")
        layer.apply_token_mixer = types.MethodType(make_token_mixer(layer_idx), layer)
        layer.apply_mlp = types.MethodType(make_mlp(layer_idx), layer)
    model.final_attn_res = Recorder("final")

    inputs_embeds = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    with torch.no_grad():
        model(inputs_embeds=inputs_embeds, use_cache=False)

    by_name = {record["name"]: record for record in records}
    layer0_attn = by_name["layer0.attn"]
    layer2_attn = by_name["layer2.attn"]
    final = by_name["final"]

    assert layer0_attn["partial"] is None
    torch.testing.assert_close(layer0_attn["blocks"][0], inputs_embeds)

    assert layer2_attn["partial"] is None
    assert len(layer2_attn["blocks"]) == 2
    torch.testing.assert_close(layer2_attn["blocks"][0], inputs_embeds)
    torch.testing.assert_close(layer2_attn["blocks"][1], torch.full_like(inputs_embeds, 33.0))

    assert len(final["blocks"]) == 2
    torch.testing.assert_close(final["blocks"][0], inputs_embeds)
    torch.testing.assert_close(final["partial"], torch.full_like(inputs_embeds, 33.0))


def test_cache_generation_matches_full_forward_with_attention_residual_blocks() -> None:
    torch.manual_seed(321)
    cfg = _tiny_config(num_layers=4)
    model = ShoujenForCausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

    with torch.no_grad():
        full = model(input_ids, use_cache=False).logits
        cache = None
        pieces = []
        for token_idx in range(input_ids.shape[1]):
            out = model(
                input_ids[:, token_idx : token_idx + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            pieces.append(out.logits)

    incremental = torch.cat(pieces, dim=1)

    torch.testing.assert_close(incremental, full, atol=1e-6, rtol=1e-6)
    assert cache is not None
    assert cache.get_seq_length() == input_ids.shape[1]
