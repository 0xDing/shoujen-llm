from __future__ import annotations

import types

import torch

from scripts.train import build_config
from shoujen.config import ShoujenConfig
from shoujen.model import ShoujenForCausalLM
from shoujen.optim.muon import _is_embedding_param


def _ple_config() -> ShoujenConfig:
    return ShoujenConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        layer_pattern=("attention",),
        pattern_repeats=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=4,
        intermediate_size=32,
        hidden_size_per_layer_input=3,
        attnres_block_size=2,
        max_seq_len=16,
        max_position_embeddings=16,
        use_cache=True,
    )


def test_per_layer_embedding_table_is_packed_like_gemma4() -> None:
    cfg = _ple_config()
    model = ShoujenForCausalLM(cfg).eval()

    per_layer_embeddings = model.get_per_layer_input_embeddings()

    assert per_layer_embeddings is not None
    assert per_layer_embeddings.num_embeddings == cfg.vocab_size
    assert per_layer_embeddings.embedding_dim == cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
    assert model.config.vocab_size_per_layer_input == cfg.vocab_size


def test_inputs_embeds_with_input_ids_uses_same_ple_as_input_ids() -> None:
    torch.manual_seed(123)
    cfg = _ple_config()
    model = ShoujenForCausalLM(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        logits_from_ids = model(input_ids=input_ids, use_cache=False).logits
        logits_from_both = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        ).logits

    torch.testing.assert_close(logits_from_both, logits_from_ids, atol=0.0, rtol=0.0)


def test_inputs_embeds_only_recovers_exact_token_embeddings_for_ple() -> None:
    torch.manual_seed(456)
    cfg = _ple_config()
    model = ShoujenForCausalLM(cfg).eval()
    input_ids = torch.tensor([[5, 6, 7]])

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        logits_from_ids = model(input_ids=input_ids, use_cache=False).logits
        logits_from_embeds = model(inputs_embeds=inputs_embeds, use_cache=False).logits

    torch.testing.assert_close(logits_from_embeds, logits_from_ids, atol=0.0, rtol=0.0)


def test_inputs_embeds_only_rejects_non_token_embeddings_without_precomputed_ple() -> None:
    torch.manual_seed(789)
    cfg = _ple_config()
    model = ShoujenForCausalLM(cfg).eval()
    input_ids = torch.tensor([[8, 9]])
    inputs_embeds = model.model.embed_tokens(input_ids).detach().clone()
    inputs_embeds[:, :, 0] += 1.0

    try:
        model(inputs_embeds=inputs_embeds, use_cache=False)
    except RuntimeError as exc:
        assert "do not exactly match the embedding weights" in str(exc)
    else:
        raise AssertionError("Expected inputs_embeds-only soft embeddings to require exact token embeddings")


def test_resize_token_embeddings_also_resizes_per_layer_embeddings() -> None:
    torch.manual_seed(111)
    cfg = _ple_config()
    model = ShoujenForCausalLM(cfg).eval()

    model.resize_token_embeddings(40)

    assert model.get_input_embeddings().num_embeddings == 40
    assert model.get_per_layer_input_embeddings().num_embeddings == 40
    assert model.config.vocab_size_per_layer_input == 40
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[39]]), use_cache=False)
    assert out.logits.shape[-1] == 40


def test_cache_generation_matches_full_forward_with_per_layer_embeddings() -> None:
    torch.manual_seed(321)
    cfg = _ple_config()
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


def test_train_build_config_syncs_ple_vocab_to_tokenizer_size() -> None:
    tokenizer = types.SimpleNamespace(
        vocab_size=123,
        model_pad_id=0,
        eos_id=1,
        im_start_id=2,
        im_end_id=3,
        think_start_id=4,
        think_end_id=5,
    )
    args = types.SimpleNamespace(config=None, block_size=64)

    cfg = build_config(args, tokenizer)

    assert cfg.vocab_size == 123
    assert cfg.vocab_size_per_layer_input == 123
    assert cfg.think_start_token_id == 4
    assert cfg.think_end_token_id == 5


def test_optimizer_treats_per_layer_embedding_as_embedding_param() -> None:
    assert _is_embedding_param("model.embed_tokens_per_layer.weight")
