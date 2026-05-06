"""Transformers-compatible Shoujen text causal LM.

The implementation follows the Qwen3.5 text-model shape: a `PreTrainedModel`
backbone plus a causal-LM wrapper. The local architectural changes are:

* RWKV7 x070 layers in place of Qwen3.5's linear-attention layers.
* Moonshot Block Attention Residuals over depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput

from shoujen.config import ShoujenConfig
from shoujen.modules.norm import RMSNorm
from shoujen.modules.rwkv7 import RWKV7TimeMix


@dataclass
class ShoujenHybridCache:
    attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None]
    rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None]
    v_first: torch.Tensor | None = None
    seq_length: int = 0

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None:
            cache = self.attn_caches[layer_idx]
            if cache is not None:
                return cache[0].shape[-2]
        if self.seq_length:
            return self.seq_length
        for cache in self.attn_caches:
            if cache is not None:
                return cache[0].shape[-2]
        return 0


@dataclass
class ShoujenBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: ShoujenHybridCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None] | None = None
    v_first: torch.Tensor | None = None


@dataclass
class ShoujenCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: ShoujenHybridCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None] | None = None
    v_first: torch.Tensor | None = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ShoujenRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: ShoujenConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float()
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.to(x.device) @ position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ShoujenMLP(nn.Module):
    def __init__(self, config: ShoujenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ShoujenAttention(nn.Module):
    def __init__(self, config: ShoujenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.qk_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.qk_norm_eps) if config.qk_norm else None
        self.track_max_qk_logit = False
        self.last_max_qk_logit: float | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0].to(key_states), key_states], dim=-2)
            value_states = torch.cat([past_key_value[1].to(value_states), value_states], dim=-2)
        present_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = None
        self.last_max_qk_logit = None
        if hidden_states.device.type == "mps" or output_attentions or self.track_max_qk_logit:
            scores = (query_states * self.scaling) @ key_states.transpose(-2, -1)
            if self.track_max_qk_logit:
                self.last_max_qk_logit = float(scores.detach().float().amax().item())
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = torch.softmax(scores.float(), dim=-1).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = attn_weights @ value_states
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value, attn_weights


class ShoujenBlockAttentionResidual(nn.Module):
    """Block AttnRes over completed block states plus current partial state."""

    def __init__(self, config: ShoujenConfig):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _logits(self, values: torch.Tensor) -> torch.Tensor:
        x = values.float()
        inv_rms = torch.rsqrt(x.square().mean(-1) + self.norm.eps)
        projection = self.norm.weight.float() * self.query.float()
        return torch.matmul(x, projection) * inv_rms

    def forward(self, blocks: torch.Tensor | list[torch.Tensor], partial_block: torch.Tensor | None) -> torch.Tensor:
        if isinstance(blocks, list):
            completed_blocks = torch.stack(blocks, dim=0) if blocks else None
        else:
            completed_blocks = blocks

        n_completed = 0 if completed_blocks is None else completed_blocks.shape[0]
        if n_completed == 0 and partial_block is None:
            raise RuntimeError("Block AttnRes needs at least one residual state")
        if n_completed == 0:
            return partial_block
        if n_completed == 1 and partial_block is None:
            return completed_blocks[0]

        if partial_block is None:
            values = completed_blocks
        else:
            values = torch.cat((completed_blocks, partial_block.unsqueeze(0)), dim=0)

        _, batch_size, seq_len, hidden_size = values.shape
        # MPS is much faster when the tiny source dimension is reduced via bmm
        # over flattened tokens instead of softmax/sum over dim 0.
        values_by_token = values.permute(1, 2, 0, 3).reshape(batch_size * seq_len, -1, hidden_size)
        logits = self._logits(values_by_token)
        weights = torch.softmax(logits.float(), dim=-1).to(values_by_token.dtype)
        output = torch.bmm(weights.unsqueeze(1), values_by_token).squeeze(1)
        return output.reshape(batch_size, seq_len, hidden_size)


class ShoujenDecoderLayer(nn.Module):
    def __init__(self, config: ShoujenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_attention = config.is_attention_layer(layer_idx)

        self.attn_res = ShoujenBlockAttentionResidual(config)
        self.mlp_res = ShoujenBlockAttentionResidual(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.is_attention:
            self.token_mixer = ShoujenAttention(config, layer_idx)
        else:
            self.token_mixer = RWKV7TimeMix(config, layer_idx)

        self.mlp = ShoujenMLP(config)

    def apply_token_mixer(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        current_token_mask: torch.Tensor | None,
        sequence_start_mask: torch.Tensor | None,
        attn_cache: tuple[torch.Tensor, torch.Tensor] | None,
        rwkv_state: tuple[torch.Tensor | None, torch.Tensor | None] | None,
        v_first: torch.Tensor | None,
        output_attentions: bool,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor | None, torch.Tensor | None] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        hidden_states = self.input_layernorm(hidden_states)
        if current_token_mask is not None:
            hidden_states = hidden_states * current_token_mask[:, :, None].to(hidden_states.dtype)

        if self.is_attention:
            out, new_attn_cache, attn_weights = self.token_mixer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=attn_cache,
                output_attentions=output_attentions,
            )
            return out, new_attn_cache, rwkv_state, v_first, attn_weights

        out, new_v_first, new_rwkv_state = self.token_mixer(
            hidden_states,
            v_first=v_first,
            state=rwkv_state,
            sequence_start_mask=sequence_start_mask,
        )
        return out, attn_cache, new_rwkv_state, new_v_first, None

    def apply_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp(hidden_states)


class ShoujenPreTrainedModel(PreTrainedModel):
    config_class = ShoujenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ShoujenDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "_skip_shoujen_init", False):
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


class ShoujenModel(ShoujenPreTrainedModel):
    def __init__(self, config: ShoujenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ShoujenDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = ShoujenRotaryEmbedding(config)
        self.final_attn_res = ShoujenBlockAttentionResidual(config)
        self.gradient_checkpointing = False
        self._causal_mask_cache: dict[tuple[str, torch.dtype, int], torch.Tensor] = {}

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def set_track_max_qk_logit(self, enabled: bool) -> None:
        for layer in self.layers:
            if layer.is_attention:
                layer.token_mixer.track_max_qk_logit = enabled

    def max_qk_logit(self) -> float | None:
        values = [
            layer.token_mixer.last_max_qk_logit
            for layer in self.layers
            if layer.is_attention and layer.token_mixer.last_max_qk_logit is not None
        ]
        return max(values) if values else None

    def _to_hybrid_cache(
        self,
        past_key_values: ShoujenHybridCache | None,
        attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None,
        rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None] | None,
        v_first: torch.Tensor | None,
    ) -> ShoujenHybridCache | None:
        if past_key_values is not None and isinstance(past_key_values, ShoujenHybridCache):
            return past_key_values
        if attn_caches is None and rwkv_states is None and v_first is None:
            return None
        attn_caches = attn_caches if attn_caches is not None else [None] * self.config.num_hidden_layers
        rwkv_states = rwkv_states if rwkv_states is not None else [None] * self.config.num_hidden_layers
        seq_length = 0
        for cache in attn_caches:
            if cache is not None:
                seq_length = cache[0].shape[-2]
                break
        return ShoujenHybridCache(attn_caches=attn_caches, rwkv_states=rwkv_states, v_first=v_first, seq_length=seq_length)

    def _prepare_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        past_seen_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        total_len = past_seen_tokens + seq_len
        min_dtype = -torch.finfo(dtype).max
        if attention_mask is not None and attention_mask.ndim in (3, 4):
            attention_mask = attention_mask.to(device=device)
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :]
            expected = (batch_size, 1, seq_len, total_len)
            if tuple(attention_mask.shape) != expected:
                raise ValueError(
                    f"packed attention_mask must have shape {expected}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            if attention_mask.dtype == torch.bool:
                packed_mask = torch.zeros(expected, dtype=dtype, device=device)
                return packed_mask.masked_fill(~attention_mask, min_dtype)
            return attention_mask.to(dtype=dtype)

        if attention_mask is None and past_seen_tokens == 0:
            cache_key = (str(device), dtype, seq_len)
            cached = self._causal_mask_cache.get(cache_key)
            if cached is None:
                row_ids = torch.arange(seq_len, device=device)[:, None]
                col_ids = torch.arange(seq_len, device=device)[None, :]
                allowed = col_ids <= row_ids
                causal_mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
                causal_mask = causal_mask.masked_fill(~allowed, min_dtype)
                cached = causal_mask[None, None, :, :]
                self._causal_mask_cache[cache_key] = cached
            return cached.expand(batch_size, 1, seq_len, seq_len)

        row_ids = torch.arange(seq_len, device=device)[:, None] + past_seen_tokens
        col_ids = torch.arange(total_len, device=device)[None, :]
        allowed = col_ids <= row_ids
        causal_mask = torch.zeros(seq_len, total_len, dtype=dtype, device=device)
        causal_mask = causal_mask.masked_fill(~allowed, min_dtype)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, total_len)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
            if attention_mask.shape[-1] == seq_len and past_seen_tokens:
                prefix = torch.ones(
                    attention_mask.shape[0],
                    past_seen_tokens,
                    dtype=attention_mask.dtype,
                    device=device,
                )
                attention_mask = torch.cat([prefix, attention_mask], dim=-1)
            if attention_mask.shape[-1] != total_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.shape[-1]} does not match "
                    f"past+current length {total_len}"
                )
            padding_mask = attention_mask[:, None, None, :].to(torch.bool)
            causal_mask = causal_mask.masked_fill(~padding_mask, min_dtype)
        return causal_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        sequence_start_mask: torch.Tensor | None = None,
        past_key_values: ShoujenHybridCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None] | None = None,
        v_first: torch.Tensor | None = None,
        **_: Any,
    ) -> ShoujenBaseModelOutputWithPast | tuple:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids, inputs_embeds, or both")

        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states or return_hidden_states)
        return_dict = True if return_dict is None else return_dict
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif input_ids is not None and input_ids.shape != inputs_embeds.shape[:-1]:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match inputs_embeds shape "
                f"{tuple(inputs_embeds.shape[:-1])} when both are provided"
            )

        cache = self._to_hybrid_cache(past_key_values, attn_caches, rwkv_states, v_first)
        past_seen_tokens = cache.get_seq_length() if cache is not None else 0
        batch_size, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        rwkv_sequence_start_mask = None
        if sequence_start_mask is None:
            sequence_start_mask = position_ids == 0
        else:
            sequence_start_mask = sequence_start_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            rwkv_sequence_start_mask = sequence_start_mask
        if tuple(sequence_start_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                f"sequence_start_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(sequence_start_mask.shape)}"
            )

        causal_mask = self._prepare_causal_mask(
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            past_seen_tokens=past_seen_tokens,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        current_token_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            current_token_mask = attention_mask[:, -seq_len:].to(inputs_embeds.device)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states_tuple: tuple[torch.Tensor, ...] = ()
        attentions: tuple[torch.Tensor, ...] = ()
        new_attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        new_rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None] = []

        old_attn_caches = cache.attn_caches if cache is not None else [None] * self.config.num_hidden_layers
        old_rwkv_states = cache.rwkv_states if cache is not None else [None] * self.config.num_hidden_layers
        v_first_state = cache.v_first if cache is not None else None

        # Moonshot Block AttnRes keeps token embeddings as the independent b0
        # source; completed depth blocks contain only accumulated layer outputs.
        completed_blocks = inputs_embeds.unsqueeze(0)
        partial_block: torch.Tensor | None = None
        hidden_states = inputs_embeds

        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx > 0 and layer_idx % self.config.attnres_block_size == 0:
                if partial_block is not None:
                    completed_blocks = torch.cat((completed_blocks, partial_block.unsqueeze(0)), dim=0)
                partial_block = None

            attnres_input = decoder_layer.attn_res(completed_blocks, partial_block)
            mixer_out, new_attn_cache, new_rwkv_state, v_first_state, attn_weights = decoder_layer.apply_token_mixer(
                attnres_input,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                current_token_mask=current_token_mask,
                sequence_start_mask=rwkv_sequence_start_mask,
                attn_cache=old_attn_caches[layer_idx],
                rwkv_state=old_rwkv_states[layer_idx],
                v_first=v_first_state,
                output_attentions=output_attentions,
            )
            partial_block = mixer_out if partial_block is None else partial_block + mixer_out

            if self.gradient_checkpointing and self.training and not output_attentions:
                def mlp_forward(
                    completed_blocks_arg: torch.Tensor,
                    partial_block_arg: torch.Tensor,
                    decoder_layer_arg: ShoujenDecoderLayer = decoder_layer,
                ) -> torch.Tensor:
                    mlp_input = decoder_layer_arg.mlp_res(completed_blocks_arg, partial_block_arg)
                    return partial_block_arg + decoder_layer_arg.apply_mlp(mlp_input)

                partial_block = self._gradient_checkpointing_func(
                    mlp_forward,
                    completed_blocks,
                    partial_block,
                )
            else:
                mlp_input = decoder_layer.mlp_res(completed_blocks, partial_block)
                partial_block = partial_block + decoder_layer.apply_mlp(mlp_input)

            hidden_states = partial_block
            new_attn_caches.append(new_attn_cache if use_cache else None)
            new_rwkv_states.append(new_rwkv_state if use_cache else None)
            if output_attentions and attn_weights is not None:
                attentions = attentions + (attn_weights,)
            if output_hidden_states:
                hidden_states_tuple = hidden_states_tuple + (hidden_states,)

        hidden_states = self.final_attn_res(completed_blocks, partial_block)
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            hidden_states_tuple = hidden_states_tuple + (hidden_states,)

        new_cache = None
        if use_cache:
            new_cache = ShoujenHybridCache(
                attn_caches=new_attn_caches,
                rwkv_states=new_rwkv_states,
                v_first=v_first_state,
                seq_length=past_seen_tokens + seq_len,
            )

        if not return_dict:
            return (hidden_states, new_cache, hidden_states_tuple, attentions)

        return ShoujenBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=new_cache,
            hidden_states=hidden_states_tuple if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
            attn_caches=new_cache.attn_caches if new_cache is not None else None,
            rwkv_states=new_cache.rwkv_states if new_cache is not None else None,
            v_first=new_cache.v_first if new_cache is not None else None,
        )


class ShoujenForCausalLM(ShoujenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: ShoujenConfig):
        super().__init__(config)
        self.model = ShoujenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def set_track_max_qk_logit(self, enabled: bool) -> None:
        self.model.set_track_max_qk_logit(enabled)

    def max_qk_logit(self) -> float | None:
        return self.model.max_qk_logit()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: ShoujenHybridCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(past_key_values, ShoujenHybridCache):
            past_key_values = None
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_length > 0:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
        model_inputs = {"input_ids": input_ids}
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache", True),
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        sequence_start_mask: torch.Tensor | None = None,
        past_key_values: ShoujenHybridCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        attn_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        rwkv_states: list[tuple[torch.Tensor | None, torch.Tensor | None] | None] | None = None,
        v_first: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ShoujenCausalLMOutputWithPast | tuple:
        return_dict = True if return_dict is None else return_dict
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sequence_start_mask=sequence_start_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_hidden_states=return_hidden_states,
            return_dict=True,
            attn_caches=attn_caches,
            rwkv_states=rwkv_states,
            v_first=v_first,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_for_logits = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden_for_logits = hidden_states[:, logits_to_keep, :]
        else:
            hidden_for_logits = hidden_states
        logits = self.lm_head(hidden_for_logits)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return ShoujenCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attn_caches=outputs.attn_caches,
            rwkv_states=outputs.rwkv_states,
            v_first=outputs.v_first,
        )

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        total = 0
        input_embeddings = self.get_input_embeddings()
        output_embeddings = self.get_output_embeddings()
        excluded = set()
        if exclude_embeddings:
            if input_embeddings is not None:
                excluded.add(id(input_embeddings.weight))
            if output_embeddings is not None:
                excluded.add(id(output_embeddings.weight))
        for p in self.parameters():
            if only_trainable and not p.requires_grad:
                continue
            if id(p) in excluded:
                continue
            total += p.numel()
        return total


ShoujenLM = ShoujenForCausalLM
