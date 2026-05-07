from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig


class ShoujenConfig(PretrainedConfig):
    """Configuration for the text-only Shoujen causal LM.

    The public fields intentionally keep the names from the project README and
    the earlier toy implementation, while the class itself follows the
    Transformers `PretrainedConfig` contract.
    """

    model_type = "shoujen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_hidden_layers: int = 24,
        layer_pattern: tuple[str, ...] | list[str] = ("rwkv7", "rwkv7", "rwkv7", "attention"),
        pattern_repeats: int = 6,
        layer_types: list[str] | tuple[str, ...] | None = None,
        shoujen_layer_types: list[str] | tuple[str, ...] | None = None,
        num_attention_heads: int = 10,
        num_kv_heads: int = 2,
        num_key_value_heads: int | None = None,
        head_dim: int = 64,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        attention_window: int | None = None,
        qk_norm: bool = False,
        qk_norm_eps: float | None = None,
        intermediate_size: int = 2048,
        hidden_act: str = "silu",
        hidden_size_per_layer_input: int = 256,
        vocab_size_per_layer_input: int | None = None,
        attnres_n_blocks: int = 4,
        attnres_block_size: int | None = None,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        max_seq_len: int = 2048,
        max_position_embeddings: int | None = None,
        pad_token_id: int | None = 0,
        eos_token_id: int = 1,
        im_start_token_id: int = 2,
        im_end_token_id: int = 3,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        rwkv_decay_lora: int | None = None,
        rwkv_aaa_lora: int | None = None,
        rwkv_v_first_lora: int | None = None,
        rwkv_gate_lora: int | None = None,
        **kwargs: Any,
    ):
        if num_key_value_heads is not None:
            num_kv_heads = num_key_value_heads
        if max_position_embeddings is None:
            max_position_embeddings = max_seq_len
        if vocab_size_per_layer_input is None:
            vocab_size_per_layer_input = vocab_size
        if attnres_block_size is None:
            attnres_block_size = attnres_n_blocks

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_pattern = tuple(layer_pattern)
        self.pattern_repeats = pattern_repeats

        if shoujen_layer_types is not None:
            derived_layer_types = list(shoujen_layer_types)
        elif layer_types is not None and set(layer_types).issubset({"rwkv7", "attention"}):
            derived_layer_types = list(layer_types)
        elif layer_types is None:
            derived_layer_types = list(self.layer_pattern) * self.pattern_repeats
        else:
            derived_layer_types = list(self.layer_pattern) * self.pattern_repeats
        self.shoujen_layer_types = derived_layer_types
        # Transformers 5 validates `layer_types` against a shared vocabulary.
        # Keep the real Shoujen layout in `shoujen_layer_types`, and expose a
        # validator-compatible view for generic Transformers utilities.
        self.layer_types = [
            "linear_attention" if layer_type == "rwkv7" else "full_attention"
            for layer_type in derived_layer_types
        ]

        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.attention_window = attention_window
        self.qk_norm = qk_norm
        self.qk_norm_eps = rms_norm_eps if qk_norm_eps is None else qk_norm_eps

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input

        self.attnres_n_blocks = attnres_n_blocks
        self.attnres_block_size = attnres_block_size

        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_position_embeddings
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.use_cache = use_cache

        self.rwkv_decay_lora = rwkv_decay_lora
        self.rwkv_aaa_lora = rwkv_aaa_lora
        self.rwkv_v_first_lora = rwkv_v_first_lora
        self.rwkv_gate_lora = rwkv_gate_lora

        self._validate()

    def _validate(self) -> None:
        if len(self.shoujen_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"shoujen_layer_types length {len(self.shoujen_layer_types)} != "
                f"num_hidden_layers {self.num_hidden_layers}"
            )
        valid = {"rwkv7", "attention"}
        unknown = sorted({t for t in self.shoujen_layer_types if t not in valid})
        if unknown:
            raise ValueError(f"Unknown layer type(s): {unknown}")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.hidden_size % self.head_dim != 0:
            raise ValueError(
                "hidden_size must be divisible by head_dim for RWKV7 head split, "
                f"got hidden_size={self.hidden_size} head_dim={self.head_dim}"
            )
        if self.attnres_block_size <= 0:
            raise ValueError("attnres_block_size must be positive")
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention_window must be positive when set")
        if self.hidden_size_per_layer_input < 0:
            raise ValueError("hidden_size_per_layer_input must be non-negative")
        if self.hidden_size_per_layer_input and self.vocab_size_per_layer_input <= 0:
            raise ValueError("vocab_size_per_layer_input must be positive when PLE is enabled")

    @property
    def num_rwkv_heads(self) -> int:
        return self.hidden_size // self.head_dim

    def is_attention_layer(self, idx: int) -> bool:
        return self.shoujen_layer_types[idx] == "attention"

    def attention_layer_index(self, idx: int) -> int:
        if not self.is_attention_layer(idx):
            return -1
        return sum(1 for t in self.shoujen_layer_types[:idx] if t == "attention")

    def get_text_config(self, *args: Any, **kwargs: Any) -> "ShoujenConfig":
        return self

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ShoujenConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
