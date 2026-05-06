"""Token-by-token generation for ShoujenLM with KV cache + RWKV recurrent state."""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn.functional as F

from shoujen.model import ShoujenLM
from shoujen.tokenizer import ShoujenTokenizer


def _filter_logits(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    """Apply optional top-k / top-p truncation. logits: (V,)."""
    if top_k > 0:
        thresh = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True  # always keep top
        masked = torch.full_like(logits, float("-inf"))
        masked[sorted_idx[keep]] = sorted_logits[keep]
        logits = masked
    return logits


@torch.no_grad()
def generate(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_ids: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> Iterator[int]:
    model.eval()
    device = device or next(model.parameters()).device
    if stop_ids is None:
        stop_ids = (tokenizer.eos_id, tokenizer.im_end_id)

    # Prefill
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(ids)
    attn_caches = out.attn_caches
    rwkv_states = out.rwkv_states
    v_first = out.v_first
    next_logits = out.logits[:, -1, :]

    for _ in range(max_new_tokens):
        if temperature <= 0:
            tok = int(next_logits.argmax(dim=-1).item())
        else:
            l = next_logits[0] / max(temperature, 1e-5)
            l = _filter_logits(l, top_k=top_k, top_p=top_p)
            probs = torch.softmax(l, dim=-1)
            tok = int(torch.multinomial(probs, 1).item())

        yield tok

        if tok in stop_ids:
            return

        ids = torch.tensor([[tok]], dtype=torch.long, device=device)
        out = model(
            ids, attn_caches=attn_caches, rwkv_states=rwkv_states, v_first=v_first
        )
        attn_caches = out.attn_caches
        rwkv_states = out.rwkv_states
        v_first = out.v_first
        next_logits = out.logits[:, -1, :]


def chat(
    model: ShoujenLM,
    tokenizer: ShoujenTokenizer,
    messages: list[dict],
    **kwargs,
) -> str:
    """One-shot helper: encode a chat with `add_generation_prompt=True` and
    decode the streamed completion until <im_end> or <eos>.
    """
    prompt_ids, _ = tokenizer.encode_chat(messages, add_generation_prompt=True)
    pieces: list[int] = []
    for tok in generate(model, tokenizer, prompt_ids, **kwargs):
        pieces.append(tok)
    return tokenizer.decode(pieces, skip_special=True)
