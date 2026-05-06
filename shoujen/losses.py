"""Losses for Shoujen-LM training.

- compute_lm_loss: standard next-token cross-entropy with optional per-token mask
- SemanticTubePredictionLoss: STP auxiliary loss for future SFT training

STP follows "Semantic Tube Prediction: Beating LLM Data Efficiency
with JEPA": for random indices s < r < t in one continuous semantic span,
L_STP = 1 - cos(h_t - h_r, h_r - h_s). It is intentionally not wired into
packed pretraining, where random triples can cross unrelated documents.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM loss.

    `logits`: (B, T, V) at positions 0..T-1.
    `labels`: (B, T) — already shifted to token-at-(t+1) targets, with -100 for
              ignored positions (e.g. padding). The caller is expected to do the
              shift, so logits[:, t] is compared to labels[:, t].
    `loss_mask`: optional (B, T) float in {0,1} multiplied with the per-token
              loss before reduction. SFT uses an assistant-only mask here.

    Returns (mean_loss, num_active_tokens).
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_labels = labels.reshape(B * T)
    per_token = F.cross_entropy(
        flat_logits, flat_labels, reduction="none", ignore_index=ignore_index
    ).view(B, T)

    keep = (flat_labels != ignore_index).view(B, T).float()
    if loss_mask is not None:
        keep = keep * loss_mask.float()

    denom = keep.sum().clamp(min=1.0)
    loss = (per_token * keep).sum() / denom
    return loss, denom


class SemanticTubePredictionLoss(nn.Module):
    def __init__(
        self,
        samples_per_sequence: int = 1,
        max_width: int | None = None,
    ):
        super().__init__()
        if samples_per_sequence < 1:
            raise ValueError("samples_per_sequence must be >= 1")
        if max_width is not None and max_width < 2:
            raise ValueError("max_width must be >= 2 when set")
        self.samples_per_sequence = samples_per_sequence
        self.max_width = max_width

    def forward(
        self,
        hidden: torch.Tensor,
        spans: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Compute STP over continuous spans.

        `hidden`: (B, T, D), usually the final layer hidden states.
        `spans`: optional (B, 2) integer tensor of [start, end) bounds in
            hidden-state positions. Future SFT code should pass assistant or
            natural-content spans here. If omitted, the full sequence is used.

        Returns scalar loss (zero if no active positions).
        """
        B, T, D = hidden.shape
        del D

        if spans is None:
            spans = torch.tensor([[0, T]], device=hidden.device).expand(B, 2)
        elif spans.shape != (B, 2):
            raise ValueError(f"spans must have shape {(B, 2)}, got {tuple(spans.shape)}")

        losses: list[torch.Tensor] = []
        for i in range(B):
            start = max(0, min(T, int(spans[i, 0].item())))
            end = max(start, min(T, int(spans[i, 1].item())))
            if end - start < 3:
                continue

            for _ in range(self.samples_per_sequence):
                max_end = end
                if self.max_width is not None:
                    # Sample s first, then cap t so that t - s <= max_width.
                    latest_s = max(start + 1, end - 2)
                    s = _randint(start, latest_s, generator=generator)
                    max_end = min(end, s + self.max_width + 1)
                    if max_end - s < 3:
                        continue
                else:
                    s = _randint(start, end - 2, generator=generator)

                r = _randint(s + 1, max_end - 1, generator=generator)
                t = _randint(r + 1, max_end, generator=generator)

                future = hidden[i, t] - hidden[i, r]
                past = hidden[i, r] - hidden[i, s]
                losses.append(
                    1.0 - F.cosine_similarity(future.float(), past.float(), dim=0)
                )

        if not losses:
            return hidden.new_zeros(())
        return torch.stack(losses).mean()


def _randint(
    low: int,
    high: int,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Return an int sampled from [low, high)."""
    if high <= low:
        raise ValueError(f"empty randint range [{low}, {high})")
    return int(torch.randint(low, high, (), generator=generator).item())
