from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


class MultipleOptimizer(Optimizer):
    """Expose multiple optimizers through the single-optimizer API Trainer expects."""

    def __init__(self, optimizers: Iterable[Optimizer]):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("MultipleOptimizer requires at least one optimizer")

        params = []
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                params.extend(group["params"])

        super().__init__(params, defaults={})
        self._sync_from_inner_optimizers()

    def _sync_from_inner_optimizers(self) -> None:
        self.param_groups = []
        self.state = defaultdict(dict)
        for optimizer_idx, optimizer in enumerate(self.optimizers):
            for group in optimizer.param_groups:
                group["_optimizer_idx"] = optimizer_idx
                self.param_groups.append(group)
            self.state.update(optimizer.state)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        for optimizer in self.optimizers:
            current = optimizer.step(closure=closure)
            if current is not None:
                loss = current
        return loss

    def state_dict(self):  # type: ignore[override]
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict):  # type: ignore[override]
        states = state_dict.get("optimizers")
        if states is None:
            raise ValueError("MultipleOptimizer state_dict must contain an 'optimizers' key")
        if len(states) != len(self.optimizers):
            raise ValueError(
                f"Expected {len(self.optimizers)} optimizer states, got {len(states)}"
            )
        for optimizer, state in zip(self.optimizers, states):
            optimizer.load_state_dict(state)
        self._sync_from_inner_optimizers()
