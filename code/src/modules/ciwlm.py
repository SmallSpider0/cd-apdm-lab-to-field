"""Category Importance Weighted Loss Module (CIWLM).

Combines the class-balanced effective-sample weighting
(Cui et al., 2019) with a focal modulator (Lin et al., 2017):

    n_eff_c = (1 - β_e^n_c) / (1 - β_e)
    α_c     = (1 / n_eff_c) normalized so Σ α_c = C
    L_i     = α_{y_i} * (1 - p_{y_i})^γ * (-log p_{y_i})

Optional MDIWM instance weighting (``w_i``) is folded in multiplicatively
so the loss matches the paper's training objective
``L_total = Σ w_i α_{y_i} (1 - p_{y_i})^γ log p_{y_i}``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_class_weights(class_counts: Sequence[int], beta_e: float = 0.999) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.where(counts <= 0, 1.0, counts)
    n_eff = (1.0 - np.power(beta_e, counts)) / (1.0 - beta_e)
    w = 1.0 / n_eff
    w = w * len(counts) / w.sum()
    return torch.from_numpy(w).float()


class CIWLMLoss(nn.Module):
    def __init__(self, class_counts: Sequence[int], beta_e: float = 0.999, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.register_buffer("alpha_c", effective_class_weights(class_counts, beta_e=beta_e))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        instance_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        gather = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - p_t).clamp_min(1e-12) ** self.focal_gamma
        alpha_t = self.alpha_c.to(logits.device)[targets]
        loss = -alpha_t * focal * gather
        if instance_weights is not None:
            loss = loss * instance_weights.to(loss.device)
        return loss.mean()
