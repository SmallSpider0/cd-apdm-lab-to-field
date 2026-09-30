"""Adaptive Confidence Rectification Module (ACRM).

For each class c we maintain EMA estimates of:
    agree_c - **student-teacher agreement rate** on the target stream,
    h_c     - average predictive entropy.

NOTE on naming (R1-2): this quantity was previously called `a_c` and documented
as "empirical pseudo-label accuracy". It is not accuracy. It is computed as
`1[student_argmax == teacher_label]`, i.e. how often the student agrees with the
teacher — a self-consistency measure. When student and teacher are confidently
wrong together, this approaches 1 while true accuracy is low, which is exactly
the confirmation-bias failure mode. The name has been corrected so the code
cannot mislead a reader into thinking a true accuracy is being tracked.
The calibration bound built on this quantity was withdrawn in the revision
(Supplementary Section S8).

Adaptive per-class threshold (deviations from the class means; the paper's
Section 3.4.3 writes ratios, see Supplementary Section S8). Used only to select
pseudo-labels, which the reported configuration does not do:
    τ_c = τ_base + λ_1 * (agree_c - mean(agree)) - λ_2 * (h_c - mean(h))

Confidence re-weighting with tail boosting:
    p̃_c = p_c * (1 + γ_boost * (n_max / n_c - 1) * 1[c is tail])

`tail` is derived from training class counts (count < tail_threshold).

In the reported configuration only the re-weighting is used, once, at final
inference (renormalised). The thresholds are used only by the pseudo-label
pipeline, which is disabled (paper, Section 3.4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class ACRMConfig:
    eta: float = 0.95          # EMA momentum
    tau_base: float = 0.7
    lambda_acc: float = 0.15
    lambda_ent: float = 0.10
    gamma_boost: float = 0.08
    tail_threshold: int = 100


class ACRM:
    def __init__(self, num_classes: int, class_counts: Sequence[int], cfg: ACRMConfig, device: torch.device):
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device
        counts = torch.tensor(class_counts, dtype=torch.float32, device=device).clamp_min(1.0)
        self.n_c = counts
        self.n_max = counts.max()
        self.tail_mask = (counts < cfg.tail_threshold).float()

        # initialise EMA buffers
        self.agree_c = torch.full((num_classes,), 0.5, device=device)
        self.h_c = torch.full((num_classes,), math.log(num_classes) / 2, device=device)
        self._initialized = False

    # ---------------------------------------------------------- EMA updates
    @torch.no_grad()
    def update_stats(
        self,
        pseudo_labels: torch.Tensor,
        teacher_labels: Optional[torch.Tensor],
        probs: torch.Tensor,
    ) -> None:
        """Update agree_c, h_c after each batch.

        - pseudo_labels: argmax over student probs
        - teacher_labels: refined labels from teacher (reference for the agreement rate);
          fall back to pseudo_labels if no teacher is supplied.
        - probs: student softmax probabilities  (B, C)
        """
        ref = teacher_labels if teacher_labels is not None else pseudo_labels
        # 这是一致率，不是准确率：ref 是教师的伪标签，不是真实标签。
        # 变量名刻意叫 agreement，避免再被误读为 accuracy（R1-2）。
        agreement = (pseudo_labels == ref).float()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

        for c in pseudo_labels.unique():
            mask = pseudo_labels == c
            if mask.any():
                agree_obs = agreement[mask].mean()
                h_obs = entropy[mask].mean()
                self.agree_c[c] = self.cfg.eta * self.agree_c[c] + (1 - self.cfg.eta) * agree_obs
                self.h_c[c] = self.cfg.eta * self.h_c[c] + (1 - self.cfg.eta) * h_obs
        self._initialized = True

    # ---------------------------------------------------------- threshold
    @torch.no_grad()
    def thresholds(self) -> torch.Tensor:
        """Per-class adaptive threshold τ_c."""
        agree_mean = self.agree_c.mean()
        h_mean = self.h_c.mean()
        return (
            self.cfg.tau_base
            + self.cfg.lambda_acc * (self.agree_c - agree_mean)
            - self.cfg.lambda_ent * (self.h_c - h_mean)
        ).clamp(0.05, 0.99)

    # ---------------------------------------------------------- rectify
    @torch.no_grad()
    def rectify(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply class-specific tail boost then renormalise."""
        ratio = self.n_max / self.n_c
        boost = 1.0 + self.cfg.gamma_boost * (ratio - 1.0) * self.tail_mask
        new = probs * boost.unsqueeze(0)
        return new / new.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, logits: torch.Tensor):
        """Return (rectified_probs, accepted_mask, predicted_labels)."""
        probs = F.softmax(logits, dim=-1)
        probs = self.rectify(probs)
        conf, pred = probs.max(dim=-1)
        tau = self.thresholds()
        tau_per_sample = tau.to(logits.device)[pred]
        accept = conf >= tau_per_sample
        return probs, accept, pred


