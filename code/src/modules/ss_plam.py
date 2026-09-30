"""Self-Supervised Pseudo-Label Adaptation Module (SS-PLAM).

Three components consistent with the paper:
1. Rotation-prediction pretext head (4-way classification) for cheap
   self-supervised pre-training on the target encoder.
2. SimCLR-style projection head + NT-Xent loss for contrastive
   pre-training.
3. Mean-Teacher EMA update of the classifier weights to generate
   pseudo-labels for the target domain, combined with logit adjustment
   τ * log π_y to compensate for class-prior bias.
"""
from __future__ import annotations

from typing import Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------- Pretext heads
class RotationHead(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 4)

    def forward(self, x):
        return self.fc(x)


class ProjectionHead(nn.Module):
    def __init__(self, feat_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def rotation_self_supervision(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns rotated batch + 4-way labels (0,90,180,270)."""
    B = image.size(0)
    k = torch.randint(0, 4, (B,), device=image.device)
    out = torch.stack([torch.rot90(image[i], k=int(k[i]), dims=(-2, -1)) for i in range(B)])
    return out, k


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR loss."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    targets = torch.arange(2 * N, device=z.device)
    targets = (targets + N) % (2 * N)
    return F.cross_entropy(sim, targets)


# -------------------------------------------------------------- Mean Teacher
@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, tau: float = 0.99) -> None:
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(tau).add_(sp.data, alpha=1.0 - tau)
    for sb, tb in zip(student.buffers(), teacher.buffers()):
        if tb.dtype == sb.dtype:
            tb.data.copy_(sb.data)


# -------------------------------------------------------------- Logit adjust
def logit_adjustment(logits: torch.Tensor, class_priors: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """y' = z - τ * log π_y  (paper: τ_adj = 0.5)."""
    return logits - tau * torch.log(class_priors.clamp_min(1e-12)).to(logits.device)


# -------------------------------------------------------------- Pseudo label
@torch.no_grad()
def pseudo_label(
    teacher: nn.Module,
    image: torch.Tensor,
    sequence: Optional[torch.Tensor],
    class_priors: torch.Tensor,
    tau_adj: float = 0.5,
    threshold: float = 0.7,
):
    teacher.eval()
    logits = teacher(image, sequence)
    logits = logit_adjustment(logits, class_priors, tau=tau_adj)
    probs = torch.softmax(logits, dim=-1)
    conf, label = probs.max(dim=-1)
    keep = conf >= threshold
    return label, conf, keep


# -------------------------------------------------------------- helper module
class SSPLAMHelper(nn.Module):
    """Bundle of rotation + projection heads for pretraining only."""

    def __init__(self, feat_dim: int, proj_dim: int = 128):
        super().__init__()
        self.rotation_head = RotationHead(feat_dim)
        self.projection_head = ProjectionHead(feat_dim, proj_dim)
