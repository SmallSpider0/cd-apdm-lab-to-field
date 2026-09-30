"""基线用的损失与约束。刻意独立实现，不复用 CD-APDM 的 CIWLM ——
基线不应继承被比较方法的实现细节（包括可能的缺陷）。"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Lin et al. (2017)：-(1 - p_t)^γ · log p_t，无类别权重 α。"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        p = logp.exp()
        return (-(1 - p).pow(self.gamma) * logp).mean()


class ClassBalancedSoftmaxLoss(nn.Module):
    """WB 官方实现所用的 CB 损失（Cui et al. 2019，loss_type="softmax"）。

    照搬官方 ``utils/class_balanced_loss.py``：权重为 (1-β)/(1-β^n)，
    归一化为总和等于类别数；损失是 **softmax 概率对 one-hot 的加权 BCE**，
    不是加权 CE —— 这是官方代码的实际行为，此处保持一致而非"改正"它。
    """

    def __init__(self, class_counts: Sequence[int], beta: float = 0.9999):
        super().__init__()
        n = np.asarray(class_counts, dtype=np.float64)
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))
        self.C = len(n)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(target, self.C).float()
        weights = (self.w[None, :] * onehot).sum(1, keepdim=True).expand(-1, self.C)
        pred = logits.float().softmax(dim=1)
        return F.binary_cross_entropy(pred, onehot, weight=weights)


class MaxNormPGD:
    """WB 官方 ``MaxNorm_via_PGD``：逐类权重向量的范数上限，投影梯度下降实现。

    阈值在第二阶段开始时**一次性**确定为
    ``min_norm + thresh · (max_norm − min_norm)``（逐行 L2 范数），
    此后每步优化后把超限的行缩放回阈值。偏置为一维，阈值为 ∞，即不约束。
    """

    def __init__(self, layer: nn.Linear, thresh: float = 0.1):
        self.layer = layer
        with torch.no_grad():
            norms = layer.weight.norm(p=2, dim=1)
            self.limit = float(norms.min() + thresh * (norms.max() - norms.min()))

    @torch.no_grad()
    def project(self) -> None:
        w = self.layer.weight
        norms = w.norm(p=2, dim=1, keepdim=True)
        scale = torch.clamp(self.limit / norms.clamp_min(1e-12), max=1.0)
        w.mul_(scale)


def dann_lambda(progress: float) -> float:
    """Ganin et al. (2016) 的 GRL 系数调度：2/(1+exp(-10p)) − 1。"""
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
