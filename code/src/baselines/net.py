"""基线共用的网络：骨干 + 与 CD-APDM 相同结构的分类头。

分类头刻意与 ``CDAPDMModel.classifier`` 同构（Linear→ReLU→Dropout→Linear），
使 Table 1 中各行的差异只来自训练目标，而不是分类头的容量。
不含 CD-APDM 的可学习温度 —— 那是 CIWLM 的一部分，不属于基线。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from ..models.backbone import build_resnet50


def build_swin_b(pretrained: bool = True) -> nn.Module:
    weights = tvm.Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
    net = tvm.swin_b(weights=weights)
    feat_dim = net.head.in_features
    net.head = nn.Identity()
    net.feature_dim = feat_dim  # type: ignore[attr-defined]
    return net


class BaselineNet(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet50":
            self.backbone = build_resnet50(pretrained=pretrained)
        elif backbone == "swin_b":
            self.backbone = build_swin_b(pretrained=pretrained)
        else:
            raise ValueError(f"未知骨干 {backbone}")
        d = self.backbone.feature_dim  # type: ignore[attr-defined]
        self.feature_dim = d
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d // 2, num_classes),
        )

    @property
    def last_layer(self) -> nn.Linear:
        """WB 第二阶段只训练、并施加 MaxNorm 的那一层。"""
        return self.classifier[3]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.backbone(x)
        logits = self.classifier(f)
        return (logits, f) if return_features else logits


# ------------------------------------------------------------------ DANN 组件
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """Ganin et al. (2016) 的三层判别器：1024-1024-1，ReLU + Dropout 0.5。"""

    def __init__(self, in_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f).squeeze(-1)
