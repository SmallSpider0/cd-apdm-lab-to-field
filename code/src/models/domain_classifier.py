"""Lightweight ResNet-18 domain classifier used by MDIWM.

Trained briefly with binary cross-entropy (source vs. target) on the
union of the two domains. Its output probability serves as the domain
discrepancy score ``d_i`` in the MDIWM weighting formula.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class DomainClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
        except AttributeError:
            net = tvm.resnet18(pretrained=pretrained)
        in_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image)).squeeze(-1)

    @torch.no_grad()
    def domain_score(self, image: torch.Tensor) -> torch.Tensor:
        """Probability that ``image`` belongs to the *target* domain.

        Used by MDIWM as a measure of how source-like (low) or target-like
        (high) a labelled source sample looks — higher scores mean the
        sample is closer to the target distribution and is therefore more
        informative for adaptation.
        """
        return torch.sigmoid(self.forward(image))
