"""Main CD-APDM model: ResNet-50 + TFEM(LSTM) + MHSA fusion + classifier."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm

from ..modules.tfem import TFEMEncoder


def build_resnet50(pretrained: bool = True) -> nn.Module:
    try:
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
    except AttributeError:  # older torchvision
        net = tvm.resnet50(pretrained=pretrained)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    net.feature_dim = feat_dim  # type: ignore[attr-defined]
    return net


class MHSAFusion(nn.Module):
    """Multi-Head Self-Attention over [image_token, temporal_token]."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, image_token: torch.Tensor, temporal_token: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([image_token, temporal_token], dim=1)
        h, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + h)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.mean(dim=1)


class LearnableTemperature(nn.Module):
    """Logit temperature scaling — ``z / T`` with optional learning."""

    def __init__(self, init_T: float = 1.2, learnable: bool = True):
        super().__init__()
        log_T = torch.log(torch.tensor(init_T, dtype=torch.float32))
        if learnable:
            self.log_T = nn.Parameter(log_T)
        else:
            self.register_buffer("log_T", log_T)

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T


class CDAPDMModel(nn.Module):
    """End-to-end CD-APDM classifier."""

    def __init__(
        self,
        num_classes: int,
        temporal_input_dim: int,
        temporal_hidden: int = 128,
        temporal_layers: int = 1,
        mhsa_heads: int = 4,
        mhsa_dropout: float = 0.1,
        init_T: float = 1.2,
        learnable_T: bool = True,
        pretrained_backbone: bool = True,
        use_temporal: bool = True,
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.backbone = build_resnet50(pretrained=pretrained_backbone)
        feat_dim = self.backbone.feature_dim  # type: ignore[attr-defined]

        self.temporal_encoder = TFEMEncoder(temporal_input_dim, temporal_hidden,
                                            num_layers=temporal_layers)
        self.temporal_proj = nn.Linear(temporal_hidden, feat_dim)
        self.fusion = MHSAFusion(feat_dim, num_heads=mhsa_heads, dropout=mhsa_dropout)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, num_classes),
        )
        self.temperature = LearnableTemperature(init_T=init_T, learnable=learnable_T)
        self._mc_dropout_active = False

    # ------------------------------------------------------------------ helpers
    def enable_mc_dropout(self, flag: bool = True) -> None:
        """Toggle dropout layers on at inference for MC-dropout uncertainty."""
        self._mc_dropout_active = flag
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(flag)

    def extract_image_feature(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def fuse(self, image_feat: torch.Tensor, seq: Optional[torch.Tensor]) -> torch.Tensor:
        if seq is None or not self.use_temporal:
            return image_feat
        t_feat = self.temporal_proj(self.temporal_encoder(seq))
        return self.fusion(image_feat, t_feat)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        image: torch.Tensor,
        sequence: Optional[torch.Tensor] = None,
        apply_temperature: bool = True,
        return_features: bool = False,
    ):
        img_feat = self.extract_image_feature(image)
        fused = self.fuse(img_feat, sequence)
        logits = self.classifier(fused)
        if apply_temperature:
            logits = self.temperature(logits)
        if return_features:
            return logits, fused
        return logits
