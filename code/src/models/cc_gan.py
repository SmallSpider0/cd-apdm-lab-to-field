"""Class-Conditional CycleGAN-style style transfer (CC-GANM).

Simplified, single-step generator pair G_{S->T}, G_{T->S} conditioned
on the (pseudo-)class label via an embedding broadcast into the feature
map. Loss = adversarial + λ_cycle * cycle. We keep it light-weight on
purpose — the goal is to deliver class-consistent style transfer for
target rare-class augmentation, not a full GAN benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)   # 与 datasets/image_dataset.imagenet_normalize 一致
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CCGANConfig:
    hidden_dim: int = 64
    cycle_weight: float = 10.0
    adv_weight: float = 1.0
    num_classes: int = 38
    image_channels: int = 3


def _conv(ic, oc, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.InstanceNorm2d(oc),
        nn.ReLU(inplace=True),
    )


def _deconv(ic, oc):
    return nn.Sequential(
        nn.ConvTranspose2d(ic, oc, 4, 2, 1),
        nn.InstanceNorm2d(oc),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, cfg: CCGANConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.label_emb = nn.Embedding(cfg.num_classes, d)
        self.head = _conv(cfg.image_channels, d, 7, 1, 3)
        self.down1 = _conv(d, d * 2, 3, 2, 1)
        self.down2 = _conv(d * 2, d * 4, 3, 2, 1)
        self.res = nn.Sequential(*[ResidualBlock(d * 4) for _ in range(3)])
        self.up1 = _deconv(d * 4, d * 2)
        self.up2 = _deconv(d * 2, d)
        self.tail = nn.Sequential(
            nn.Conv2d(d, cfg.image_channels, 7, 1, 3),
            nn.Tanh(),
        )
        # 输入与真实图像都处于 ImageNet 标准化空间（约 [-2.1, 2.6]），而 Tanh 输出 [-1, 1]。
        # 此前直接输出 Tanh：生成图与真实图数值范围不同，判别器仅凭范围即可区分真伪，
        # 学生网络也在范围错误的图像上训练（2026-09-22 诊断，results/exp-2-diag-s0.json）。
        # 现把 Tanh 映射到像素空间 [0, 1] 再做同一标准化，使生成图与真实图处于同一空间。
        self.register_buffer("px_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("px_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h1 = self.down1(h)
        h2 = self.down2(h1)
        emb = self.label_emb(label).unsqueeze(-1).unsqueeze(-1)
        emb = emb.expand(-1, -1, h2.size(2), h2.size(3))
        # broadcast label embedding into the bottleneck by addition along channel dim 0..d
        h2 = h2 + F.pad(emb, (0, 0, 0, 0, 0, h2.size(1) - emb.size(1)))
        h2 = self.res(h2)
        u1 = self.up1(h2)
        u2 = self.up2(u1)
        pixel = (self.tail(u2) + 1.0) / 2.0          # [-1, 1] → [0, 1]
        return (pixel - self.px_mean) / self.px_std  # → ImageNet 标准化空间


class PatchDiscriminator(nn.Module):
    def __init__(self, cfg: CCGANConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(cfg.image_channels, d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.InstanceNorm2d(d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.InstanceNorm2d(d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class CCGANM(nn.Module):
    """Class-conditional CycleGAN trainer (kept minimal but functional)."""

    def __init__(self, cfg: CCGANConfig):
        super().__init__()
        self.cfg = cfg
        self.G_S2T = ConditionalGenerator(cfg)
        self.G_T2S = ConditionalGenerator(cfg)
        self.D_T = PatchDiscriminator(cfg)
        self.D_S = PatchDiscriminator(cfg)

    @staticmethod
    def adv_real(d_out):
        return F.mse_loss(d_out, torch.ones_like(d_out))

    @staticmethod
    def adv_fake(d_out):
        return F.mse_loss(d_out, torch.zeros_like(d_out))

    def generator_step(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_label: torch.Tensor,
        tgt_label: torch.Tensor,
    ):
        fake_t = self.G_S2T(src, src_label)
        fake_s = self.G_T2S(tgt, tgt_label)
        cyc_s = self.G_T2S(fake_t, src_label)
        cyc_t = self.G_S2T(fake_s, tgt_label)
        adv = self.adv_real(self.D_T(fake_t)) + self.adv_real(self.D_S(fake_s))
        cycle = F.l1_loss(cyc_s, src) + F.l1_loss(cyc_t, tgt)
        return (
            self.cfg.adv_weight * adv + self.cfg.cycle_weight * cycle,
            fake_t.detach(),
            fake_s.detach(),
        )

    def discriminator_step(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        fake_t: torch.Tensor,
        fake_s: torch.Tensor,
    ):
        d_t = self.adv_real(self.D_T(tgt)) + self.adv_fake(self.D_T(fake_t))
        d_s = self.adv_real(self.D_S(src)) + self.adv_fake(self.D_S(fake_s))
        return 0.5 * (d_t + d_s)

    @torch.no_grad()
    def translate_to_target(self, image: torch.Tensor, pseudo_label: torch.Tensor) -> torch.Tensor:
        self.G_S2T.eval()
        return self.G_S2T(image, pseudo_label)
