"""Multi-Source Domain Instance Weighting Module (MDIWM).

For each source sample i:
    w_i = alpha * d_i + beta * u_i
where d_i is the domain similarity score (from ResNet-18 domain
classifier) and u_i is the predictive entropy estimated via MC dropout.

The weights drive instance sampling:
* head classes (count > head_threshold) -> Instance-Balanced Sampling (IBS)
  with replacement, proportional to weight,
* tail classes (count < tail_threshold) -> Re-weighting Sampling (RS)
  to guarantee at least ``min_tail_ratio`` of each mini-batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from ..models.domain_classifier import DomainClassifier


@dataclass
class MDIWMConfig:
    alpha: float = 0.7
    beta: float = 0.3
    mc_dropout_p: float = 0.2
    mc_passes: int = 10
    head_threshold: int = 500
    tail_threshold: int = 100
    min_tail_ratio: float = 0.15


# ----------------------------------------------------------------- weighting
@torch.no_grad()
def mc_dropout_entropy(model, image: torch.Tensor, passes: int = 10) -> torch.Tensor:
    """Average softmax then take entropy — matches Eq. for u_i."""
    was_training = model.training
    model.enable_mc_dropout(True)
    probs = None
    for _ in range(passes):
        logits = model(image)
        p = torch.softmax(logits, dim=-1)
        probs = p if probs is None else probs + p
    probs = probs / passes
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    model.enable_mc_dropout(False)
    if not was_training:
        model.eval()
    return entropy


def compute_instance_weights(
    domain_scores: np.ndarray,
    uncertainties: np.ndarray,
    alpha: float,
    beta: float,
    normalize: bool = True,
) -> np.ndarray:
    """w_i = α·d_i + β·u_i, optionally normalized to mean 1."""
    w = alpha * domain_scores + beta * uncertainties
    w = np.clip(w, a_min=1e-6, a_max=None)
    if normalize and w.mean() > 0:
        w = w / w.mean()
    return w


# --------------------------------------------------------- combined sampler
class HeadTailWeightedSampler(Sampler[int]):
    """Concatenates an IBS draw over head samples with an RS draw over tail samples.

    The sampler returns ``num_samples`` indices per epoch. Within each
    mini-batch, by construction ``min_tail_ratio`` of slots are reserved
    for tail-class samples (paper requires >=15%).
    """

    def __init__(
        self,
        labels: Sequence[int],
        class_counts: Sequence[int],
        weights: np.ndarray,
        batch_size: int,
        num_samples: int,
        head_threshold: int,
        tail_threshold: int,
        min_tail_ratio: float = 0.15,
        seed: int = 0,
        head_classes: Sequence[int] | None = None,
        tail_classes: Sequence[int] | None = None,
    ):
        self.labels = np.asarray(labels)
        self.weights = weights
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

        cls_counts = np.asarray(class_counts)
        # 优先使用显式给定的类别集合（来自划分文件的 OLTR 分桶）。
        # 阈值判定仅作后备 —— 它依赖运行时的类别计数，而训练集计数已因
        # 切出验证集而偏离构造时的分布，据此重判会错划边界类别。
        if head_classes is not None and tail_classes is not None:
            head_classes = np.asarray(sorted(head_classes), dtype=np.int64)
            tail_classes = np.asarray(sorted(tail_classes), dtype=np.int64)
        else:
            head_classes = np.where(cls_counts > head_threshold)[0]
            tail_classes = np.where(cls_counts < tail_threshold)[0]
        mid_classes = np.setdiff1d(np.arange(len(cls_counts)), np.concatenate([head_classes, tail_classes]))

        self.head_idx = np.where(np.isin(self.labels, head_classes))[0]
        self.tail_idx = np.where(np.isin(self.labels, tail_classes))[0]
        self.mid_idx = np.where(np.isin(self.labels, mid_classes))[0]
        self.min_tail_ratio = min_tail_ratio if len(self.tail_idx) > 0 else 0.0

    # ------------------------------------------------------------------ helpers
    def _draw(self, pool: np.ndarray, k: int) -> np.ndarray:
        if len(pool) == 0 or k <= 0:
            return np.empty(0, dtype=np.int64)
        w = self.weights[pool]
        w = w / w.sum()
        return self.rng.choice(pool, size=k, replace=True, p=w)

    # ------------------------------------------------------------------ Sampler
    def __iter__(self):
        n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        out: List[int] = []
        n_tail = int(round(self.batch_size * self.min_tail_ratio))
        for _ in range(n_batches):
            batch = []
            if n_tail > 0:
                batch.extend(self._draw(self.tail_idx, n_tail).tolist())
            remaining = self.batch_size - len(batch)
            # IBS for head, with the residual filled from non-tail pool
            pool = np.concatenate([self.head_idx, self.mid_idx])
            if len(pool) == 0:
                pool = self.tail_idx
            batch.extend(self._draw(pool, remaining).tolist())
            self.rng.shuffle(batch)
            out.extend(batch)
        return iter(out[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples


# --------------------------------------------------------------- main wrapper
class MDIWM:
    """Coordinates domain-classifier training and per-sample weight computation."""

    def __init__(self, cfg: MDIWMConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.domain_classifier: Optional[DomainClassifier] = None

    # --------------------------------------------------------- domain training
    def fit_domain_classifier(self, source_loader, target_loader, epochs: int = 3, lr: float = 1e-3):
        self.domain_classifier = DomainClassifier(pretrained=True).to(self.device)
        opt = torch.optim.Adam(self.domain_classifier.parameters(), lr=lr)
        bce = torch.nn.BCEWithLogitsLoss()

        for ep in range(epochs):
            self.domain_classifier.train()
            for src_batch, tgt_batch in zip(source_loader, target_loader):
                src_img = src_batch["image"].to(self.device)
                tgt_img = tgt_batch["image"].to(self.device)
                imgs = torch.cat([src_img, tgt_img], dim=0)
                labels = torch.cat([
                    torch.zeros(src_img.size(0), device=self.device),
                    torch.ones(tgt_img.size(0), device=self.device),
                ], dim=0)
                logits = self.domain_classifier(imgs)
                loss = bce(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

    # ------------------------------------------------------ compute per-sample d_i
    @torch.no_grad()
    def domain_scores(self, image: torch.Tensor) -> torch.Tensor:
        assert self.domain_classifier is not None, "fit_domain_classifier() first"
        self.domain_classifier.eval()
        return self.domain_classifier.domain_score(image.to(self.device))

    # ------------------------------------------------------ compute u_i (MC dropout)
    def uncertainties(self, classifier_model, image: torch.Tensor) -> torch.Tensor:
        return mc_dropout_entropy(classifier_model, image.to(self.device), passes=self.cfg.mc_passes)

    # --------------------------------------------------------- batch reweighting
    def batch_instance_weights(self, model, batch) -> torch.Tensor:
        img = batch["image"].to(self.device)
        d = self.domain_scores(img).cpu().numpy()
        u = self.uncertainties(model, img).cpu().numpy()
        w = compute_instance_weights(d, u, self.cfg.alpha, self.cfg.beta)
        return torch.from_numpy(w).float().to(self.device)
