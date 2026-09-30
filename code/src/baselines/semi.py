"""FlexMatch 阈值与 CoTTA 适配 —— 均按官方实现逐行对照移植。

FlexMatch：microsoft/Semi-supervised-learning，
    semilearn/algorithms/flexmatch/utils.py 的 FlexMatchThresholdingHook
CoTTA：qinenergy/cotta，cifar/cotta.py

与官方的偏离均在此处与 configs/baselines.yaml 中写明，不做隐性修改。
"""
from __future__ import annotations

import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


# ------------------------------------------------------------------ FlexMatch
class FlexMatchThreshold:
    """课程式逐类阈值。

    selected_label 初始全为 -1；置信度 ≥ p_cutoff 的样本记下其伪标签。
    classwise_acc[c] = 该类被选中数 / 各计数最大值（warmup 时 -1 也参与取最大值，
    故训练初期阈值接近 0、随学习推进逐步升至 p_cutoff）。
    掩码阈值 = p_cutoff · β/(2 − β)，其中 β = classwise_acc[伪标签]。
    顺序与官方一致：先用当前阈值算掩码，再更新 selected_label 与 classwise_acc。
    """

    def __init__(self, n_unlabeled: int, num_classes: int, p_cutoff: float = 0.95,
                 thresh_warmup: bool = True, device="cpu"):
        self.n = n_unlabeled
        self.C = num_classes
        self.p_cutoff = p_cutoff
        self.warmup = thresh_warmup
        self.selected = torch.full((n_unlabeled,), -1, dtype=torch.long, device=device)
        self.classwise_acc = torch.zeros(num_classes, device=device)

    def _update(self) -> None:
        counter = Counter(self.selected.tolist())
        if max(counter.values()) < self.n:
            if self.warmup:
                denom = max(counter.values())
            else:
                wo = {k: v for k, v in counter.items() if k != -1}
                if not wo:
                    return
                denom = max(wo.values())
            for i in range(self.C):
                self.classwise_acc[i] = counter.get(i, 0) / denom

    @torch.no_grad()
    def mask(self, probs: torch.Tensor, idx: torch.Tensor):
        max_probs, max_idx = probs.max(dim=-1)
        beta = self.classwise_acc[max_idx]
        mask = max_probs.ge(self.p_cutoff * (beta / (2.0 - beta))).float()
        select = max_probs.ge(self.p_cutoff)
        if select.any():
            self.selected[idx[select]] = max_idx[select]
            self._update()
        return mask, max_idx

    def status(self) -> dict:
        return {"selected_ratio": float((self.selected >= 0).float().mean()),
                "mean_class_threshold": float((self.p_cutoff * self.classwise_acc
                                               / (2 - self.classwise_acc)).mean())}


def flexmatch_views(size: int, mean, std):
    """弱/强两个视图。弱：随机裁剪 + 翻转；强：弱 + RandAugment(2, 10) + RandomErasing（Cutout 的等价物）。"""
    norm = T.Normalize(mean, std)
    weak = T.Compose([T.Resize((size + 32, size + 32)), T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                      T.RandomHorizontalFlip()])
    strong = T.Compose([T.Resize((size + 32, size + 32)), T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                        T.RandomHorizontalFlip(), T.RandAugment(num_ops=2, magnitude=10)])
    to_t = T.Compose([T.ToTensor(), norm])
    erase = T.RandomErasing(p=1.0, scale=(0.02, 0.2), value=0)

    def two_view(img):
        return to_t(weak(img)), erase(to_t(strong(img)))
    return two_view


# ---------------------------------------------------------------------- CoTTA
def _softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def _configure(model: nn.Module) -> nn.Module:
    """照官方 configure_model：train 模式、BN 用批统计、全部参数可训练。
    偏离：分类头的 Dropout 置为 eval —— 官方模型不含 Dropout，
    保持其开启会在伪标签目标上叠加与方法无关的随机性。"""
    model.train()
    model.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.Dropout):
            m.eval()
    return model


class CoTTA:
    def __init__(self, model: nn.Module, mean, std, lr=1e-3, mt=0.999, rst=0.01, ap=0.72,
                 n_aug=32, seed=0):
        self.model = _configure(model)
        self.source_state = copy.deepcopy(self.model.state_dict())
        self.ema = copy.deepcopy(self.model)
        self.anchor = copy.deepcopy(self.model)
        for m in (self.ema, self.anchor):
            m.requires_grad_(False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.mt, self.rst, self.ap, self.n_aug = mt, rst, ap, n_aug
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.gen = torch.Generator().manual_seed(seed)
        # 官方 get_tta_transforms 的 224 版对应：色彩扰动、仿射、模糊、翻转、高斯噪声
        self.aug = T.Compose([
            T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.7, 1.3), saturation=(0.5, 1.5), hue=(-0.06, 0.06)),
            T.RandomAffine(degrees=15, translate=(1 / 16, 1 / 16), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=5, sigma=(0.001, 0.5)),
            T.RandomHorizontalFlip(),
        ])
        self.triggered = 0
        self.batches = 0

    def _tta(self, x: torch.Tensor) -> torch.Tensor:
        """官方把 transform 作用于整个批张量 —— 一次随机参数作用于全批，此处一致。"""
        mean, std = self.mean.to(x.device), self.std.to(x.device)
        raw = (x * std + mean).clamp(0, 1)
        aug = self.aug(raw)
        aug = (aug + 0.005 * torch.randn_like(aug)).clamp(0, 1)
        return (aug - mean) / std

    def step(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            anchor_prob = F.softmax(self.anchor(x), dim=1).max(1)[0]
            out_ema = self.ema(x)
            if anchor_prob.mean() < self.ap:
                out_ema = torch.stack([self.ema(self._tta(x)) for _ in range(self.n_aug)]).mean(0)
                self.triggered += 1
        out = self.model(x)
        loss = _softmax_entropy(out, out_ema).mean()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        with torch.no_grad():
            for pe, p in zip(self.ema.parameters(), self.model.parameters()):
                pe.data.mul_(self.mt).add_((1 - self.mt) * p.data)
            for name, p in self.model.named_parameters():
                if name.split(".")[-1] in ("weight", "bias") and p.requires_grad:
                    m = (torch.rand(p.shape, generator=self.gen) < self.rst).float().to(p.device)
                    p.data = self.source_state[name].to(p.device) * m + p.data * (1 - m)
        self.batches += 1
        return out_ema
