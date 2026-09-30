"""图像数据集（旧版，按目录扫描）。

⚠ **主基准请改用 `splits.py` 的 `SplitFileDataset`。** 本模块保留仅为
向后兼容与合成数据冒烟测试。按目录扫描的划分不可复现，且
`synthetic_when_missing` 会在图像缺失时静默回退到合成数据 ——
`code/checkpoints/smoke/` 下 `top1: 0.0` 的产物即由此而来。

Each sample yields::

    image_tensor       FloatTensor (3, H, W)
    label              int
    temporal_sequence  FloatTensor (W_days, n_feat)   from AgriNet TFEM
    domain_id          int   (0 = source, 1 = target)
    image_id           str
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .agrinet import AgriNetTFEM


# ---------------------------------------------------------------------- utils
def imagenet_normalize() -> transforms.Normalize:
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def build_train_transform(size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])


def build_eval_transform(size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])


# ----------------------------------------------------------------- real dataset
def _scan_folder(root: Path) -> Tuple[List[Path], List[int], List[str]]:
    if not root.exists():
        return [], [], []
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    paths, labels = [], []
    for idx, cls in enumerate(classes):
        for p in (root / cls).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(p)
                labels.append(idx)
    return paths, labels, classes


# ------------------------------------------------------------- synthetic data
def _synthetic_long_tail_counts(num_classes: int, head: int, tail: int) -> List[int]:
    """Power-law decay from `head` (class 0) to `tail` (class C-1)."""
    counts = []
    for i in range(num_classes):
        frac = i / max(num_classes - 1, 1)
        n = head * (tail / max(head, 1)) ** frac
        counts.append(max(int(round(n)), 1))
    return counts


def _synthetic_image(label: int, size: int, rng: random.Random) -> torch.Tensor:
    """Class-conditional procedural image so a model can actually learn."""
    img = torch.zeros(3, size, size)
    base = ((label * 53) % 255) / 255.0
    img[0].add_(base)
    img[1].add_((label * 31 % 255) / 255.0)
    img[2].add_((label * 19 % 255) / 255.0)
    img.add_(torch.randn(3, size, size) * 0.05)
    # add a class-specific block to give the network a learnable signal
    block = max(4, size // 4)
    span = max(size - block, 1)
    y0 = (label * 7) % span
    x0 = (label * 11) % span
    img[:, y0:y0 + block, x0:x0 + block].add_(0.6)
    img.clamp_(0.0, 1.0)
    return imagenet_normalize()(img)


# --------------------------------------------------------------------- Dataset
class AgriImageDataset(Dataset):
    """Image dataset that emits TFEM-ready temporal sequences alongside images."""

    def __init__(
        self,
        root: str,
        domain_id: int,
        num_classes: int,
        tfem: AgriNetTFEM,
        train: bool = True,
        image_size: int = 224,
        synthetic_when_missing: bool = True,
        synthetic_head: int = 600,
        synthetic_tail: int = 50,
        synthetic_seed: int = 0,
    ):
        self.domain_id = domain_id
        self.num_classes = num_classes
        self.tfem = tfem
        self.transform = (
            build_train_transform(image_size) if train else build_eval_transform(image_size)
        )

        paths, labels, classes = _scan_folder(Path(root))
        self.is_synthetic = not paths
        if self.is_synthetic:
            if not synthetic_when_missing:
                raise FileNotFoundError(
                    f"No images under {root} and synthetic fallback disabled."
                )
            counts = _synthetic_long_tail_counts(num_classes, synthetic_head, synthetic_tail)
            self.paths: List[Optional[Path]] = []
            self.labels: List[int] = []
            for cls, n in enumerate(counts):
                for _ in range(n):
                    self.paths.append(None)
                    self.labels.append(cls)
            self.classes = [f"class_{i}" for i in range(num_classes)]
            self._rng = random.Random(synthetic_seed + domain_id)
            self._image_size = image_size
        else:
            self.paths = paths
            self.labels = labels
            self.classes = classes
            self._image_size = image_size
            self._rng = random.Random(synthetic_seed + domain_id)

        # cache class counts (used by samplers / loss)
        self.class_counts = [0] * num_classes
        for lbl in self.labels:
            self.class_counts[lbl] += 1

    # ------------------------------------------------------------------ helpers
    def head_indices(self, head_threshold: int) -> List[int]:
        return [i for i, lbl in enumerate(self.labels) if self.class_counts[lbl] > head_threshold]

    def tail_indices(self, tail_threshold: int) -> List[int]:
        return [i for i, lbl in enumerate(self.labels) if self.class_counts[lbl] < tail_threshold]

    # ------------------------------------------------------------------ Dataset
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        if self.is_synthetic:
            image = _synthetic_image(label, self._image_size, self._rng)
            image_id = f"SYN_{self.domain_id}_{idx}"
        else:
            with Image.open(self.paths[idx]) as img:
                image = self.transform(img.convert("RGB"))
            image_id = self.paths[idx].stem

        # 此前此处在对齐失败时会注入一条随机 AgriNet 时序，等于编造模型输入，
        # 且在真实数据上必然触发（对齐表用的是生成的占位标识）。已移除。
        # 时序对齐失败现由 AgriNetTFEM 在 strict 模式下抛错，见 agrinet.py。
        seq = self.tfem.get_sequence(image_id)

        return {
            "image": image,
            "label": label,
            "sequence": seq,
            "domain": self.domain_id,
            "image_id": image_id,
            "index": idx,
        }


def split_indices(n: int, ratios=(0.7, 0.15, 0.15), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
