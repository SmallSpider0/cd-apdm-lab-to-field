"""基于划分文件的数据集 —— 本基准的唯一事实来源。

替代原先扫描目录的做法。划分文件由 `scripts/analysis/build_lt_splits.py`
以固定种子生成，随仓库公开（`dataset/splits/`）：

  plantvillage_lt_source.csv   源域 PlantVillage-LT(IF=500)，含 train/val 与 OLTR 分桶
  plantdoc_target.csv          目标域 PlantDoc，含原生 train/test 标记

与原 `image_dataset.py` 的三点关键差异：

1. **不扫描目录。** 划分由文件决定，任何人拿同一份 CSV 得到同一个划分，
   不受目录内容、文件系统排序或大小写敏感性影响。
2. **不静默回退。** 图像缺失、类别对不上、时序对齐失败 —— 一律抛错。
   原实现在对齐失败时会**注入一条随机的 AgriNet 时序**，等于凭空编造输入。
3. **尾类来自分桶而非阈值。** OLTR 约定（Many >100 / Medium 20–100 / Few <20）
   在构造时写入 CSV，运行时不再重新判定。

图像本体不在仓库内，位于 `$AGRI_WORKSPACE/data/`，见 WORKSPACE.md。
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .agrinet import AgriNetTFEM
from .image_dataset import build_eval_transform, build_train_transform

REPO = Path(__file__).resolve().parents[3]
SPLIT_DIR = REPO / "dataset" / "splits"


def workspace_data() -> Path:
    ws = Path(os.environ.get("AGRI_WORKSPACE", Path.home() / "agri-cnz-workspace"))
    for bad in ("Mobile Documents", "CloudDocs", "Dropbox", "OneDrive"):
        if bad in str(ws):
            raise RuntimeError(
                f"AGRI_WORKSPACE 指向云同步目录：{ws}；数据集不得放在同步目录（见 WORKSPACE.md）"
            )
    return ws / "data"


@dataclass
class ClassIndex:
    """源域与目标域共享的类别索引 —— 由源域划分文件确定。"""
    names: List[str]
    counts: List[int]
    buckets: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def tail_classes(self) -> List[int]:
        """Tail = Medium ∪ Few。身份由源域构造分布确定，与目标域计数无关。"""
        return [i for i, b in enumerate(self.buckets) if b != "many"]

    def bucket_classes(self, bucket: str) -> List[int]:
        return [i for i, b in enumerate(self.buckets) if b == bucket]


def _read(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"划分文件不存在：{path}\n"
            f"请先运行：python3 scripts/analysis/build_lt_splits.py"
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_class_index(source_csv: Path | str = SPLIT_DIR / "plantvillage_lt_source.csv") -> ClassIndex:
    source_csv = Path(source_csv)
    if not source_csv.is_absolute():
        source_csv = SPLIT_DIR / source_csv
    rows = _read(source_csv)
    by_idx: Dict[int, dict] = {}
    counts: Counter = Counter()
    for r in rows:
        i = int(r["class_idx"])
        by_idx.setdefault(i, r)
        counts[i] += 1
    order = sorted(by_idx)
    if order != list(range(len(order))):
        raise ValueError(f"class_idx 不连续：{order[:5]}…")
    return ClassIndex(
        names=[by_idx[i]["class"] for i in order],
        counts=[counts[i] for i in order],
        buckets=[by_idx[i]["bucket"] for i in order],
    )


class SplitFileDataset(Dataset):
    """从划分文件读取的图像数据集。

    ``temporal`` 为 None 时不提供时序特征，返回全零占位并置
    ``has_temporal=False``；模型侧应以 ``use_temporal=False`` 构造。
    这是主基准的默认状态 —— PlantVillage 与 PlantDoc 均无可信的采集时间戳
    与地块标识，无法诚实地配对时序数据（对审稿意见 R2-2 的正面回答）。
    """

    def __init__(
        self,
        csv_name: str,
        data_subdir: str,
        domain_id: int,
        class_index: ClassIndex,
        split: Optional[str] = None,
        split_column: str = "split",
        train: bool = True,
        image_size: int = 224,
        temporal: Optional[AgriNetTFEM] = None,
        temporal_dim: int = 16,
        temporal_window: int = 7,
        strict_paths: bool = True,
        transform=None,
    ):
        self.root = workspace_data() / data_subdir
        rows = _read(SPLIT_DIR / csv_name)
        if split is not None:
            rows = [r for r in rows if r.get(split_column) == split]
        if not rows:
            raise ValueError(f"{csv_name} 中 {split_column}={split!r} 没有任何样本")

        self.rows = rows
        self.domain_id = domain_id
        self.class_index = class_index
        self.temporal = temporal
        self._t_dim, self._t_win = temporal_dim, temporal_window
        # transform 可显式覆盖：基线方法需要以训练增强读取目标域（DANN），
        # 或返回弱/强两个视图（FlexMatch）。默认行为不变。
        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_train_transform(image_size) if train else build_eval_transform(image_size)
        self.labels = [int(r["class_idx"]) for r in rows]

        if strict_paths:
            missing = [r["path"] for r in rows if not (self.root / r["path"]).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} / {len(rows)} 个文件在 {self.root} 下不存在，"
                    f"首个：{missing[0]}\n请先运行 ./scripts/fetch-datasets.sh"
                )

        c: Counter = Counter(self.labels)
        self.class_counts = [c.get(i, 0) for i in range(class_index.num_classes)]

    # -------------------------------------------------------------- 便捷属性
    @property
    def num_classes(self) -> int:
        return self.class_index.num_classes

    @property
    def tail_classes(self) -> List[int]:
        return self.class_index.tail_classes

    def indices_of_bucket(self, bucket: str) -> List[int]:
        want = set(self.class_index.bucket_classes(bucket))
        return [i for i, l in enumerate(self.labels) if l in want]

    def head_indices(self, _unused: int = 0) -> List[int]:
        return self.indices_of_bucket("many")

    def tail_indices(self, _unused: int = 0) -> List[int]:
        want = set(self.tail_classes)
        return [i for i, l in enumerate(self.labels) if l in want]

    # -------------------------------------------------------------- Dataset
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        path = self.root / r["path"]
        try:
            with Image.open(path) as img:
                image = self.transform(img.convert("RGB"))
        except Exception as e:  # 损坏文件必须暴露，不得跳过
            raise RuntimeError(f"读取失败 {path}: {e}") from e

        if self.temporal is None:
            seq = torch.zeros(self._t_win, self._t_dim)
            has_temporal = False
        else:
            # 对齐失败时 get_sequence 抛错（strict 模式），不再静默注入随机序列
            seq = self.temporal.get_sequence(r["path"])
            has_temporal = True

        return {
            "image": image,
            "label": int(r["class_idx"]),
            "sequence": seq,
            "has_temporal": has_temporal,
            "domain": self.domain_id,
            "image_id": r["path"],
            "index": idx,
        }


class StripLabels(Dataset):
    """用于训练/自适应的目标域样本：去掉 label，使任何误用都当场抛 KeyError。

    目标域标签只允许出现在评测路径上。凡是参与参数更新的目标域数据
    （基线的 DANN/FlexMatch/CoTTA，CD-APDM 的 MDIWM、CC-GANM、SS-PLAM、
    Mean-Teacher、ACRM 统计）一律经由本类，在结构上拿不到标签。
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = dict(self.ds[i])
        item.pop("label", None)
        return item


def build_datasets(cfg: dict, temporal: Optional[AgriNetTFEM] = None):
    """按配置构建源域 train/val 与目标域数据集。

    返回 ``(source_train, source_val, target, class_index)``。
    目标域不做二次划分 —— 其使用方式（transductive / inductive）属
    fix-experiment-protocol 的待决项，此处保留全量与原生划分标记。
    """
    d = cfg["data"]
    ci = load_class_index(d.get("source_split", "plantvillage_lt_source.csv"))
    size = d.get("image_size", 224)
    common = dict(class_index=ci, image_size=size, temporal=temporal,
                  temporal_window=cfg.get("tfem", {}).get("window_days", 7))

    src_csv = d.get("source_split", "plantvillage_lt_source.csv")
    tgt_csv = d.get("target_split", "plantdoc_target.csv")
    src_train = SplitFileDataset(src_csv, "plantvillage", 0,
                                 split="train", train=True, **common)
    src_val = SplitFileDataset(src_csv, "plantvillage", 0,
                               split="val", train=False, **common)
    tgt = SplitFileDataset(tgt_csv, "plantdoc", 1,
                           split=None, train=False, **common)
    return src_train, src_val, tgt, ci
