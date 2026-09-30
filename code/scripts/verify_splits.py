#!/usr/bin/env python3
"""本机验证新数据层：划分文件 → DataLoader → 模型前向。

只跑几个 batch，CPU/MPS 即可，用于在上服务器前确认流水线正确。

    conda activate agri
    python code/scripts/verify_splits.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.datasets.splits import build_datasets, load_class_index
from src.datasets.agrinet import AgriNetTFEM, AgriNetTFEMConfig
from src.models.backbone import CDAPDMModel
from src.utils.config import load_config


def device_of():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "cd_apdm.yaml"))
    dev = device_of()
    print(f"设备: {dev}   torch {torch.__version__}\n")

    ci = load_class_index()
    print(f"类别索引: {ci.num_classes} 类")
    print(f"  分桶  many {len(ci.bucket_classes('many'))} / "
          f"medium {len(ci.bucket_classes('medium'))} / few {len(ci.bucket_classes('few'))}")
    print(f"  Tail  {len(ci.tail_classes)} 类 -> {ci.tail_classes}")
    assert cfg["data"]["num_classes"] == ci.num_classes, "配置的 num_classes 与划分文件不符"

    print("\n构建数据集（strict 路径校验）…")
    t0 = time.time()
    src_tr, src_val, tgt, _ = build_datasets(cfg, temporal=None)
    print(f"  source train {len(src_tr):>6}   source val {len(src_val):>5}   target {len(tgt):>5}"
          f"   ({time.time()-t0:.1f}s)")

    assert set(src_tr.labels) <= set(range(ci.num_classes))
    assert len(set(src_val.labels)) == ci.num_classes, "验证集未覆盖全部类别"
    print(f"  源域类别计数（训练）: max {max(src_tr.class_counts)}  min {min(src_tr.class_counts)}"
          f"  IF {max(src_tr.class_counts)/min(src_tr.class_counts):.1f}")
    print(f"  尾类样本索引数: train {len(src_tr.tail_indices())}  target {len(tgt.tail_indices())}")

    print("\n取 batch…")
    for name, ds in (("source-train", src_tr), ("target", tgt)):
        dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
        b = next(iter(dl))
        print(f"  {name:<13} image {tuple(b['image'].shape)}  label {tuple(b['label'].shape)}"
              f"  seq {tuple(b['sequence'].shape)}  has_temporal={bool(b['has_temporal'][0])}")
        assert b["image"].shape[1:] == (3, cfg["data"]["image_size"], cfg["data"]["image_size"])

    print("\n模型前向（use_temporal=False，主基准默认）…")
    model = CDAPDMModel(num_classes=ci.num_classes, temporal_input_dim=16,
                        pretrained_backbone=False, use_temporal=False).to(dev)
    dl = DataLoader(src_tr, batch_size=8, shuffle=True, num_workers=0)
    b = next(iter(dl))
    t0 = time.time()
    with torch.no_grad():
        logits = model(b["image"].to(dev))
    print(f"  logits {tuple(logits.shape)}   {time.time()-t0:.2f}s")
    assert logits.shape == (8, ci.num_classes)

    print("\nAgriNet strict 模式应对未知 image_id 抛错…")
    try:
        tf = AgriNetTFEM(AgriNetTFEMConfig(
            window_csv=str(Path(__file__).resolve().parents[2] / "dataset/AgriNet/AgriNet_sliding_window_7d.csv"),
            align_csv=str(Path(__file__).resolve().parents[2] / "dataset/AgriNet/AgriNet_image_alignment.csv"),
            feature_columns=cfg["tfem"]["feature_columns"],
            event_columns=cfg["tfem"]["event_columns"], strict=True))
        tf.get_sequence("raw/color/Apple___Apple_scab/whatever.JPG")
        print("  ✗ 未抛错 —— strict 模式失效")
        return 1
    except KeyError as e:
        print(f"  ✓ 按预期抛出 KeyError：{str(e)[:60]}…")

    print("\n✓ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
