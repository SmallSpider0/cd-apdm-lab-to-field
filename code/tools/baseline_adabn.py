"""开发用：对已训练的基线 checkpoint 做推理期 AdaBN，导出原始 logits 供离线组合（定稿门槛的补充对照）。

    python -m tools.baseline_adabn --run_dir <基线产物目录> --ckpt bestval --out <npz>

回答的问题：修订版 CD-APDM（AdaBN + logit 调整 τ + ACRM 尾类重标定 γ）相对基线的增益，有多少来自
AdaBN 这一通用技术本身。导出同一模型在 AdaBN 前后对全部目标图的 logits，以及源域逐类计数与尾类，
τ 与 γ 的组合在本机离线计算（scripts/analysis/adabn_decompose.py）。
AdaBN 只用目标图像，不用标签；指标只在开发集上计算（决策记录 6），测试集不参与。
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.baselines.net import BaselineNet
from src.datasets.splits import StripLabels, build_datasets
from src.utils.config import load_config


@torch.no_grad()
def logits_of(model, ds, device):
    model.eval()
    Z, Y, ids = [], [], []
    for b in DataLoader(ds, batch_size=128, num_workers=6):
        Z.append(model(b["image"].to(device)).float().cpu()); Y.append(b["label"]); ids += list(b["image_id"])
    return torch.cat(Z).numpy(), torch.cat(Y).numpy(), ids


@torch.no_grad()
def adabn(model, ds, device):
    """与 src.inference.adabn 相同：重置 BN 运行统计量，以全部目标图的累计平均重估（momentum=None）。"""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    for b in DataLoader(StripLabels(ds), batch_size=64, num_workers=6, shuffle=False):
        model(b["image"].to(device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="bestval", choices=["bestval", "final"])
    ap.add_argument("--config", default="configs/cd_apdm.yaml"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = torch.device("cuda")
    src_train, _, tgt, ci = build_datasets(cfg, temporal=None)
    ck = torch.load(f"{a.run_dir}/model_{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = BaselineNet(ck["num_classes"], ck["backbone"], pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    z0, y, ids = logits_of(model, tgt, dev)
    adabn(model, tgt, dev)
    z1, y1, ids1 = logits_of(model, tgt, dev)
    assert ids == ids1 and (y == y1).all()
    np.savez_compressed(a.out, logits_plain=z0, logits_adabn=z1, labels=y, image_ids=np.array(ids),
                        class_counts=np.array(src_train.class_counts), tail=np.array(ci.tail_classes),
                        method=ck["method"], seed=ck["seed"], ckpt=a.ckpt, epoch=ck.get("epoch", -1))
    print(f"已写入 {a.out}（{len(ids)} 张目标图，{ck['method']}-s{ck['seed']} {a.ckpt}）")


if __name__ == "__main__":
    main()
