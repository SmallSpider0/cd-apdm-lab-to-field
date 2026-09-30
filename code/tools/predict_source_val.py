"""EXP-5：导出源域验证集的逐图预测，用于比较"与训练集同叶片"的验证图与其余验证图的准确率。

    python -m tools.predict_source_val --run_dir <baselines.run 产物目录> --config <训练所用配置> --out <npz>

在修复前的比例划分（dataset/splits/plantvillage_lt_source_f06c00b.csv，49/2,372 张验证图与训练集同叶片）上
训练 source_only，再在同一验证集上逐图预测；哪些验证图属于泄漏组由当前划分文件的 moved_to_train 标记给出，
离线对照（scripts/analysis/exp5_leakage_effect.py）。只涉及源域，不涉及目标域标签。
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baselines.net import BaselineNet
from src.datasets.splits import build_datasets
from src.utils.config import load_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="bestval", choices=["bestval", "final"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = torch.device("cuda")
    _, src_val, _, ci = build_datasets(cfg, temporal=None)
    ck = torch.load(f"{a.run_dir}/model_{a.ckpt}.pt", map_location="cpu", weights_only=False)
    model = BaselineNet(ck["num_classes"], ck["backbone"], pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    P, Y, ids = [], [], []
    for b in DataLoader(src_val, batch_size=128, num_workers=6):
        P.append(torch.softmax(model(b["image"].to(dev)).float(), -1).cpu()); Y.append(b["label"]); ids += list(b["image_id"])
    np.savez_compressed(a.out, probs=torch.cat(P).numpy().astype(np.float32), labels=torch.cat(Y).numpy(),
                        image_ids=np.array(ids), ckpt=a.ckpt, epoch=ck.get("epoch", -1))
    acc = float((torch.cat(P).argmax(1).numpy() == torch.cat(Y).numpy()).mean())
    print(f"已写入 {a.out}：{len(ids)} 张源域验证图，Top-1 {acc * 100:.2f}")


if __name__ == "__main__":
    main()
