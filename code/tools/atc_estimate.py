"""开发用：检验"不用目标域标签的准确率估计量"在本基准上是否可靠（R1-2：修正 ACRM 的估计量）。

    python -m tools.atc_estimate --config configs/cd_apdm.yaml --checkpoint <ckpt> [--adabn] --out <json>

估计量（均只用**源域验证集标签** + 目标域无标签图像）：
  AC   平均置信度（Hendrycks & Gimpel 2017 的基线）
  DoC  源域验证准确率 − (源域平均置信度 − 目标平均置信度)（Guillory et al. 2021）
  ATC  在源域验证集上取阈值 t，使"分数 > t 的比例"等于源域验证准确率；目标域上分数 > t 的比例
       即为目标准确率估计（Garg et al. 2022）。分数取最大置信度与负熵两种。
  ATC-c 逐类版：按预测类别分组，在源域验证集上逐类定阈值，用于估计逐类伪标签精度。
另输出按置信度阈值筛选后伪标签精度的 ATC 估计，与开发集真实值对照。
目标域真实标签只在本诊断中用于**开发集**对照（决策记录 6）；测试集不参与。
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.splits import SPLIT_DIR, build_datasets
from src.inference import adabn, load_model
from src.utils.config import load_config


@torch.no_grad()
def predict(model, ds, device):
    model.eval()
    P, Y, ids = [], [], []
    for b in DataLoader(ds, batch_size=128, num_workers=6):
        P.append(F.softmax(model(b["image"].to(device), b["sequence"].to(device)).float(), -1).cpu())
        Y.append(b["label"]); ids += list(b["image_id"])
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), ids


def atc_threshold(score, correct):
    """阈值 t：源域上 score > t 的比例 = 源域准确率。"""
    acc = correct.mean()
    return np.quantile(score, 1 - acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adabn", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--save_npz", default=None,
                    help="另存源域验证与全部目标图的概率、标签、图像 ID，供定稿后在测试集上离线计算（EXP-10）")
    a = ap.parse_args()
    cfg = load_config(a.config); dev = torch.device("cuda")
    _, src_val, tgt, ci = build_datasets(cfg, temporal=None)
    model, _ = load_model(a.checkpoint, cfg, 16, dev); model.use_temporal = False
    ps, ys, _ = predict(model, src_val, dev)                 # 源域验证：源模型的 BN 统计
    if a.adabn:
        from src.datasets.splits import StripLabels
        adabn(model, DataLoader(StripLabels(tgt), batch_size=64, num_workers=6), dev)
    pt, yt, ids = predict(model, tgt, dev)
    if a.save_npz:
        np.savez_compressed(a.save_npz, src_val_probs=ps.astype(np.float32), src_val_labels=ys,
                            tgt_probs=pt.astype(np.float32), tgt_labels=yt, tgt_image_ids=np.array(ids), adabn=a.adabn)
    split = {r["path"]: r["eval_split"] for r in csv.DictReader(open(SPLIT_DIR / "plantdoc_target.csv"))}
    m = np.array([split[i] == "dev" for i in ids])
    pt, yt = pt[m], yt[m]

    def score(p, kind):
        return p.max(1) if kind == "maxconf" else (p * np.log(p + 1e-12)).sum(1)

    cs = ps.argmax(1) == ys; ct = pt.argmax(1) == yt
    out = {"adabn": a.adabn, "source_val_acc": float(cs.mean()), "target_dev_true_acc": float(ct.mean()),
           "AC": float(pt.max(1).mean()),
           "DoC": float(cs.mean() - (ps.max(1).mean() - pt.max(1).mean()))}
    for kind in ("maxconf", "negent"):
        t = atc_threshold(score(ps, kind), cs)
        out[f"ATC_{kind}"] = float((score(pt, kind) > t).mean())
    # 逐类（按预测类别）的伪标签精度：ATC-c 估计 vs 开发集真实
    rows = []
    t_glob = atc_threshold(score(ps, "negent"), cs)
    for c in range(ci.num_classes):
        ks, kt = ps.argmax(1) == c, pt.argmax(1) == c
        if kt.sum() < 5 or ks.sum() < 5:
            continue
        tc = atc_threshold(score(ps[ks], "negent"), cs[ks])
        rows.append({"class": c, "n_pred_target": int(kt.sum()),
                     "est_precision": float((score(pt[kt], "negent") > tc).mean()),
                     "est_precision_global_t": float((score(pt[kt], "negent") > t_glob).mean()),
                     "true_precision": float(ct[kt].mean())})
    out["per_class"] = rows
    if rows:
        from scipy import stats
        e = [r["est_precision"] for r in rows]; g = [r["true_precision"] for r in rows]
        out["per_class_spearman"] = float(stats.spearmanr(e, g)[0])
        out["per_class_mean_abs_err"] = float(np.mean(np.abs(np.array(e) - np.array(g))))
    # 按置信度筛选后的伪标签精度：估计（在源域上同样筛选后的 ATC）vs 真实
    sel = []
    for th in (0.5, 0.7, 0.9):
        ks, kt = ps.max(1) >= th, pt.max(1) >= th
        if ks.sum() < 20 or kt.sum() < 5:
            continue
        tt = atc_threshold(score(ps[ks], "negent"), cs[ks])
        sel.append({"conf_threshold": th, "coverage": float(kt.mean()),
                    "est_precision": float((score(pt[kt], "negent") > tt).mean()),
                    "true_precision": float(ct[kt].mean())})
    out["selected_pseudo_labels"] = sel
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
