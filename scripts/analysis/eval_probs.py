#!/usr/bin/env python3
"""从保存的逐图目标域概率计算指标，可选开发集 / 测试集 / 全部。

    python scripts/analysis/eval_probs.py --subset dev <probs.npz> [...]

口径与 aggregate_table1.py 相同（排除目标域图像 < 5 张的不可评测类）。
**方法开发只看 --subset dev**；测试集只在最终报告时使用（决策记录 6）。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_table1 import load_class_index, metrics, target_meta  # noqa: E402
import csv

ROOT = Path(__file__).resolve().parents[2]


def subset_ids(name):
    rows = list(csv.DictReader(open(ROOT / "dataset/splits/plantdoc_target.csv", newline="")))
    return None if name == "all" else {r["path"] for r in rows if r["eval_split"] == name}


def evaluate(path, subset="dev", drop_conflict=False):
    ci = load_class_index(); conflict, per_class, _ = target_meta()
    evaluable = [c for c in range(ci.num_classes) if per_class[c] >= 5]
    tail = [c for c in ci.tail_classes if c in evaluable]
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["image_ids"]]
    keep = subset_ids(subset)
    m = np.ones(len(ids), bool) if keep is None else np.array([i in keep for i in ids])
    out = metrics(z["probs"][m], z["labels"][m], [i for i, k in zip(ids, m) if k], tail=tail,
                  evaluable=evaluable, conflict=conflict, drop_conflict=drop_conflict, test_only=False)
    pred = z["probs"][m].astype(float).argmax(-1)
    c = np.bincount(pred, minlength=ci.num_classes)
    out["top3_pred_share"] = float(np.sort(c)[-3:].sum() / c.sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("probs", nargs="+")
    ap.add_argument("--subset", default="dev", choices=["dev", "test", "all"])
    a = ap.parse_args()
    if a.subset == "test":
        print("⚠ 测试集只应在方法定稿后的最终报告中使用（决策记录 6）", file=sys.stderr)
    for p in a.probs:
        r = evaluate(p, a.subset)
        print(f"{p}: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, ensure_ascii=False))
