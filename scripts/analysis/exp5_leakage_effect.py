"""EXP-5（rebuild-cross-domain-benchmark 5.4）：修复前的比例划分中，验证集泄漏对源域验证准确率与目标域结果的影响。

    python scripts/analysis/exp5_leakage_effect.py results/remote/exp-final --out results/exp-5-leakage-effect.json

泄漏验证图：修复前划分（dataset/splits/plantvillage_lt_source_f06c00b.csv）的验证集中，与该划分训练集同属一组
（当前划分文件的 leaf_group：近重复 ∪ 同叶片）的图像，共 49 / 2,372 张（与 results/exp-5-leakage-audit.json 一致）。
对每个 seed：泄漏验证图与其余验证图的 Top-1；以及在修复前划分上训练的 source_only 与正式划分上训练的同 seed
source_only（results/remote/exp2）在目标域测试集上的 Top-1 / 尾类召回，检验泄漏是否改变选型从而影响报告的目标域结果。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_table1 as t1  # noqa: E402


def leaked_val_ids():
    old = {r["path"]: r for r in csv.DictReader(open(ROOT / "dataset/splits/plantvillage_lt_source_f06c00b.csv"))}
    cur = {r["path"]: r for r in csv.DictReader(open(ROOT / "dataset/splits/plantvillage_lt_source.csv"))}
    splits = collections.defaultdict(set)
    for p, r in old.items():
        g = cur[p]["leaf_group"]
        if g:
            splits[g].add(r["split"])
    return {p for p, r in old.items() if r["split"] == "val" and cur[p]["leaf_group"] and len(splits[cur[p]["leaf_group"]]) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    leaked = leaked_val_ids()
    ci = t1.load_class_index()
    conflict, per_class, _ = t1.target_meta()
    evaluable = [c for c in range(ci.num_classes) if per_class[c] >= t1.UNEVALUABLE_MIN_IMAGES]
    tail = [c for c in ci.tail_classes if c in evaluable]
    rows = {}
    for d in sorted(a.root.glob("exp5-s*")):
        s = int(d.name.split("-s")[1])
        z = np.load(d / "src_val_pred.npz")
        ok = z["probs"].argmax(1) == z["labels"]
        m = np.array([str(i) in leaked for i in z["image_ids"]])
        r = {"n_val": int(len(ok)), "n_leaked": int(m.sum()), "acc_all": float(ok.mean()) * 100,
             "acc_leaked": float(ok[m].mean()) * 100, "acc_clean": float(ok[~m].mean()) * 100}
        r["acc_all_minus_clean"] = r["acc_all"] - r["acc_clean"]
        for tag, f in (("original_split", d / "probs_bestval.npz"),
                       ("repaired_split", ROOT / f"results/remote/exp2/source_only-s{s}/probs_bestval.npz")):
            if f.exists():
                p = np.load(f)
                mt = t1.metrics(p["probs"], p["labels"], [str(i) for i in p["image_ids"]], tail=tail,
                                evaluable=evaluable, conflict=conflict, drop_conflict=False)
                r[f"target_test_{tag}"] = {"top1": mt["top1"], "tail_recall": mt["tail_recall"]}
        rows[s] = r
    agg = {}
    for k in ("acc_all", "acc_leaked", "acc_clean", "acc_all_minus_clean"):
        v = [r[k] for r in rows.values()]
        agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")}
    # 目标域：修复前划分训练 vs 正式划分训练（同 seed 配对）
    from scipy import stats
    for k in ("top1", "tail_recall"):
        o = [r["target_test_original_split"][k] for r in rows.values()]
        p = [r["target_test_repaired_split"][k] for r in rows.values()]
        agg[f"target_{k}"] = {"original_mean": float(np.mean(o)), "repaired_mean": float(np.mean(p)),
                              "diff_mean": float(np.mean(o) - np.mean(p)), "paired_p": float(stats.ttest_rel(o, p).pvalue)}
    out = {"n_leaked_expected": len(leaked), "per_seed": rows, "aggregate": agg}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
