"""EXP-9 确认偏差 / EXP-10 可靠性估计量 → 补充材料 S6（revise-cdapdm-method 4.6、4.7）。

    python scripts/analysis/exp9_10.py results/remote/exp9-10 --out results/exp-9-10.json [--split test]

输入（队列 exp9-10，每个 seed 三个任务）：
    stages-s<k>/      原提出流程（CC-GANM + SS-PLAM 预训练 + Mean-Teacher，γ=0.08）以 --dump_stages 运行：
                      stage_0_source / 1_after_ssplam / 2_after_mt_student / 2_after_mt_teacher.npz
                      （logit 调整后的逐图概率），eval_full.json 含 ACRM 的逐类师生一致率 acrm_agree_c
    atc_source-s<k>/  源模型（不自适应）：atc.npz 含源域验证与全部目标图的概率
    atc_adabn-s<k>/   定稿模型（AdaBN）：同上

只在 --split 指定的目标域划分上计算（默认 test；dev 仅用于检查脚本）。目标域标签在此只用于诊断，
不参与任何训练、自适应或选型（稿件 4.2.4）。不可评测类别（目标域少于 5 张）不计入逐类统计。

EXP-9  每个阶段：Top-1、按置信度分箱的伪标签精度与覆盖率、置信度 ≥ 0.9（原流程的伪标签阈值）的精度、
       预测集中度（被预测最多的 3 个类所占比例、被预测到的类数）
EXP-10 (a) 师生一致率：按学生预测类别分组，教师与学生一致的比例 vs 该组真实精度（均值与逐类 Spearman）；
           ACRM 在自适应中跟踪的 EMA 一致率 acrm_agree_c vs 最终预测的逐类真实精度
       (b) 无标签准确率估计量 AC / DoC / ATC（最大置信度、负熵）vs 真实准确率；
           逐类 ATC-c 估计的预测精度 vs 真实精度（Spearman、平均绝对误差）
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
STAGES = ["0_source", "1_after_ssplam", "2_after_mt_student", "2_after_mt_teacher"]
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
PSEUDO_THRESHOLD = 0.9
MIN_PRED = 5            # 逐类统计：该类被预测至少 5 次才计入


def split_mask(ids, split):
    rows = list(csv.DictReader(open(ROOT / "dataset/splits/plantdoc_target.csv", newline="")))
    sp = {r["path"]: r["eval_split"] for r in rows}
    per_class = defaultdict(int)
    for r in rows:
        per_class[int(r["class_idx"])] += 1
    evaluable = {c for c, n in per_class.items() if n >= 5}
    return np.array([sp[str(i)] == split for i in ids]), evaluable


def per_class_precision(pred, y, classes):
    return {c: float((y[pred == c] == c).mean()) for c in classes if (pred == c).sum() >= MIN_PRED}


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0]) if len(a) >= 3 else float("nan")


def exp9_stage(p, y):
    conf, pred = p.max(1), p.argmax(1)
    counts = np.bincount(pred, minlength=p.shape[1])
    bins = []
    for lo, hi in BINS:
        m = (conf >= lo) & (conf < hi)
        bins.append({"conf": [lo, min(hi, 1.0)], "coverage": float(m.mean()),
                     "precision": float((pred[m] == y[m]).mean()) if m.any() else None})
    sel = conf >= PSEUDO_THRESHOLD
    return {"top1": float((pred == y).mean()) * 100, "bins": bins,
            "coverage_at_0.9": float(sel.mean()),
            "precision_at_0.9": float((pred[sel] == y[sel]).mean()) if sel.any() else None,
            "top3_pred_share": float(np.sort(counts)[::-1][:3].sum() / len(pred)),
            "n_classes_predicted": int((counts > 0).sum())}


def exp10_agreement(ps, pt, y, evaluable):
    """学生预测为 c 的图像中，教师也预测 c 的比例（一致率） vs 真实精度。"""
    s, t = ps.argmax(1), pt.argmax(1)
    rows = []
    for c in sorted(evaluable):
        m = s == c
        if m.sum() >= MIN_PRED:
            rows.append({"class": c, "n": int(m.sum()), "agreement": float((t[m] == c).mean()),
                         "true_precision": float((y[m] == c).mean())})
    a = [r["agreement"] for r in rows]; g = [r["true_precision"] for r in rows]
    return {"mean_agreement": float(np.mean(a)), "mean_true_precision": float(np.mean(g)),
            "spearman": spearman(a, g), "per_class": rows}


def exp10_estimators(z, mask, evaluable):
    ps, ys = z["src_val_probs"], z["src_val_labels"]
    pt, yt = z["tgt_probs"][mask], z["tgt_labels"][mask]
    cs, ct = ps.argmax(1) == ys, pt.argmax(1) == yt
    score = {"maxconf": lambda p: p.max(1), "negent": lambda p: (p * np.log(p + 1e-12)).sum(1)}

    def atc_t(sc, correct):
        return np.quantile(sc, 1 - correct.mean())

    out = {"true_acc": float(ct.mean()), "source_val_acc": float(cs.mean()),
           "AC": float(pt.max(1).mean()), "DoC": float(cs.mean() - (ps.max(1).mean() - pt.max(1).mean()))}
    for k, f in score.items():
        out[f"ATC_{k}"] = float((f(pt) > atc_t(f(ps), cs)).mean())
    rows = []
    for c in sorted(evaluable):
        ks, kt = ps.argmax(1) == c, pt.argmax(1) == c
        if kt.sum() < MIN_PRED or ks.sum() < MIN_PRED:
            continue
        tc = atc_t(score["negent"](ps[ks]), cs[ks])
        rows.append({"class": c, "est_precision": float((score["negent"](pt[kt]) > tc).mean()),
                     "true_precision": float(ct[kt].mean())})
    e = [r["est_precision"] for r in rows]; g = [r["true_precision"] for r in rows]
    out["per_class_spearman"] = spearman(e, g)
    out["per_class_mae"] = float(np.mean(np.abs(np.array(e) - np.array(g)))) if rows else float("nan")
    out["per_class"] = rows
    return out


def summarize(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = sorted({int(m[1]) for d in a.root.iterdir() if (m := re.match(r".+-s(\d+)$", d.name))})
    per_seed = {}
    for k in seeds:
        r = {}
        sd = a.root / f"stages-s{k}"
        if (sd / "DONE").exists():
            z = {s: np.load(sd / f"stage_{s}.npz") for s in STAGES}
            mask, evaluable = split_mask(z[STAGES[0]]["image_ids"], a.split)
            y = z[STAGES[0]]["labels"][mask]
            for s in STAGES:
                assert (z[s]["image_ids"] == z[STAGES[0]]["image_ids"]).all()
            r["exp9"] = {s: exp9_stage(z[s]["probs"].astype(np.float64)[mask], y) for s in STAGES}
            r["exp10_agreement_after_mt"] = exp10_agreement(
                z["2_after_mt_student"]["probs"][mask], z["2_after_mt_teacher"]["probs"][mask], y, evaluable)
            ev = json.loads((sd / "eval_full.json").read_text())
            if "acrm_agree_c" in ev:
                # ACRM 的 EMA 一致率对照最终学生预测（原流程以学生预测）的逐类真实精度
                pred = z["2_after_mt_student"]["probs"][mask].argmax(1)
                prec = per_class_precision(pred, y, evaluable)
                cls = sorted(prec)
                ag = [ev["acrm_agree_c"][c] for c in cls]
                r["exp10_acrm_ema"] = {"mean_agreement": float(np.mean(ag)),
                                       "mean_true_precision": float(np.mean([prec[c] for c in cls])),
                                       "spearman": spearman(ag, [prec[c] for c in cls]), "n_classes": len(cls)}
        for tag in ("atc_source", "atc_adabn"):
            f = a.root / f"{tag}-s{k}" / "atc.npz"
            if f.exists():
                z = np.load(f, allow_pickle=False)
                mask, evaluable = split_mask(z["tgt_image_ids"], a.split)
                r[f"exp10_{tag}"] = exp10_estimators(z, mask, evaluable)
        per_seed[k] = r

    agg = {}
    first = next((r for r in per_seed.values() if "exp9" in r), None)
    if first:
        agg["exp9"] = {s: {m: summarize([per_seed[k]["exp9"][s][m] for k in per_seed if "exp9" in per_seed[k]])
                           for m in ("top1", "precision_at_0.9", "coverage_at_0.9", "top3_pred_share", "n_classes_predicted")}
                       for s in STAGES}
        agg["exp10_agreement_after_mt"] = {m: summarize([per_seed[k]["exp10_agreement_after_mt"][m] for k in per_seed
                                                         if "exp10_agreement_after_mt" in per_seed[k]])
                                           for m in ("mean_agreement", "mean_true_precision", "spearman")}
        if any("exp10_acrm_ema" in r for r in per_seed.values()):
            agg["exp10_acrm_ema"] = {m: summarize([r["exp10_acrm_ema"][m] for r in per_seed.values() if "exp10_acrm_ema" in r])
                                     for m in ("mean_agreement", "mean_true_precision", "spearman")}
    for tag in ("atc_source", "atc_adabn"):
        rs = [r[f"exp10_{tag}"] for r in per_seed.values() if f"exp10_{tag}" in r]
        if rs:
            agg[f"exp10_{tag}"] = {m: summarize([x[m] for x in rs])
                                   for m in ("true_acc", "AC", "DoC", "ATC_maxconf", "ATC_negent", "per_class_spearman", "per_class_mae")}
    a.out.write_text(json.dumps({"split": a.split, "seeds": seeds, "aggregate": agg, "per_seed": per_seed},
                                indent=1, ensure_ascii=False))
    print(json.dumps({"split": a.split, "seeds": seeds, "aggregate": agg}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
