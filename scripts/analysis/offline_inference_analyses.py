"""推理期参数的离线分析：Table 4（τ、γ 敏感性）、EXP-11（尾类重标定后归一化与否的 ECE）、
Table 3 案例、错误分析、Fig. 4 可靠性图与 Fig. 5 混淆矩阵的数据。全部在测试集上，6 seed。

    python scripts/analysis/offline_inference_analyses.py --out results/exp-offline-inference.json

起点：exp4/adabn_g0-s*/probs_w-o-ACRM.npz = 定稿 checkpoint 经 AdaBN、logit 调整 τ = 0.5、不做尾类重标定（γ = 0）的概率。
  - 换 τ'：p ∝ p · π^{−(τ'−0.5)}（softmax(z − τ log π) 的精确变换，π 为源域训练集类别频率）
  - 加 γ：p̃_c ∝ p_c · (1 + γ (n_max/n_c − 1) · 1[c ∈ tail])，与 src/modules/acrm.ACRM.rectify 一致
先核验：由 γ = 0 的概率按定稿 γ = 0.005 重算，须与 exp2-ours 的最终概率一致（float16 存储误差内）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_table1 as t1  # noqa: E402

R = ROOT / "results/remote"
SEEDS = range(6)
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMAS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.08]


def crop(name):
    return name.split("___")[0]


def pretty(name):
    """PlantVillage 类名 → 稿件写法：Apple___Cedar_apple_rust → Apple cedar apple rust；Tomato___Tomato_mosaic_virus → Tomato mosaic virus。"""
    special = {"Apple___Cedar_apple_rust": "Apple cedar rust",
               "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn gray leaf spot",
               "Corn_(maize)___Northern_Leaf_Blight": "Corn northern leaf blight",
               "Corn_(maize)___Common_rust_": "Corn common rust"}
    if name in special:
        return special[name]
    c, d = name.split("___")
    c = c.replace(",_", " ").replace("_", " ").replace(" (including sour)", "").replace(" (maize)", "").strip()
    d = d.replace("_", " ").strip().lower()
    if d.startswith(c.split()[0].lower()):
        return d[0].upper() + d[1:]
    return f"{c} {d}"


def train_counts(C):
    c = Counter(int(r["class_idx"]) for r in csv.DictReader(open(ROOT / "dataset/splits/plantvillage_lt_source.csv"))
                if r["split"] == "train")
    return np.array([max(c[i], 1) for i in range(C)], float)


def rectify(p, counts, tail, gamma, renorm=True):
    m = np.zeros(len(counts)); m[tail] = 1
    q = p * (1 + gamma * (counts.max() / counts - 1) * m)
    return q / q.sum(1, keepdims=True) if renorm else q


def retau(p, prior, tau):
    q = p * prior ** (-(tau - 0.5))
    return q / q.sum(1, keepdims=True)


def ece_conf(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    conf = np.minimum(conf, 1.0)   # 未归一化时最大"概率"可能超过 1：按 1 归入最后一箱（仅 EXP-11 对照用）
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ci = t1.load_class_index()
    conflict, per_class, _ = t1.target_meta()
    evaluable = [c for c in range(ci.num_classes) if per_class[c] >= t1.UNEVALUABLE_MIN_IMAGES]
    tail = [c for c in ci.tail_classes if c in evaluable]
    counts = train_counts(ci.num_classes); prior = counts / counts.sum()
    out = {"tail_classes": tail, "class_names": ci.names, "class_names_pretty": [pretty(n) for n in ci.names]}

    base, ids0, labels0 = {}, None, None
    for s in SEEDS:
        z = np.load(R / f"exp4/adabn_g0-s{s}/probs_w-o-ACRM.npz")
        base[s] = z["probs"].astype(np.float64); base[s] /= base[s].sum(1, keepdims=True)
        ids0, labels0 = [str(i) for i in z["image_ids"]], z["labels"]
    test = np.array([i in t1.TEST_IDS for i in ids0])

    # ---- 核验：γ = 0.005 重算 vs exp2-ours 最终概率
    diffs = []
    for s in SEEDS:
        ref = np.load(R / f"exp2-ours/ours-s{s}/bestval/probs_full.npz")
        assert [str(i) for i in ref["image_ids"]] == ids0
        mine = rectify(base[s], counts, tail, 0.005)
        diffs.append(float(np.abs(mine - ref["probs"].astype(np.float64)).max()))
        assert (mine.argmax(1) == ref["probs"].argmax(1)).mean() > 0.999
    out["check_reproduce_final_max_abs_diff"] = max(diffs)

    def m(p, s):
        return t1.metrics(p, labels0, ids0, tail=tail, evaluable=evaluable, conflict=conflict, drop_conflict=False)

    def agg(fn):
        rs = [fn(s) for s in SEEDS]
        return {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs], ddof=1))}
                for k in ("top1", "tail_recall", "macro_f1", "ece")}

    # ---- 同一开发集规则用于本方法的 λ（γ = 0.005 固定）：须选出报告配置所用的 0.5（与基线的 λ 选取对称，aggregate_table1 tuned 口径）
    def dev_eval(lam):
        return t1.metrics(rectify(retau(base[t1.SELECT_SEED], prior, lam), counts, tail, 0.005), labels0, ids0,
                          tail=tail, evaluable=evaluable, conflict=conflict, drop_conflict=False, split="dev")
    lam_ours, grid_ours = t1.select_by_dev(dev_eval)
    out["ours_dev_lambda_selection"] = {"lambda": lam_ours, "dev_grid_seed0": grid_ours}
    assert lam_ours == 0.5, f"开发集规则为本方法选出 λ = {lam_ours}，与报告配置的 0.5 不符"

    # ---- 尾类重标定与更大的 λ 可互换：γ = 0、λ = 0.75 对报告配置（λ = 0.5、γ = 0.005）的配对检验（稿件 4.6 节）
    from scipy import stats as _st
    fin = [m(rectify(base[s], counts, tail, 0.005), s) for s in SEEDS]
    alt = [m(retau(base[s], prior, 0.75), s) for s in SEEDS]
    out["rectification_vs_lambda"] = {"setting": "gamma=0, lambda=0.75 vs reported (lambda=0.5, gamma=0.005)", **{
        k: {"reported": float(np.mean([r[k] for r in fin])), "alternative": float(np.mean([r[k] for r in alt])),
            "p_paired_t": float(_st.ttest_rel([r[k] for r in fin], [r[k] for r in alt]).pvalue)}
        for k in ("top1", "tail_recall", "macro_f1", "ece")}}

    # ---- Table 4：τ（γ = 0.005 固定）与 γ（τ = 0.5 固定）
    out["table4_tau"] = {str(t): agg(lambda s, t=t: m(rectify(retau(base[s], prior, t), counts, tail, 0.005), s)) for t in TAUS}
    out["table4_gamma"] = {str(g): agg(lambda s, g=g: m(rectify(base[s], counts, tail, g), s)) for g in GAMMAS}

    # ---- EXP-11：γ = 0.005 与原稿 0.08 下，重标定后归一化与否的 ECE（测试集）
    e11 = {}
    for g in (0.005, 0.08):
        for renorm in (True, False):
            v = []
            for s in SEEDS:
                q = rectify(base[s], counts, tail, g, renorm)[test]
                y = labels0[test]
                v.append(ece_conf(q.max(1), (q.argmax(1) == y).astype(float)))
            e11[f"gamma={g}|{'renormalized' if renorm else 'unnormalized'}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
    out["exp11_ece_renormalization"] = e11

    # ---- Fig. 4 / Fig. 5 / 错误分析：6 seed 合并的测试集尾类样本
    def pooled(get):
        P, Y = [], []
        for s in SEEDS:
            p, y = get(s); P.append(p[test]); Y.append(y[test])
        return np.concatenate(P), np.concatenate(Y)
    ours = lambda s: (rectify(base[s], counts, tail, 0.005), labels0)

    def from_npz(f):
        return lambda s: (np.load(f.format(s=s))["probs"].astype(np.float64), labels0)
    so = from_npz(str(R / "exp2/source_only-s{s}/probs_bestval.npz"))
    focal_ad = lambda s: _focal_adabn(s, counts)
    rel, conf_mats = {}, {}
    for name, g in (("resnet50_source_only", so), ("focal_adabn_la", focal_ad), ("cd_apdm", ours)):
        p, y = pooled(g)
        tm = np.isin(y, tail)
        conf, corr = p[tm].max(1), (p[tm].argmax(1) == y[tm]).astype(float)
        edges = np.linspace(0, 1, 11); bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (conf > lo) & (conf <= hi)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(mm.sum()),
                         "acc": float(corr[mm].mean()) if mm.any() else None, "conf": float(conf[mm].mean()) if mm.any() else None})
        rel[name] = {"tail_ece_15bins": ece_conf(conf, corr), "bins10": bins, "n": int(tm.sum())}
        pred = p.argmax(1)
        M = np.zeros((len(tail), len(tail) + 1), int)
        for yi, pi in zip(y[tm], pred[tm]):
            M[tail.index(yi), tail.index(pi) if pi in tail else len(tail)] += 1
        conf_mats[name] = M.tolist()
    out["fig4_reliability_tail"] = rel
    out["fig5_confusion_tail"] = {"rows_true_tail": tail, "cols": tail + ["non-tail"], "counts": conf_mats}

    # 错误分析：CD-APDM 在尾类测试样本上最常见的错误去向（6 seed 合并）
    p, y = pooled(ours); tm = np.isin(y, tail); pred = p.argmax(1)
    errs = Counter((int(a), int(b)) for a, b in zip(y[tm], pred[tm]) if a != b)
    same_crop = sum(n for (a, b), n in errs.items() if crop(ci.names[a]) == crop(ci.names[b]))
    out["error_analysis_tail"] = {"n_tail_errors": int(sum(errs.values())), "n_tail": int(tm.sum()),
                                  "same_crop_share": same_crop / max(sum(errs.values()), 1),
                                  "to_non_tail_share": sum(n for (a, b), n in errs.items() if b not in tail) / max(sum(errs.values()), 1),
                                  "top": [{"true": pretty(ci.names[a]), "pred": pretty(ci.names[b]), "n": n} for (a, b), n in errs.most_common(8)]}

    # ---- Table 3：seed 0，测试集尾类样本，规则写死（与 Fig. 2 同一批图像）
    fd = np.load(R / "exp-final/figdata-s0/figdata.npz")
    so0 = np.load(R / "exp2/source_only-s0/probs_bestval.npz")["probs"].astype(np.float64)
    fa0 = _focal_adabn(0, counts)[0]; ou0 = rectify(base[0], counts, tail, 0.005)
    pos = {i: n for n, i in enumerate(ids0)}
    rows = []
    for iid, kind in zip(fd["cam_image_ids"], fd["cam_kind"]):
        k = pos[str(iid)]
        cell = lambda p: {"pred": pretty(ci.names[int(p[k].argmax())]), "conf": float(p[k].max())}
        rows.append({"image_id": str(iid), "kind": str(kind), "true": pretty(ci.names[int(labels0[k])]),
                     "resnet50": cell(so0), "focal_adabn_la": cell(fa0), "cd_apdm": cell(ou0)})
    out["table3_cases"] = rows
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k in ("check_reproduce_final_max_abs_diff", "ours_dev_lambda_selection", "rectification_vs_lambda", "table4_tau", "table4_gamma",
                                                           "exp11_ece_renormalization", "error_analysis_tail")},
                     indent=1, ensure_ascii=False))


def _focal_adabn(s, counts):
    """Focal Loss + AdaBN + LA（τ = 0.5）：与 aggregate_table1.load_probs 相同的组合。"""
    z = np.load(R / f"exp2-adabn/focal-s{s}/bestval.npz")
    logits = z["logits_adabn"].astype(np.float64) - 0.5 * np.log(z["class_counts"].clip(1) / z["class_counts"].clip(1).sum())
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return p, z["labels"]


if __name__ == "__main__":
    main()
