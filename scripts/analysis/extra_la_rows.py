"""补算 Swin-B 与 CoTTA 的"+ 调好的 post-hoc LA"行（2026-09-29 二审评估：审稿人会问为什么只有 ResNet-50 系基线有 LA 行）。

    ~/miniconda3/envs/agri-ctta/bin/python scripts/analysis/extra_la_rows.py

这两个基线的 run 只存了 softmax 概率；post-hoc logit 调整作用在 log p 上与作用在 logits 上等价
（两者每个样本只差一个常数）。先验取源域训练集类计数（与其余 LA 行相同），λ 按 aggregate_table1 的同一规则
在开发集 seed 0 上选取；与 CD-APDM 逐 seed 配对 t 检验，Holm 族 = Table 1 现有 17 行 + 这 2 行。
输出 results/exp-extra-la-rows.json（主口径：bestval、测试集、保留冲突图像）。只作决策参考，未进稿件。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_table1 as A  # noqa: E402

ROOT = A.ROOT
EXP2 = ROOT / "results/remote/exp2"
RULE = "bestval"


def main():
    ci = A.load_class_index()
    conflict, per_class, n_total = A.target_meta()
    evaluable = [c for c in range(ci.num_classes) if per_class[c] >= A.UNEVALUABLE_MIN_IMAGES]
    tail = [c for c in ci.tail_classes if c in evaluable]
    counts = np.load(ROOT / "results/remote/exp2-adabn/source_only-s0/bestval.npz")["class_counts"].astype(float)
    counts = np.where(counts <= 0, 1, counts)
    log_prior = np.log(counts / counts.sum())
    main_view = json.load(open(ROOT / "results/exp-2-table1.json"))["views"][0]
    assert main_view["rule"] == RULE and not main_view["drop_label_conflict"] and main_view["la_policy"] == "tuned"
    ours = main_view["table"]["ours"]

    def load(method, seed, lam):
        run = "cotta" if method == "cotta" else method
        z = np.load(EXP2 / f"{run}-s{seed}" / f"probs_{RULE}.npz", allow_pickle=False)
        lg = np.log(np.clip(z["probs"].astype(np.float64), 1e-12, None)) - lam * log_prior
        p = np.exp(lg - lg.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True), z["labels"], [str(i) for i in z["image_ids"]], EXP2 / f"{run}-s{seed}"

    out = {"rule": RULE, "prior": "source training class counts", "rows": {}}
    raw = {k: {m: v["p_t"] for m, v in ((key.split("|")[0], val) for key, val in main_view["paired_tests_vs_reference"].items()
                                          if key.endswith("|" + k) and "p_t" in val)} for k in A.METRICS}
    for method in ("swin_b", "cotta"):
        def dev_eval(l):
            p, y, ids, _ = load(method, A.SELECT_SEED, l)
            return A.metrics(p, y, ids, tail=tail, evaluable=evaluable, conflict=conflict, drop_conflict=False, split="dev")
        lam, grid = A.select_by_dev(dev_eval)
        per_seed = {}
        for seed in range(6):
            p, y, ids, run_dir = load(method, seed, lam)
            assert len(ids) == n_total
            m = A.metrics(p, y, ids, tail=tail, evaluable=evaluable, conflict=conflict, drop_conflict=False)
            m["src_val_top1"] = A.source_val_top1(run_dir, method, RULE)
            m["cross_domain_gap"] = m["src_val_top1"] - m["top1"]
            per_seed[seed] = m
        row = {"lambda": lam, "dev_grid_seed0": grid}
        for k in A.METRICS:
            v = [per_seed[s][k] for s in range(6)]
            a = np.array([ours[k]["per_seed"][str(s)] for s in range(6)])
            t = stats.ttest_rel(a, np.array(v))
            raw[k][f"{method}+la"] = float(t.pvalue)
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "ci95": A.ci95(v),
                      "per_seed": dict(zip(range(6), v)), "ours_minus_row": float((a - np.array(v)).mean()),
                      "p_t": float(t.pvalue)}
        out["rows"][f"{method}+la"] = row
    for k in A.METRICS:
        h = A.holm(raw[k])
        for name in out["rows"]:
            out["rows"][name][k]["p_t_holm_19rows"] = h[name]
    (ROOT / "results/exp-extra-la-rows.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for name, r in out["rows"].items():
        print(f"{name}  λ = {r['lambda']:g}")
        for k in ("top1", "tail_recall", "macro_f1", "ece"):
            q = r[k]
            print(f"   {k:<12} {q['mean']:.3f} ± {q['std']:.3f}   ours−row {q['ours_minus_row']:+.3f}   "
                  f"p = {q['p_t']:.3g}  Holm(19) = {q['p_t_holm_19rows']:.3g}")


if __name__ == "__main__":
    main()
