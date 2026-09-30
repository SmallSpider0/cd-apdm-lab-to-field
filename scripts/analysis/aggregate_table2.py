"""EXP-3 + EXP-4 → Table 2（消融，含负面组件）：跨 seed 汇总与配对检验。

    python scripts/analysis/aggregate_table2.py results/remote --out results/exp-3-4-table2.json --md results/exp-3-4-table2.md

评测口径、指标、统计与 Table 1 完全相同（直接复用 aggregate_table1 的函数）：测试集 1,784 张、
类别平均指标排除不可评测类别、6 seed 配对 t 检验（df = 5）、每个指标内对全部消融行 Holm 校正。
只用主选型规则 bestval（EXP-4 只在 bestval checkpoint 上做；EXP-3 的 final 结果另见 json）。

各行（均与定稿 CD-APDM 只差一个组件，稿件 4.2.4）：
    Ours                        exp2-ours/ours-s*/bestval/probs_full.npz
    w/o MDIWM / w/o CIWLM       exp3/wo_{mdiwm,ciwlm}-s*/bestval/probs_full.npz（训练期，重训）
    w/o AdaBN                   exp4/noadapt-s*/probs_full.npz（不重估 BN；LA 与 γ 保留）
    w/o ACRM rectification      exp4/adabn_g0-s*/probs_w-o-ACRM.npz（γ = 0）
    + Mean-Teacher              exp4/neg_mt-s*（负面组件，伪标签阈值 0.9）
    + Mean-Teacher + CC-GANM    exp4/neg_ccganm-s*
    + Mean-Teacher + SS-PLAM    exp4/neg_ssplam-s*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_table1 as t1  # noqa: E402

ROWS = [
    ("ours", "CD-APDM (Ours)", "exp2-ours/ours-s{s}/{rule}/probs_full.npz"),
    ("wo_mdiwm", "w/o MDIWM", "exp3/wo_mdiwm-s{s}/{rule}/probs_full.npz"),
    ("wo_ciwlm", "w/o CIWLM", "exp3/wo_ciwlm-s{s}/{rule}/probs_full.npz"),
    ("noadapt", "w/o AdaBN", "exp4/noadapt-s{s}/probs_full.npz"),
    ("adabn_g0", "w/o ACRM rectification (γ = 0)", "exp4/adabn_g0-s{s}/probs_w-o-ACRM.npz"),
    ("neg_mt", "+ Mean-Teacher", "exp4/neg_mt-s{s}/probs_full.npz"),
    ("neg_ccganm", "+ Mean-Teacher + CC-GANM", "exp4/neg_ccganm-s{s}/probs_full.npz"),
    ("neg_ssplam", "+ Mean-Teacher + SS-PLAM", "exp4/neg_ssplam-s{s}/probs_full.npz"),
]
METRICS = ["top1", "tail_recall", "macro_f1", "ece"]


def aggregate(root: Path, drop_conflict: bool, rule: str = "bestval"):
    ci = t1.load_class_index()
    conflict, per_class, n_total = t1.target_meta()
    evaluable = [c for c in range(ci.num_classes) if per_class[c] >= t1.UNEVALUABLE_MIN_IMAGES]
    tail = [c for c in ci.tail_classes if c in evaluable]
    per_run = {}
    for key, _, pat in ROWS:
        runs = {}
        for s in range(6):
            f = root / pat.format(s=s, rule=rule)
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            ids = [str(i) for i in z["image_ids"]]
            if len(ids) != n_total:
                raise SystemExit(f"{f}：{len(ids)} 张，应为 {n_total}")
            runs[s] = t1.metrics(z["probs"], z["labels"], ids, tail=tail, evaluable=evaluable,
                                 conflict=conflict, drop_conflict=drop_conflict)
        if runs:
            per_run[key] = runs
    table, tests = {}, {}
    for key, name, _ in ROWS:
        if key not in per_run:
            continue
        runs = per_run[key]
        row = {"name": name, "seeds": sorted(runs), "n": len(runs)}
        for k in METRICS:
            v = [runs[s][k] for s in sorted(runs)]
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                      "ci95": t1.ci95(v), "per_seed": dict(zip(sorted(runs), v))}
        table[key] = row
    raw = {}
    for key in table:
        if key == "ours":
            continue
        common = sorted(set(per_run["ours"]) & set(per_run[key]))
        for k in METRICS:
            a = np.array([per_run["ours"][s][k] for s in common]); b = np.array([per_run[key][s][k] for s in common])
            e = {"n_pairs": len(common), "df": len(common) - 1, "mean_diff_ours_minus_row": float((a - b).mean())}
            if len(common) >= 2 and np.any(a != b):
                r = stats.ttest_rel(a, b); e.update(t=float(r.statistic), p_t=float(r.pvalue)); raw[(key, k)] = float(r.pvalue)
            tests[f"{key}|{k}"] = e
    for k in METRICS:
        for (key, _), p in t1.holm({mk: v for mk, v in raw.items() if mk[1] == k}).items():
            tests[f"{key}|{k}"]["p_t_holm"] = p
    return {"rule": rule, "drop_label_conflict": drop_conflict, "table": table, "paired_tests_vs_ours": tests}


def to_markdown(res):
    L = [f"### 选型规则 `{res['rule']}`，{'剔除' if res['drop_label_conflict'] else '保留'} label_conflict 图像", "",
         "| 设定 | n | Top-1 | Tail Recall | Macro-F1 | ECE | Ours−本行 Top-1（Holm p） | Ours−本行 Tail（Holm p） |",
         "|---|---:|---|---|---|---|---|---|"]
    for key, r in res["table"].items():
        f = lambda k: f"{r[k]['mean']:.2f} ± {r[k]['std']:.2f}" if k != "ece" else f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}"
        def cmp(k):
            t = res["paired_tests_vs_ours"].get(f"{key}|{k}")
            return f"{t['mean_diff_ours_minus_row']:+.2f}（{t.get('p_t_holm', float('nan')):.3g}）" if t else "—"
        L.append(f"| {r['name']} | {r['n']} | {f('top1')} | {f('tail_recall')} | {f('macro_f1')} | {f('ece')} | {cmp('top1')} | {cmp('tail_recall')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="results/remote")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()
    views = [aggregate(a.root, drop) for drop in (False, True)]
    a.out.write_text(json.dumps({"exp": "EXP-3 + EXP-4 Table 2", "views": views}, indent=2, ensure_ascii=False))
    md = "\n".join(to_markdown(v) for v in views)
    if a.md:
        a.md.write_text("# Table 2 消融汇总（由 aggregate_table2.py 生成，勿手改）\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()
