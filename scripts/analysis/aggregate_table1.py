"""EXP-2 → Table 1：跨 seed 汇总与显著性检验。

    python scripts/analysis/aggregate_table1.py results/remote/exp2 results/remote/exp2-ours \
        [results/remote/exp2-adabn] --out results/exp-2-table1.json [--md results/exp-2-table1.md]

所有指标都从各 run 保存的**逐图目标域概率**重算，而不是读训练时打印的数字 ——
这样评测口径（下列两个开关）改变时无需重跑任何实验，且全部方法走同一段代码。

选型规则（fix-experiment-protocol，2026-09-21 定）
    bestval  主结果：源域验证集 Top-1 最高的 checkpoint
    final    附录：最后一个 epoch
CoTTA 本身不选 epoch，其起点随规则走（cotta-s* 起于 bestval，cotta_final-s* 起于 final）。

Cross-Domain Gap（稿件 4.2.3，EDIT-27e）
    = 所选 checkpoint 的源域验证 Top-1 − 目标域测试集 Top-1，逐 run 计算后再跨 seed 汇总。
    源域验证 Top-1 取自训练时记录：基线读 summary.json 的 selection，Ours 读 selection.json
    （bestval）或 train_history.json 末个 epoch（final）。推理期处理（AdaBN、logit 调整、ACRM）
    只作用于目标域，源域一侧始终是适配前的同一 checkpoint。

"基线 + LA"与"基线 + AdaBN + LA"行（Open Decision 3；2026-09-28 起 λ 按开发集选取）
    exp2-adabn/<方法>-s<seed>/{bestval,final}.npz 存 AdaBN 前后的 logits（logits_plain / logits_adabn），
    本脚本对二者分别做 post-hoc logit 调整 z − λ log π 得到概率（不加 ACRM 尾类重标定）；源域一侧取 exp2 中同一 run。
    λ 的两种口径（views 中的 la_policy）：
      tuned   主口径。每个方法、每种变体各自在**目标域开发集**、seed 0 上按选 γ 的同一规则选 λ：
              网格 0–4、步长 0.25；Macro-F1 最高者及与之相差 0.5 以内的视为并列，取其中尾类召回最高者
              （补充材料 S7）。同一规则用于 CD-APDM（γ = 0.005 固定）选出 λ = 0.5，即报告配置所用值，本脚本核验这一点。
              此前基线一律用 λ = 0.5 而本方法的 γ 在开发集上调过，是不对称的比较（审稿答复评估，2026-09-28）。
      shared  λ 一律取 0.5（本方法的取值），即此前的比较口径，保留作对照。
    Swin-B 与 CoTTA 的"+ LA"行（2026-09-30 加入 Table 1，二审满意度评估）：这两个 run 只存了 softmax 概率，
    post-hoc LA 在 log p 上做（与在 logits 上做等价，每个样本只差一个常数）；先验取源域训练集类计数
    （由划分文件统计，与 exp2-adabn 文件中的 class_counts 逐类一致）。CoTTA 的 bestval / final 两个概率文件内容相同。
    Swin-B 用 LayerNorm，CoTTA 本身已更新 BN 统计量，两者都没有"+ AdaBN + LA"行。

评测口径
    **只在目标域测试集（1,784 张）上计算**（决策记录 6：开发集 782 张的标签仅用于方法开发，
    不进入任何报告数字）；自适应仍使用全部目标图且不带标签
    主口径     测试集全部图像；类别平均指标排除"不可评测类别"
               （目标域只有 2 张的 Tomato spider mites，dataset/splits/README.md）
    敏感性口径 另外剔除 66 张 label_conflict 图像（同一图像在 PlantDoc 中带两个
               不同标签，任何模型的准确率上限因此为 98.71%）

统计（回应 R1-3：原稿 3 次运行、检验自由度 2）
    每个方法 n 个 seed，报告 均值 ± 样本标准差 与 t 分布 95% CI（df = n−1）。
    Ours 对每个基线做**按 seed 配对**的 t 检验（df = n−1，主检验）与 Wilcoxon 符号秩检验，
    每个指标内对 7 个基线做 Holm 校正。配对的依据：同一 seed 下各方法共用数据顺序与初始化种子。
    注意：n=6 时 Wilcoxon 双侧 p 的下限为 2/2^6 = 0.03125，经 7 重 Holm 校正后不可能显著；
    它在 n=6 时只作描述，要让它有检验力须 n ≥ 10（下限 0.00195）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from src.datasets.splits import load_class_index  # noqa: E402
from src.utils.metrics import expected_calibration_error, macro_f1  # noqa: E402

METHOD_ORDER = ["source_only", "focal", "dann", "wb_dann", "flexmatch", "swin_b", "cotta", "ours"]
DISPLAY = {"source_only": "ResNet-50 (source only)", "focal": "Focal Loss", "dann": "DANN",
           "wb_dann": "WB + DANN", "flexmatch": "FlexMatch", "swin_b": "Swin-B",
           "cotta": "CoTTA", "ours": "CD-APDM (Ours)"}
METRICS = ["top1", "tail_recall", "macro_f1", "ece", "cross_domain_gap"]
HIGHER_BETTER = {"top1": True, "tail_recall": True, "macro_f1": True, "ece": False, "cross_domain_gap": False}
ADABN_SUFFIX = "+adabn_la"   # "基线 + AdaBN + LA"行的方法名后缀
PLAIN_SUFFIX = "+la"         # "基线 + LA"行（不加 AdaBN，只做 post-hoc logit 调整）的方法名后缀
LA_TAU = 0.5                 # 与 code/configs/cd_apdm.yaml 的 logit 调整 τ 一致（shared 口径）
LA_GRID = [i / 4 for i in range(17)]   # tuned 口径的 λ 网格：0–4，步长 0.25
TIE_MF1 = 0.5                # Macro-F1 并列容差（与选 γ 的规则相同，补充材料 S7）
SELECT_SEED = 0              # 开发集选取只用 seed 0（与选 γ 相同）
PROB_LA_METHODS = ("swin_b", "cotta")   # 只存概率、在 log p 上补做 post-hoc LA 的基线
UNEVALUABLE_MIN_IMAGES = 5   # 目标域图像少于此数的类别不计入类别平均指标


# ------------------------------------------------------------------ 定位产物
def probs_path(run_dir: Path, method: str, rule: str) -> Path:
    if method == "ours":
        return run_dir / rule / "probs_full.npz"
    if method == "cotta":
        return run_dir / "probs_final.npz"      # 测试时方法：单一输出，规则体现在起点
    return run_dir / f"probs_{rule}.npz"


def source_val_top1(run_dir: Path, method: str, rule: str) -> float:
    """所选 checkpoint 的源域验证 Top-1（训练时记录，见文件头 Cross-Domain Gap 一节）。"""
    if method == "ours":
        if rule == "bestval":
            return float(json.loads((run_dir / "selection.json").read_text())["best_source_val"]["src_top1"])
        return float(json.loads((run_dir / "train_history.json").read_text())[-1]["src_top1"])
    sel = json.loads((run_dir / "summary.json").read_text())["selection"]
    return float(sel["best_source_val" if rule == "bestval" else "final"]["src_val"]["top1"])


def discover(roots: list[Path], rule: str) -> dict[str, dict[int, tuple[Path, Path]]]:
    """返回 {method: {seed: (probs 文件, 读源域验证 Top-1 的 run 目录)}}；只收完成（有 DONE）且非冒烟的任务。

    exp2-adabn 队列的产物（<方法>-s<seed>/<规则>.npz）记为 "<方法>+adabn_la"，源域一侧指向 exp2 中同名 run。
    """
    found: dict[str, dict[int, tuple[Path, Path]]] = defaultdict(dict)
    pat = re.compile(r"^(?P<m>[a-z_]+)-s(?P<s>\d+)$")
    exp2_dirs = {d.name: d for root in roots for d in root.iterdir() if d.is_dir() and (d / "summary.json").exists()}
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            mm = pat.match(d.name)
            if not mm or not (d / "DONE").exists():
                continue
            name, seed = mm["m"], int(mm["s"])
            if (d / f"{rule}.npz").exists() and not (d / "summary.json").exists():
                if d.name not in exp2_dirs:
                    raise SystemExit(f"{d}：找不到对应的 exp2 run（源域验证 Top-1 从那里读），把 exp2 目录一并传入")
                for suffix in (PLAIN_SUFFIX, ADABN_SUFFIX):
                    method = name + suffix
                    if seed in found[method]:
                        raise SystemExit(f"重复的 run：{method} seed {seed}")
                    found[method][seed] = (d / f"{rule}.npz", exp2_dirs[d.name])
                continue
            if name == "cotta_final":
                if rule != "final":
                    continue
                method = "cotta"
            elif name == "cotta":
                if rule != "bestval":
                    continue
                method = "cotta"
            else:
                method = name
            summ = d / "summary.json"
            if summ.exists() and json.loads(summ.read_text()).get("smoke"):
                continue
            p = probs_path(d, method, rule)
            if p.exists():
                if seed in found[method]:
                    raise SystemExit(f"重复的 run：{method} seed {seed}（{found[method][seed][0]} 与 {p}）")
                found[method][seed] = (p, d)
                if method in PROB_LA_METHODS:           # 只存概率的基线：在 log p 上做 post-hoc LA
                    found[method + PLAIN_SUFFIX][seed] = (p, d)
    return found


def source_class_counts() -> np.ndarray:
    """源域训练集各类图像数（LA 的先验），由划分文件统计。"""
    ci = load_class_index()
    counts = np.zeros(ci.num_classes)
    for r in csv.DictReader(open(ROOT / "dataset/splits/plantvillage_lt_source.csv", newline="")):
        if r["split"] == "train":
            counts[int(r["class_idx"])] += 1
    return counts


def load_probs(path: Path, method: str = "", lam: float = LA_TAU):
    """返回 (probs, labels, image_ids)；exp2-adabn 的 logits 文件在此做 λ 的 post-hoc logit 调整
    （方法名以 +la 结尾用 AdaBN 前的 logits，以 +adabn_la 结尾用 AdaBN 后的 logits）。
    只存概率的基线（Swin-B、CoTTA）的 +la 行在 log p 上做同样的调整。"""
    z = np.load(path, allow_pickle=False)
    if "probs" in z.files:
        if not method.endswith(PLAIN_SUFFIX):
            return z["probs"], z["labels"], z["image_ids"]
        counts = source_class_counts()
        lg = np.log(np.clip(z["probs"].astype(np.float64), 1e-12, None)) - lam * np.log(counts / counts.sum())
        p = np.exp(lg - lg.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True), z["labels"], z["image_ids"]
    kind = "logits_plain" if method.endswith(PLAIN_SUFFIX) else "logits_adabn"
    counts = np.where(z["class_counts"] <= 0, 1, z["class_counts"]).astype(float)
    logits = z[kind].astype(np.float64) - lam * np.log(counts / counts.sum())
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True), z["labels"], z["image_ids"]


def select_by_dev(evaluate):
    """选 γ 的同一规则（补充材料 S7）：Macro-F1 最高者及与之相差 TIE_MF1 以内的视为并列，取尾类召回最高者
    （再并列取较小的 λ）。evaluate(λ) 返回开发集指标。返回 (λ, 网格上的开发集指标)。"""
    grid = [(lam, evaluate(lam)) for lam in LA_GRID]
    best = max(r["macro_f1"] for _, r in grid)
    tied = [(lam, r) for lam, r in grid if r["macro_f1"] >= best - TIE_MF1]
    lam = max(tied, key=lambda x: (x[1]["tail_recall"], -x[0]))[0]
    return lam, [{"lambda": l, **{k: r[k] for k in ("top1", "tail_recall", "macro_f1")}} for l, r in grid]


# ------------------------------------------------------------------ 指标
def target_meta():
    rows = list(csv.DictReader(open(ROOT / "dataset/splits/plantdoc_target.csv", newline="")))
    conflict = {r["path"] for r in rows if "label_conflict" in r["leak_flags"]}
    if rows and "eval_split" in rows[0]:
        global TEST_IDS
        TEST_IDS = {r["path"] for r in rows if r["eval_split"] == "test"}
        global DEV_IDS
        DEV_IDS = {r["path"] for r in rows if r["eval_split"] == "dev"}
    per_class = defaultdict(int)
    for r in rows:
        per_class[int(r["class_idx"])] += 1
    return conflict, per_class, len(rows)


TEST_IDS = None   # 由 target_meta() 从划分文件读取
DEV_IDS = None


def metrics(probs, labels, ids, *, tail, evaluable, conflict, drop_conflict, test_only=True, split="test"):
    keep = np.ones(len(labels), bool)
    if test_only:
        if TEST_IDS is None:
            raise SystemExit("划分文件缺 eval_split 列：先运行 build_target_devtest.py")
        keep &= np.array([i in (TEST_IDS if split == "test" else DEV_IDS) for i in ids])
    if drop_conflict:
        keep &= np.array([i not in conflict for i in ids])
    p, y = probs[keep].astype(np.float64), labels[keep]
    pred = p.argmax(-1)
    rec = {c: float((pred[y == c] == c).mean()) for c in evaluable if (y == c).any()}
    tails = [rec[c] for c in tail if c in rec]
    return {
        "top1": float((pred == y).mean()) * 100,
        "tail_recall": float(np.mean(tails)) * 100,
        "macro_f1": macro_f1(pred, y, probs.shape[1], classes=evaluable) * 100,
        "ece": expected_calibration_error(p, y),
        "n_images": int(keep.sum()),
    }


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean() - h), float(x.mean() + h)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ------------------------------------------------------------------ 主流程
def base_name(method: str) -> str:
    return method.removesuffix(ADABN_SUFFIX).removesuffix(PLAIN_SUFFIX)


def aggregate(roots, rule, drop_conflict, la_policy="tuned", ref="ours"):
    ci = load_class_index()
    conflict, per_class, n_total = target_meta()
    evaluable = [c for c in range(ci.num_classes) if per_class[c] >= UNEVALUABLE_MIN_IMAGES]
    tail = [c for c in ci.tail_classes if c in evaluable]
    found = discover(roots, rule)

    # λ：tuned 口径下每个"+LA"变体在开发集 seed 0 上选取（开发集指标与报告口径无关，一律保留冲突图像）
    lam, selection = {}, {}
    for method, seeds in found.items():
        if not method.endswith((PLAIN_SUFFIX, ADABN_SUFFIX)):
            continue
        if la_policy == "shared":
            lam[method] = LA_TAU
            continue
        path = seeds[SELECT_SEED][0]
        def dev_eval(l, path=path, method=method):
            p, y, ids = load_probs(path, method, l)
            return metrics(p, y, [str(i) for i in ids], tail=tail, evaluable=evaluable,
                           conflict=conflict, drop_conflict=False, split="dev")
        lam[method], grid = select_by_dev(dev_eval)
        selection[method] = {"lambda": lam[method], "dev_grid_seed0": grid}

    per_run = defaultdict(dict)
    for method, seeds in found.items():
        base = base_name(method)
        for seed, (path, run_dir) in sorted(seeds.items()):
            probs, labels, image_ids = load_probs(path, method, lam.get(method, LA_TAU))
            ids = [str(i) for i in image_ids]
            if len(ids) != n_total:
                raise SystemExit(f"{path}：{len(ids)} 张，应为全部 {n_total} 张目标域图像")
            m = metrics(probs, labels, ids, tail=tail, evaluable=evaluable,
                        conflict=conflict, drop_conflict=drop_conflict)
            m["src_val_top1"] = source_val_top1(run_dir, base, rule)
            m["cross_domain_gap"] = m["src_val_top1"] - m["top1"]
            per_run[method][seed] = m

    table = {}
    order = ([m for m in METHOD_ORDER if m != "ours"] + [m + PLAIN_SUFFIX for m in METHOD_ORDER]
             + [m + ADABN_SUFFIX for m in METHOD_ORDER] + ["ours"])
    for method in [m for m in order if m in per_run] + sorted(set(per_run) - set(order)):
        runs = per_run[method]
        row = {"seeds": sorted(runs), "n": len(runs)}
        for k in METRICS:
            v = [runs[s][k] for s in sorted(runs)]
            row[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                      "ci95": ci95(v), "per_seed": dict(zip(sorted(runs), v))}
        table[method] = row

    tests = {}
    if ref in per_run:
        raw_t, raw_w = {}, {}
        for method in table:
            if method == ref:
                continue
            common = sorted(set(per_run[ref]) & set(per_run[method]))
            for k in METRICS:
                a = np.array([per_run[ref][s][k] for s in common])
                b = np.array([per_run[method][s][k] for s in common])
                d = a - b
                entry = {"n_pairs": len(common), "df": len(common) - 1,
                         "mean_diff": float(d.mean()) if len(d) else float("nan")}
                if len(common) >= 2 and np.any(d != 0):
                    t = stats.ttest_rel(a, b)
                    entry.update(t=float(t.statistic), p_t=float(t.pvalue))
                    raw_t[(method, k)] = float(t.pvalue)
                    try:
                        w = stats.wilcoxon(a, b)
                        entry["p_wilcoxon"] = float(w.pvalue)
                        raw_w[(method, k)] = float(w.pvalue)
                    except ValueError:
                        pass
                tests[f"{method}|{k}"] = entry
        # Holm 的比较族 = 同一指标下 Ours 对 Table 1 其余各行（含"基线 + LA""基线 + AdaBN + LA"行），各指标分别校正
        for k in METRICS:
            for (method, _), p in holm({mk: v for mk, v in raw_t.items() if mk[1] == k}).items():
                tests[f"{method}|{k}"]["p_t_holm"] = p
            for (method, _), p in holm({mk: v for mk, v in raw_w.items() if mk[1] == k}).items():
                tests[f"{method}|{k}"]["p_wilcoxon_holm"] = p

    return {
        "rule": rule, "drop_label_conflict": drop_conflict, "la_policy": la_policy,
        "la_grid": LA_GRID if la_policy == "tuned" else [LA_TAU], "la_selection": selection,
        "evaluable_classes": len(evaluable), "excluded_classes": [c for c in range(ci.num_classes) if c not in evaluable],
        "tail_classes": tail, "reference": ref, "table": table, "paired_tests_vs_reference": tests,
    }


def to_markdown(res) -> str:
    L = [f"### 选型规则 `{res['rule']}`，{'剔除' if res['drop_label_conflict'] else '保留'} label_conflict 图像，λ 口径 `{res['la_policy']}`",
         "", "| 方法 | n | Top-1 | 95% CI | Tail Recall | Macro-F1 | ECE | Cross-Domain Gap | Ours−本行 Top-1（p, Holm） | Ours−本行 Tail（p, Holm） |",
         "|---|---:|---|---|---|---|---|---|---|---|"]
    for m, r in res["table"].items():
        f = lambda k: f"{r[k]['mean']:.2f} ± {r[k]['std']:.2f}" if k != "ece" else f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}"
        lo, hi = r["top1"]["ci95"]
        def cmp(k):
            t = res["paired_tests_vs_reference"].get(f"{m}|{k}")
            return f"{t['mean_diff']:+.2f}（{t.get('p_t_holm', float('nan')):.3g}）" if t else "—"
        base = base_name(m)
        lam = res["la_selection"].get(m, {}).get("lambda", LA_TAU)
        name = DISPLAY.get(base, base) + (f" + AdaBN + LA（λ={lam:g}）" if m.endswith(ADABN_SUFFIX)
                                          else f" + LA（λ={lam:g}）" if m.endswith(PLAIN_SUFFIX) else "")
        L.append(f"| {name} | {r['n']} | {f('top1')} | [{lo:.2f}, {hi:.2f}] | "
                 f"{f('tail_recall')} | {f('macro_f1')} | {f('ece')} | {f('cross_domain_gap')} | {cmp('top1')} | {cmp('tail_recall')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path, help="qfetch 取回的队列目录")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    # views[0] 为主口径（bestval、保留冲突图像、λ 按开发集选取），下游脚本默认读它
    results = [aggregate(args.roots, rule, drop, pol) for pol in ("tuned", "shared")
               for rule in ("bestval", "final") for drop in (False, True)]
    args.out.write_text(json.dumps({"exp": "EXP-2 Table 1", "views": results}, indent=2, ensure_ascii=False))
    md = "\n".join(to_markdown(r) for r in results)
    if args.md:
        args.md.write_text("# EXP-2 Table 1 汇总（由 aggregate_table1.py 生成，勿手改）\n\n" + md)
    print(md)


if __name__ == "__main__":
    main()
