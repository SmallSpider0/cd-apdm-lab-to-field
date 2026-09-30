#!/usr/bin/env python3
"""把目标域 PlantDoc 划分为开发集 / 测试集（作者决策 2026-09-22，fix-experiment-protocol 决策记录 6）。

为什么需要
  修复实现缺陷后，CD-APDM 的伪标签自训练在本基准上仍坍缩（results/exp-2-diag-s0.json）。
  作者授权修订方法。在测试集上反复试方法等于在测试集上调参，因此划出一个开发集：
  其标签**只**用于方法设计与超参选择；测试集只在最终报告时使用一次。

规则
  * 自适应仍使用**全部**目标域图像且不带标签（transductive，与基线相同）；本划分只决定
    指标在哪个子集上计算，因此已完成的基线运行无需重跑，指标由保存的逐图概率重算。
  * 按类分层，开发集约占 DEV_FRACTION；以**组**为单位分配：近重复组与标签冲突组整组进入
    同一侧（组可能跨类，只有当组内涉及的每个类的开发集份额都不超过上限时才进开发集）。
  * 固定种子，写入 plantdoc_target.csv 的 eval_split 列（dev / test）。
"""
import collections, csv, json, math, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "dataset/splits/plantdoc_target.csv"
SEED = "20260922-target-devtest"
DEV_FRACTION = 0.30


def main():
    rows = list(csv.DictReader(open(SPLIT, newline="")))
    fields = list(rows[0].keys())
    if "eval_split" not in fields:
        fields.append("eval_split")
    per_class = collections.Counter(r["class_idx"] for r in rows)
    cap = {c: math.ceil(n * DEV_FRACTION) for c, n in per_class.items()}

    units = collections.defaultdict(list)            # 组（或单张）→ 成员
    for i, r in enumerate(rows):
        units[r["dup_group"] or f"single-{i}"].append(i)
    keys = sorted(units)
    random.Random(SEED).shuffle(keys)

    dev_count = collections.Counter()
    assign = {}
    for k in keys:
        need = collections.Counter(rows[i]["class_idx"] for i in units[k])
        if all(dev_count[c] + n <= cap[c] for c, n in need.items()):
            dev_count.update(need)
            side = "dev"
        else:
            side = "test"
        for i in units[k]:
            assign[i] = side
    for i, r in enumerate(rows):
        r["eval_split"] = assign[i]

    # 自检：任何组都不跨越 dev / test
    for k, members in units.items():
        assert len({assign[i] for i in members}) == 1, k
    with open(SPLIT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    cnt = collections.Counter(r["eval_split"] for r in rows)
    by_class = {c: {"dev": dev_count[c], "test": per_class[c] - dev_count[c]} for c in sorted(per_class, key=int)}
    conflict = collections.Counter(r["eval_split"] for r in rows if "label_conflict" in r["leak_flags"])
    meta = {
        "task": "fix-experiment-protocol 决策记录 6：目标域开发集 / 测试集",
        "seed": SEED, "dev_fraction_target": DEV_FRACTION,
        "rule": "按类分层、以近重复/标签冲突组为单位整组分配；开发集标签仅用于方法设计与超参选择，测试集只在最终报告时使用",
        "adaptation_data": "自适应仍使用全部目标域图像且不带标签（与基线相同）；划分只决定指标计算的子集",
        "counts": dict(cnt), "label_conflict_images": dict(conflict),
        "min_test_per_class": min(v["test"] for v in by_class.values()),
        "per_class": by_class,
    }
    (REPO / "results/exp-protocol-6-target-devtest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({k: meta[k] for k in ("counts", "label_conflict_images", "min_test_per_class")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
