"""补充材料 Table S1：PlantDoc ↔ PlantVillage 逐类映射与各类样本数（审稿意见 R1-1 要求的 exact class mappings）。

    python scripts/analysis/make_mapping_table.py

映射取自 results/exp-1a-class-stats.json（overlap.PlantVillage_x_PlantDoc.mapping）；
源域训练数、分桶取自 dataset/splits/plantvillage_lt_source.csv（split = train）；
目标域开发 / 测试数取自 dataset/splits/plantdoc_target.csv（eval_split）。输出 Markdown 表，供粘贴进补充材料 S1。
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ov = json.load(open(ROOT / "results/exp-1a-class-stats.json"))["overlap"]["PlantVillage_x_PlantDoc"]
    mapping = ov["mapping"]                       # PlantDoc 类名 -> PlantVillage 类名
    src = list(csv.DictReader(open(ROOT / "dataset/splits/plantvillage_lt_source.csv")))
    tgt = list(csv.DictReader(open(ROOT / "dataset/splits/plantdoc_target.csv")))
    idx, bucket = {}, {}
    train = Counter()
    for r in src:
        idx[r["class"]] = int(r["class_idx"]); bucket[r["class"]] = r["bucket"]
        if r["split"] == "train":
            train[r["class"]] += 1
    ev = defaultdict(Counter)
    for r in tgt:
        ev[r["plantvillage_class"]][r["eval_split"]] += 1
    assert len(mapping) == 28 and set(mapping.values()) <= set(idx)
    pv = lambda n: n.replace("___", ": ").replace("_", " ").replace(",", "").replace("  ", " ").strip()
    rows = sorted(mapping.items(), key=lambda kv: idx[kv[1]])
    print("| # | PlantDoc class | PlantVillage class | Bucket | Source training images | Target images (dev / test) |")
    print("|---|---|---|---|---:|---|")
    for d, v in rows:
        e = ev[v]
        print(f"| {idx[v]} | {d} | {pv(v)} | {bucket[v].capitalize()} | {train[v]:,} | {e['dev'] + e['test']} ({e['dev']} / {e['test']}) |")
    print()
    print("Discarded PlantVillage classes (no PlantDoc counterpart): " + "; ".join(pv(c) for c in ov["plantvillage_only"]) + ".")
    print(f"\n<!-- 合计：训练 {sum(train[v] for v in mapping.values()):,}，目标 {sum(sum(ev[v].values()) for v in mapping.values()):,} -->")


if __name__ == "__main__":
    main()
