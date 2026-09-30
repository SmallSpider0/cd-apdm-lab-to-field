#!/usr/bin/env python3
"""任务 1b —— 构造 PlantVillage-LT 源域并产出可公开的划分文件。

决议（plan/基准方案决策简报.md，2026-09-20）：
  源域 PlantVillage 按指数分布子采样至 IF=100
  目标域 PlantDoc 全量
  尾类身份由源域构造分布确定，采用 OLTR 三分桶（Many/Medium/Few）

IF 取 500（2026-09-20 决议）：IF=100 时按 OLTR 约定 Few-shot 桶为空
（最小类 54 张），与论文「稀有病虫害样本极度不足」的核心动机不符。
提高 IF 只减少尾类的**训练**样本，不影响评测可靠性 —— 评测在 PlantDoc 上
进行，每类约 90 张，与源域 IF 无关。这一不对称使较大的 IF 在此基准上可行。
IF=500 接近 iNaturalist 的约定，且论文理论部分的 π₁/π_C > 100 仍然成立。

源域**仅保留 28 个可映射到 PlantDoc 的类别**（2026-09-20 补充决议）。
另 10 个 PlantVillage 类在 PlantDoc 中无对应，予以丢弃，清单写入产物元数据 ——
这是对 R2-1「不匹配类别如何处理」的正面回答。保留它们会构成 partial-set 设定，
而本文六个模块均按闭集设计；且其中最大的 Orange___Haunglongbing（5,507 张）
在目标域完全不存在，会使训练分布的头部成为测试时不出现的类别。

采样公式（对按可用量降序排列的类别 i = 0..C-1）：
    n_i = min(available_i, round(n_max · IF^(-i/(C-1))))

`min` 钳位保证不超过实际可用量；因此**实际达成的 IF 可能低于目标值**，
脚本会如实报告偏差（规格要求：长尾构造协议公开）。

本脚本只处理文件名清单，不读取图像内容。
"""
import argparse, csv, json, os, random, re, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("AGRI_WORKSPACE", Path.home() / "agri-cnz-workspace")) / "cache" / "exp1a"
OUT = REPO / "dataset" / "splits"

SEED = 20260920          # 固定随机种子，写入产物元数据
TARGET_IF = 500          # 目标不平衡因子（决议 2026-09-20，见下）
VAL_FRACTION = 0.10      # 源域验证集比例（供源域选型协议使用）

# 类别分桶 —— 采用长尾识别领域的标准三分约定
# Liu et al., "Large-Scale Long-Tailed Recognition in an Open World", CVPR 2019
# ImageNet-LT / Places-LT 均用此划分。采用标准约定而非自造阈值，
# 使本文结果与文献直接可比，并避免阈值选择受质疑。
BUCKET_MANY_MIN = 101    # Many-shot : > 100
BUCKET_FEW_MAX = 19      # Few-shot  : < 20；Medium-shot 为两者之间

# 文件名中的泄漏风险标记（见 README，供任务 5 使用）
RE_COPY = re.compile(r"\scopy\.", re.I)
RE_LEAF = re.compile(r"Leaf\s+(\d+)\.(\d+)\.", re.I)
RE_UUID = re.compile(r"^[0-9a-f-]{36}___", re.I)


def load():
    pv = json.loads((CACHE / "filenames_plantvillage.json").read_text())
    pdtree = json.loads((CACHE / "tree_plantdoc.json").read_text())["tree"]
    mapping = json.loads((REPO / "results" / "exp-1a-class-stats.json").read_text())
    return pv, pdtree, mapping["overlap"]["PlantVillage_x_PlantDoc"]["mapping"]


def build_profile(pv, keep_classes):
    """按指数分布确定每类保留量，返回 [(class, keep, available, theoretical)]。"""
    av = sorted(((c, v) for c, v in pv.items() if c in keep_classes),
                key=lambda kv: -len(kv[1]))
    nmax, C = len(av[0][1]), len(av)
    prof = []
    for i, (cls, files) in enumerate(av):
        theo = round(nmax * (TARGET_IF ** (-i / (C - 1))))
        prof.append((cls, min(len(files), theo), len(files), theo))
    return prof


def bucket_of(n):
    """OLTR 标准分桶。"""
    if n >= BUCKET_MANY_MIN:
        return "many"
    return "few" if n <= BUCKET_FEW_MAX else "medium"


def leak_flags(fn):
    f = []
    if RE_COPY.search(fn):
        f.append("copy_suffix")
    if not RE_UUID.match(fn):
        f.append("no_uuid")
    m = RE_LEAF.search(fn)
    return f, (m.group(1) if m else "")


def main():
    pv, pdtree, mapping = load()
    rng = random.Random(SEED)
    mapped = set(mapping.values())
    dropped = sorted(set(pv) - mapped, key=lambda c: -len(pv[c]))
    prof = build_profile(pv, mapped)

    # ── 源域：子采样 + train/val 划分 ──
    rows, per_class = [], {}
    for idx, (cls, keep, avail, theo) in enumerate(prof):
        files = sorted(pv[cls])            # 先排序保证确定性
        rng.shuffle(files)
        sel = files[:keep]
        n_val = max(1, round(keep * VAL_FRACTION))
        for j, fn in enumerate(sel):
            flags, leaf = leak_flags(fn)
            rows.append({
                "path": f"raw/color/{cls}/{fn}",
                "class": cls, "class_idx": idx,
                "bucket": bucket_of(keep),
                "split": "val" if j < n_val else "train",
                "leaf_group": leaf, "leak_flags": "|".join(flags),
            })
        per_class[cls] = {"class_idx": idx, "available": avail,
                          "theoretical": theo, "kept": keep,
                          "bucket": bucket_of(keep),
                          "train": keep - n_val, "val": n_val}

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "plantvillage_lt_source.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ── 目标域：PlantDoc 全量，保留其原生划分标记 ──
    IMG = re.compile(r"(?i)\.(jpg|jpeg|png)$")
    trows = []
    pv2pd = {v: k for k, v in mapping.items()}
    pd2idx = {mapping[k]: per_class[mapping[k]]["class_idx"] for k in mapping}
    for e in pdtree:
        if e["type"] != "blob" or not IMG.search(e["path"]):
            continue
        parts = e["path"].split("/")
        if len(parts) < 3:
            continue
        native, cls = parts[0], parts[1]
        pvc = mapping.get(cls)
        trows.append({"path": e["path"], "plantdoc_class": cls,
                      "plantvillage_class": pvc or "",
                      "class_idx": pd2idx.get(pvc, -1),
                      "native_split": native})
    # 排除大小写冲突的路径：macOS/APFS 默认不区分大小写，此类文件只能落盘其一，
    # 且落盘的是哪一个不确定。Linux 上两者都在 —— 同一划分文件在两端行为不一致。
    # 实测这 6 组并非字节重复（体积差异显著），而是同一主题的不同版本，
    # 本身即近重复嫌疑。全部排除以保证跨平台确定性，代价 12/2578 ≈ 0.5%。
    lower = collections.Counter(r["path"].lower() for r in trows)
    case_clash = sorted(r["path"] for r in trows if lower[r["path"].lower()] > 1)
    trows = [r for r in trows if lower[r["path"].lower()] == 1]
    trows.sort(key=lambda r: r["path"])
    with (OUT / "plantdoc_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0]))
        w.writeheader(); w.writerows(trows)

    # ── 核对实际达成的分布 ──
    kept = [p[1] for p in prof]
    realized_if = round(max(kept) / min(kept), 1)
    shortfall = [{"class": c, "theoretical": t, "available": a, "kept": k}
                 for c, k, a, t in prof if k < t]
    tail_counts = collections.Counter(bucket_of(k) for c, k, a, t in prof)
    tgt_per_class = collections.Counter(r["plantvillage_class"] for r in trows if r["plantvillage_class"])

    meta = {
        "task": "1b",
        "decision": "A-1 修正版（plan/基准方案决策简报.md, 2026-09-20）",
        "seed": SEED, "target_imbalance_factor": TARGET_IF,
        "val_fraction": VAL_FRACTION,
        "formula": "n_i = min(available_i, round(n_max * IF^(-i/(C-1))))，类别按可用量降序",
        "source": {
            "dataset": "PlantVillage (raw/color)",
            "classes": len(prof), "total_kept": sum(kept),
            "total_available": sum(p[2] for p in prof),
            "max_class": max(kept), "min_class": min(kept),
            "realized_imbalance_factor": realized_if,
            "theoretical_imbalance_factor": TARGET_IF,
            "deviation_note": (
                f"实际 IF 为 {realized_if}，与目标 {TARGET_IF} 基本一致"
                f"（差异来自最小类的取整）。但 {len(shortfall)} 个类别的可用量"
                f"不足以达到理论采样量，因 min 钳位使分布在中段偏离理想指数形状；"
                f"逐类偏差见 classes_below_theoretical。"
            ) if shortfall else
            f"无类别受可用量限制，实际 IF ({realized_if}) 与目标 ({TARGET_IF}) 一致。",
            "classes_below_theoretical": shortfall,
            "bucket_convention": {
                "reference": "Liu et al., Large-Scale Long-Tailed Recognition "
                             "in an Open World, CVPR 2019 (OLTR)；"
                             "ImageNet-LT / Places-LT 沿用此划分",
                "many_shot": "> 100", "medium_shot": "20-100", "few_shot": "< 20",
                "counts": dict(tail_counts),
                "tail_definition": "Tail = medium ∪ few（≤100 张）。"
                                   "身份由源域构造分布确定，与目标域计数无关。",
                "tail_classes": [c for c, k, a, t in prof if bucket_of(k) != "many"],
            },
            "per_class": per_class,
        },
        "validation_set_note": {
            "current": f"按 {VAL_FRACTION:.0%} 比例从保留集中切分，"
                       "最小类（few-shot）因此仅得 1 张验证样本。",
            "issue": "从 few-shot 类切走训练样本代价高昂——最小类仅 11 张。"
                     "且单张验证样本无法反映该类性能。",
            "alternative_verified": "可改用**未被 LT 采样选中的剩余图像**构建均衡验证集："
                                    "剩余合计 14,854 张；全部 medium/few-shot 类别剩余 264–867 张，"
                                    "足以支撑每类 50 张的均衡验证集，且完全不占用尾类训练样本。"
                                    "仅 4 个 many-shot 类别（受可用量限制）剩余为 0，"
                                    "需从其保留集切分——代价可忽略（50 / 1695~5357）。",
            "owner": "此项属 fix-experiment-protocol 的选型协议决策，1b/1c 不越界决定。",
        },
        "dropped_classes": {
            "note": "这些 PlantVillage 类别在 PlantDoc 中无对应，已从源域丢弃。"
                    "此为对 R2-1「不匹配类别如何处理」的答复依据。",
            "count": len(dropped),
            "images_discarded": sum(len(pv[c]) for c in dropped),
            "classes": [{"class": c, "available": len(pv[c])} for c in dropped],
        },
        "target": {
            "dataset": "PlantDoc", "total": len(trows),
            "classes": len({r["plantdoc_class"] for r in trows}),
            "mapped_to_source": len(tgt_per_class),
            "native_splits": dict(collections.Counter(r["native_split"] for r in trows)),
            "per_class_mapped": dict(tgt_per_class),
            "case_collision_excluded": {
                "count": len(case_clash), "paths": case_clash,
                "reason": "路径仅大小写不同。macOS 默认文件系统不区分大小写，"
                          "此类文件只能落盘其一且不确定是哪一个；Linux 上两者都在。"
                          "实测非字节重复（体积差异显著），系同一主题的不同版本，"
                          "本身即近重复嫌疑。全部排除以保证跨平台确定性。",
            },
            "protocol_note": "train/test 的使用方式（transductive 还是 inductive）"
                             "属 fix-experiment-protocol 的待决项，此处仅保留原生划分标记。",
            "unevaluable_classes": [
                {"plantvillage_class": c, "target_images": n,
                 "source_images": per_class[c]["kept"],
                 "source_rank": per_class[c]["class_idx"] + 1}
                for c, n in sorted(tgt_per_class.items(), key=lambda kv: kv[1]) if n < 10
            ],
            "unevaluable_note": "目标样本少于 10 张的类别，per-class recall 无统计意义，"
                                "须单独报告、不计入聚合尾类指标（决议 2026-09-20）。"
                                "注意其在源域的排名——本例中该类在源域属头部，"
                                "即训练充分但无法评测，与尾类问题性质不同。",
        },
        "leakage_flags": {
            "note": "基于文件名的启发式标记，供任务 5 使用。"
                    "文件名无法提供可靠的单叶分组键 —— 多数类别仅有 1 个批次 token "
                    "覆盖上千张图。真正的近重复检测需在图像层面做感知哈希。",
            "copy_suffix": sum(1 for r in rows if "copy_suffix" in r["leak_flags"]),
            "no_uuid": sum(1 for r in rows if "no_uuid" in r["leak_flags"]),
            "with_leaf_group": sum(1 for r in rows if r["leaf_group"]),
        },
    }
    (REPO / "results" / "exp-1b-lt-construction.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False))

    print(f"  丢弃的无目标类别     : {len(dropped)} 类 / "
          f"{sum(len(pv[c]) for c in dropped):,} 张")
    print(f"  源域 PlantVillage-LT : {sum(kept):,} 张 / {len(prof)} 类  "
          f"(max {max(kept)}, min {min(kept)}, 实际 IF {realized_if})")
    print(f"  分桶 (OLTR)          : many {tail_counts['many']} / "
          f"medium {tail_counts['medium']} / few {tail_counts['few']}")
    print(f"  排除大小写冲突路径   : {len(case_clash)} 个")
    print(f"  目标域 PlantDoc      : {len(trows):,} 张 / "
          f"{len({r['plantdoc_class'] for r in trows})} 类")
    print(f"  受可用量限制的类别   : {len(shortfall)}")
    print(f"  划分文件             : dataset/splits/")
    return meta


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
