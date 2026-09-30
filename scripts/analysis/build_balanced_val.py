#!/usr/bin/env python3
"""fix-experiment-protocol 任务 0 —— 以 LT 采样未选中的剩余图像构建均衡源域验证集。

决议（2026-09-21，作者同意按推荐执行）：
  主结果按源域验证集选型（R1-3）。原验证集按 10% 比例从 LT 保留集切出，
  继承了长尾分布（2,323 张，最小类仅 1 张），在其上选型等于按头部类性能选型，
  与本文以尾类为核心的目标相悖；同时切走了尾类本就稀缺的训练样本。

新构造
  * LT 保留集（23,688 张）**全部**用于训练，长尾分布与 IF 不变。
  * 每类 VAL_PER_CLASS 张验证图，优先取自 LT 采样未选中的剩余图像。
  * 剩余不足的类别（受可用量限制、LT 采样已用尽）从其保留集中补足，
    只取与任何其它图像都不成组的"孤立"图像。这些类别均为头部类，代价可忽略。
  * 验证集上的 Top-1 = 类别均衡准确率，选型因此不再偏向头部类。

防泄漏（与 EXP-5 同一套分组，在**完整图像池**上建立）
  A 近重复：256 位 phash，D4 变体最小汉明距离 ≤ 16（near_dup_pairs_p256wide）
  B 同叶片：文件名 "Leaf <n>" + 批次前缀
  验证图所在的组不得含有任何训练图；每组至多一张进入验证集。
  写回后应再运行 exp5_apply_groups.py 复核（应报告 0 个跨越 train/val 的组）。

只处理文件名与已缓存的哈希配对，不读取图像内容。
"""
import collections, csv, gzip, json, os, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lt_splits import SEED  # noqa: E402
from exp5_apply_groups import THRESHOLD, leaf_key  # noqa: E402

WS = Path(os.environ.get("AGRI_WORKSPACE", Path.home() / "agri-cnz-workspace"))
SPLIT = REPO / "dataset/splits/plantvillage_lt_source.csv"
VAL_PER_CLASS = 50


def main():
    rows = list(csv.DictReader(open(SPLIT, newline="")))
    fields = list(rows[0].keys())
    pool = json.loads((WS / "cache/exp1a/filenames_plantvillage.json").read_text())
    kept = {r["path"] for r in rows}
    by_cls = collections.defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)

    # ---- 在完整图像池上建组：近重复(A) ∪ 同叶片(B)
    par = {}

    def find(x):
        while par.get(x, x) != x:
            par[x] = par.get(par[x], par[x])
            x = par[x]
        return x

    def union(a, b):
        par.setdefault(a, a); par.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    all_paths = [f"raw/color/{c}/{fn}" for c in by_cls for fn in pool[c]]
    for p in all_paths:
        par.setdefault(p, p)
    with gzip.open(WS / "cache/phash/near_dup_pairs_p256wide.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if int(r["dist"]) > THRESHOLD or r["dataset_a"] != "plantvillage" or r["dataset_b"] != "plantvillage":
                continue
            a, b = f"raw/color/{r['relpath_a']}", f"raw/color/{r['relpath_b']}"
            if a in par and b in par:
                union(a, b)
    first_leaf = {}
    for p in all_paths:
        lk = leaf_key({"path": p, "class": p.split("/")[2]})
        if lk:
            if lk in first_leaf:
                union(first_leaf[lk], p)
            else:
                first_leaf[lk] = p
    members = collections.defaultdict(list)
    for p in all_paths:
        members[find(p)].append(p)

    rng = random.Random(f"{SEED}-balanced-val")
    val, from_kept, per_class = set(), set(), {}
    for cls in sorted(by_cls):
        remainder = sorted(set(f"raw/color/{cls}/{fn}" for fn in pool[cls]) - kept)
        rng.shuffle(remainder)
        chosen, used_groups = [], set()
        for p in remainder:                       # 剩余图像：所在组不得含 LT 保留图
            g = find(p)
            if g in used_groups or any(m in kept for m in members[g]):
                continue
            chosen.append(p); used_groups.add(g)
            if len(chosen) == VAL_PER_CLASS:
                break
        n_rem = len(chosen)
        if len(chosen) < VAL_PER_CLASS:           # 不足：从保留集取孤立图像
            cand = sorted(r["path"] for r in by_cls[cls])
            rng.shuffle(cand)
            for p in cand:
                if len(members[find(p)]) == 1:
                    chosen.append(p); from_kept.add(p)
                    if len(chosen) == VAL_PER_CLASS:
                        break
        if len(chosen) < VAL_PER_CLASS:
            raise SystemExit(f"{cls}：只找到 {len(chosen)} 张合格验证图")
        val.update(chosen)
        n_kept = len(by_cls[cls])
        per_class[cls] = {"bucket": by_cls[cls][0]["bucket"], "lt_kept": n_kept,
                          "remainder_available": len(remainder), "val_from_remainder": n_rem,
                          "val_from_kept": VAL_PER_CLASS - n_rem,
                          "train": n_kept - (VAL_PER_CLASS - n_rem)}

    # ---- 写回：保留集全部为 train（被取作验证的孤立图除外），剩余图像追加为 val
    old_val = sum(r["split"] == "val" for r in rows)
    for r in rows:
        r["split"] = "val" if r["path"] in from_kept else "train"
    tmpl = {r["class"]: r for r in rows}
    for p in sorted(val - kept):
        cls = p.split("/")[2]
        t = tmpl[cls]
        rows.append({**{k: "" for k in fields}, "path": p, "class": cls, "class_idx": t["class_idx"],
                     "bucket": t["bucket"], "split": "val"})
    # 自检：验证图的组内无训练图
    train = {r["path"] for r in rows if r["split"] == "train"}
    bad = [p for p in val if any(m in train for m in members[find(p)])]
    assert not bad, bad[:5]
    rows.sort(key=lambda r: (int(r["class_idx"]), r["split"] != "val", r["path"]))
    with open(SPLIT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    tr = [v["train"] for v in per_class.values()]
    meta = {
        "task": "fix-experiment-protocol 0.1–0.2 均衡源域验证集",
        "decision": "2026-09-21：主结果按源域验证集选型；验证集改为 LT 剩余图像构建的均衡集（推荐方案）",
        "seed": f"{SEED}-balanced-val", "val_per_class": VAL_PER_CLASS,
        "grouping": {"near_dup": f"phash 256 位，D4 最小汉明距离 ≤ {THRESHOLD}", "same_leaf": "文件名 Leaf <n> + 批次前缀",
                     "rule": "验证图所在组不含任何训练图；每组至多一张入验证集"},
        "previous_val": {"construction": "LT 保留集按 10% 比例切分", "images": old_val,
                         "issue": "继承长尾分布，最小类 1 张；在其上选型偏向头部类，且占用尾类训练样本"},
        "result": {"train": sum(tr), "val": len(val),
                   "val_from_remainder": len(val - kept), "val_from_kept": len(from_kept),
                   "classes_topped_up_from_kept": sorted(c for c, v in per_class.items() if v["val_from_kept"]),
                   "train_imbalance_factor": round(max(tr) / min(tr), 1),
                   "train_min_class": min(tr), "train_max_class": max(tr)},
        "selection_metric": "验证集 Top-1；因验证集类别均衡，等于类别平均准确率",
        "per_class": per_class,
    }
    (REPO / "results/exp-protocol-0-balanced-val.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta["result"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
