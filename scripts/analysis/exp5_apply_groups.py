#!/usr/bin/env python3
"""EXP-5 第三步：把近重复分组写回划分文件，并修复横跨 train/val 的组。

两种互补的分组依据
  A. 近重复分组：256 位 phash，D4 变体最小汉明距离 <= 16。
  B. 叶片标识分组：部分 PlantVillage 文件名带 "Leaf <n>"（可再附 "Day <n>"）
     标识，同一批次前缀下同一编号即同一枚叶片的多次拍摄。

为什么两者都要
  实测 A 对 B 的覆盖为 0：同一枚叶片在不同日期/角度下的照片外观差异很大，
  感知哈希判为不同；而它们恰恰是最严重的泄漏形式。仅做 A 会把
  Tomato___Late_blight 中 18.4% 的验证样本误判为洁净。

重要限定（必须随结果一起陈述，不得省略）
  B 只覆盖源域 7.0% 的图像，且集中在 3 个 Tomato 类别；其余 93% 的图像
  没有任何植株/地块/采集日元数据。因此：
  可以声称：「近重复图像不跨越 train/val 边界；在具备叶片标识的类别上，
            同一枚叶片的所有照片不跨越边界」。
  不可以声称：「全部同株照片不跨越边界」—— 该保证在现有元数据下无法建立，
            稿件与回应信必须如实说明这一残余风险。

修复策略
  横跨 train/val 的近重复组整体并入 **train**（除非全部成员原本就在 val）。
  依据：重复样本进入评测集会直接扭曲指标，进入训练集只相当于重加权。
  评测集的洁净度优先。
"""
import argparse, collections, csv, gzip, json, os, re, sys

LEAF_RE = re.compile(r"\bLeaf[\s_]*(\d+)", re.I)


def leaf_key(row):
    """从文件名解析 (批次前缀, 叶片编号)；无标识返回 None。"""
    fn = os.path.basename(row["path"])
    tail = fn.split("___", 1)[1] if "___" in fn else fn
    m = LEAF_RE.search(tail)
    if not m:
        return None
    return (row["class"], tail[:m.start()].strip(), m.group(1))

THRESHOLD = 16  # 256 位 phash；见 results/exp-5-leakage-audit.json 的距离分布依据

# 任务 5.3 的代码审计结论。写在脚本里而非事后手工补进 JSON，
# 否则重跑本脚本会把它覆盖掉。
TARGET_LABEL_USAGE = {
    "task": "5.3 声明目标域标注的用途边界",
    "manuscript_claim": "正文两处声称使用目标域训练集 10% 的标注进行自适应（paper_v0 纯文本第 603 行、第 1102–1103 行）",
    "implementation_audit": {
        "supervised_loss_on_target_labels": "无 —— 全仓库无任何以目标域真实标签计算的分类损失",
        "target_data_in_training": [
            "code/src/train.py 域分类器 —— 只用域身份（源/目标），数据经 StripLabels，不读类别标签",
            "code/src/inference.py CC-GANM —— 目标域图像用于风格迁移，类别条件取源域模型伪标签",
            "code/src/modules/ss_plam.py 伪标签 —— 由模型自身预测产生，不读真实标签"
        ],
        "target_labels_in_model_selection": "无 —— train.py 按源域验证集 Top-1 保存 cd_apdm_bestval.pt（主结果），另存 cd_apdm_final.pt（附录）。**更正**：此前该字段称“选型用源域验证集”，但当时 train.py 实际只保存最后一个 epoch，并无选型逻辑；2026-09-21 补上。",
        "target_labels_in_evaluation": "是 —— 仅用于计算上报指标，code/src/train.py:305",
        "target_labels_as_conditioning": "**有过，已改正（2026-09-21）**：code/src/inference.py 的 cc_ganm_train 曾以目标域真实标签 tgt_b['label'] 作为 G_T2S 的类别条件，并经循环一致损失进入 G_S2T 的训练；G_S2T 随后为 Mean-Teacher 生成学生输入。这不是监督损失，但属目标域标签泄漏。已改为源域模型的伪标签，目标域自适应数据经 StripLabels 去掉 label 字段；回归测试 tests/test_baselines.py::test_ccganm_adaptation_never_reads_target_labels （改回旧写法即失败）。此前的 grep 核查只搜了损失相关写法，漏掉了这一处。"
    },
    "conclusion": "修复 CC-GANM 条件标签后，实际协议是**完全无监督**的域自适应：目标域标注 0% 进入训练与生成器条件、0% 进入选型，仅用于评测。修复前的结果（含服务器冒烟）不满足这一点，均不得使用。稿件「10% 标注用于自适应」的表述须改写。",
    "verification": "grep -rn --include='*.py' -E \"tgt.*\\[.label.\\]|batch\\[.label.\\]\" code/src/ 并逐处确认只出现在评测路径；结构保证见 StripLabels"
}


def union_groups(pairs_path, thr):
    par = {}
    def find(x):
        while par.get(x, x) != x:
            par[x] = par.get(par[x], par[x]); x = par[x]
        return x
    with gzip.open(pairs_path, "rt") as fh:
        for r in csv.DictReader(fh):
            if int(r["dist"]) > thr:
                continue
            a = (r["dataset_a"], r["relpath_a"]); b = (r["dataset_b"], r["relpath_b"])
            par.setdefault(a, a); par.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
    root2id, out = {}, {}
    for k in sorted(par):
        r = find(k)
        if r not in root2id:
            root2id[r] = f"g{len(root2id):04d}"
        out[k] = root2id[r]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.environ.get("AGRI_WORKSPACE", os.path.expanduser("~/agri-cnz-workspace")))
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ws = os.path.abspath(os.path.expanduser(args.workspace))
    cache = os.path.join(ws, "cache", "phash")

    grp = union_groups(os.path.join(cache, "near_dup_pairs_p256wide.csv.gz"), args.threshold)
    sha = {}
    with gzip.open(os.path.join(cache, "image_hashes.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            sha[(r["dataset"], r["relpath"])] = r["sha256"]
    exact = collections.Counter(sha.values())

    report = {"threshold_bits": 256, "threshold_hamming": args.threshold,
              "source": {}, "target": {}, "target_label_usage": TARGET_LABEL_USAGE}

    # ---------- 源域 ----------
    sp = "dataset/splits/plantvillage_lt_source.csv"
    rows = list(csv.DictReader(open(sp)))
    fields = list(rows[0].keys())
    def key(r): return ("plantvillage", r["path"].split("raw/color/", 1)[1])

    # 并查集合并两种分组关系：近重复(A) 与 同叶片(B)
    par = {}
    def find(x):
        while par.get(x, x) != x:
            par[x] = par.get(par[x], par[x]); x = par[x]
        return x
    def union(a, b):
        par.setdefault(a, a); par.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    for r in rows:
        par.setdefault(r["path"], r["path"])
    first_nd, first_leaf = {}, {}
    src_of = {}
    for r in rows:
        g = grp.get(key(r))
        if g:
            src_of.setdefault(r["path"], set()).add("near")
            if g in first_nd:
                union(first_nd[g], r["path"])
            else:
                first_nd[g] = r["path"]
        lk = leaf_key(r)
        if lk:
            src_of.setdefault(r["path"], set()).add("leaf")
            if lk in first_leaf:
                union(first_leaf[lk], r["path"])
            else:
                first_leaf[lk] = r["path"]
    members = collections.defaultdict(list)
    for r in rows:
        members[find(r["path"])].append(r)
    bygroup = {}
    gid = {}
    for root, ms in members.items():
        if len(ms) < 2:
            continue
        gid[root] = f"s{len(gid):04d}"
        bygroup[gid[root]] = ms
    leafgrp_multi = sum(1 for k, v in collections.Counter(
        leaf_key(r) for r in rows if leaf_key(r)).items() if v > 1)
    # 修复前的污染度量：逐类统计「验证样本与训练集同源」的比例。
    # 这是回应 R1-1 的核心证据，必须可复现，不能只在临时脚本里算过。
    pre_val = collections.Counter(r["class"] for r in rows if r["split"] == "val")
    contaminated = collections.Counter()
    for g, ms in bygroup.items():
        if len({m["split"] for m in ms}) > 1:
            for m in ms:
                if m["split"] == "val":
                    contaminated[m["class"]] += 1
    report["source_before_repair"] = {
        "val_total": sum(pre_val.values()),
        "val_contaminated": sum(contaminated.values()),
        "val_contaminated_pct": round(100.0 * sum(contaminated.values()) / sum(pre_val.values()), 2),
        "by_class": {c: {"val": pre_val[c], "contaminated": contaminated[c],
                         "pct": round(100.0 * contaminated[c] / pre_val[c], 1)}
                     for c in sorted(contaminated, key=lambda x: -contaminated[x])},
    }

    # 须在下面的修复循环之前统计：修复会把横跨组的成员全部改成 train，之后再按"横跨"筛选恒为 0
    # （2026-09-28 查出：此前在修复之后统计，caught_by_phash_alone 一直报 0，实际为 3，均为字节相同副本）。
    nd_only_caught = sum(1 for g, ms in bygroup.items()
                         if len({m["split"] for m in ms}) > 1
                         for m in ms if m["split"] == "val"
                         and "near" in src_of.get(m["path"], set()))
    moved = []
    for g, members in bygroup.items():
        splits = {m["split"] for m in members}
        if len(splits) > 1:
            for m in members:
                if m["split"] != "train":
                    moved.append((g, m["path"], m["split"]))
                    m["split"] = "train"
    moved_paths = {m[1] for m in moved}
    for r in rows:
        root = find(r["path"])
        g = gid.get(root, "")
        r["leaf_group"] = g
        flags = []
        if g:
            srcs = src_of.get(r["path"], set())
            if "near" in srcs:
                flags.append("exact_dup" if exact[sha[key(r)]] > 1 else "near_dup")
            if "leaf" in srcs:
                flags.append("same_leaf")
        if r["path"] in moved_paths:
            flags.append("moved_to_train")
        r["leak_flags"] = "|".join(flags)
    report["source_before_repair"]["caught_by_phash_alone"] = nd_only_caught
    report["source_before_repair"]["note"] = (
        "感知哈希单独使用时只能捕获上述污染样本中的 %d 例；其余为同一枚叶片在"
        "不同日期/角度下的照片，外观差异大，哈希判为不同。两种分组必须并用。" % nd_only_caught)

    cnt = collections.Counter(r["split"] for r in rows)
    report["source"] = {
        "images": len(rows), "grouped_images": sum(1 for r in rows if r["leaf_group"]),
        "groups": len(bygroup),
        "groups_from_near_dup_only": len({g for g, ms in bygroup.items()
                                          if all("leaf" not in src_of.get(m["path"], set()) for m in ms)}),
        "groups_with_leaf_id": len({g for g, ms in bygroup.items()
                                    if any("leaf" in src_of.get(m["path"], set()) for m in ms)}),
        "images_with_leaf_id": sum(1 for r in rows if leaf_key(r)),
        "leaf_id_coverage_pct": round(100.0 * sum(1 for r in rows if leaf_key(r)) / len(rows), 2),
        "straddling_groups_repaired": len({m[0] for m in moved}),
        "images_moved_to_train": len(moved),
        "split_counts_after": dict(cnt),
    }

    # ---------- 目标域 ----------
    tp = "dataset/splits/plantdoc_target.csv"
    trows = list(csv.DictReader(open(tp)))
    tfields = list(trows[0].keys())
    for f in ("dup_group", "leak_flags"):
        if f not in tfields:
            tfields.append(f)
    tby = collections.defaultdict(list)
    for r in trows:
        g = grp.get(("plantdoc", r["path"]))
        if g:
            tby[g].append(r)
    conflict_groups, ceiling_loss = set(), 0
    for g, members in tby.items():
        labels = collections.Counter(m["plantvillage_class"] for m in members)
        if len(labels) > 1:
            conflict_groups.add(g)
            # 组内至多只有一个标签能被答对，其余成员必错
            ceiling_loss += len(members) - labels.most_common(1)[0][1]
    for r in trows:
        g = grp.get(("plantdoc", r["path"]))
        r["dup_group"] = g or ""
        flags = []
        if g:
            flags.append("exact_dup" if exact[sha[("plantdoc", r["path"])]] > 1 else "near_dup")
            if g in conflict_groups:
                flags.append("label_conflict")
        r["leak_flags"] = "|".join(flags)
    report["target"] = {
        "images": len(trows), "grouped_images": sum(1 for r in trows if r["dup_group"]),
        "groups": len(tby), "label_conflict_groups": len(conflict_groups),
        "label_conflict_images": sum(len(v) for g, v in tby.items() if g in conflict_groups),
        "unavoidable_errors": ceiling_loss,
        "accuracy_ceiling_pct": round(100.0 * (len(trows) - ceiling_loss) / len(trows), 2),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("\n[dry-run] 未写回文件"); return
    with open(sp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    with open(tp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=tfields, lineterminator="\n"); w.writeheader(); w.writerows(trows)
    print(f"\n已写回 {sp}\n已写回 {tp}")
    os.makedirs("results", exist_ok=True)
    # 保留由其它脚本写入、本脚本不产出的记录（如 exp5_original_split.py 的修复前审计），
    # 否则重跑本脚本会把它们抹掉 —— 修复前的污染数字正是这样丢过一次。
    out = "results/exp-5-leakage-audit.json"
    if os.path.exists(out):
        prev = json.load(open(out))
        for k, v in prev.items():
            report.setdefault(k, v)
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
