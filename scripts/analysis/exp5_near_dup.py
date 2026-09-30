#!/usr/bin/env python3
"""EXP-5 第二步：从逐图像哈希做近重复配对与分组。

方法
  距离 = 图像 A 的 8 个二面体变体（4 旋转 × 2 翻转）与图像 B 原图之间
         phash 的最小汉明距离。变体集对 D4 群封闭，只需单边展开。
  分组 = 对阈值内的配对做并查集，得到近重复连通分量。

阈值不靠断言选取。脚本对每个距离输出：
  * 配对数
  * 同类配对占比 —— 叶片整体相似，阈值过松时不同类别会开始相连
  * 跨数据集配对数 —— 源域与目标域本应无重叠，出现即为误配或真泄漏
这两条曲线的转折点给出可观测的阈值上界。

注：phash 以中位数为阈值二值化，每个哈希的置位数恒为位长的一半，
故任意两个哈希的汉明距离必为偶数 —— 奇数距离为空是结构性的，不是缺陷。
"""
import argparse, collections, csv, gzip, json, os, sys, time
import numpy as np

BANNER = "PlantDoc_Examples.png"  # 仓库说明用图，非数据


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.environ.get("AGRI_WORKSPACE", os.path.expanduser("~/agri-cnz-workspace")))
    ap.add_argument("--hashes", default=None)
    ap.add_argument("--bits", type=int, default=256, choices=(64, 256))
    ap.add_argument("--max-dist", type=int, default=None, help="默认取位长的 1/16")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ws = os.path.abspath(os.path.expanduser(args.workspace))
    hp = args.hashes or os.path.join(ws, "cache", "phash", "image_hashes.csv.gz")
    words = args.bits // 64
    maxd = args.max_dist if args.max_dist is not None else args.bits // 16
    tag = args.tag or f"p{args.bits}"

    rows = []
    with gzip.open(hp, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                sys.exit(f"哈希文件含失败记录，先修复再分组：{r['dataset']}/{r['relpath']}")
            if r["dataset"] == "plantdoc" and r["relpath"] == BANNER:
                continue
            rows.append(r)
    n = len(rows)
    print(f"[exp5-dup] 载入 {n} 张图像哈希，使用 {args.bits} 位 phash，统计至距离 {maxd}", flush=True)

    ds = np.array([r["dataset"] for r in rows])
    cls = np.array([r["relpath"].split(os.sep)[0] if r["dataset"] == "plantvillage"
                    else r["relpath"].split(os.sep)[1] for r in rows])

    H = np.zeros((n, 8, words), dtype=np.uint64)
    for i, r in enumerate(rows):
        for v in range(8):
            h = int(r[f"p{args.bits}_{v}"], 16)
            for w in range(words):
                H[i, v, w] = np.uint64((h >> (64 * (words - 1 - w))) & 0xFFFFFFFFFFFFFFFF)
    ref = np.ascontiguousarray(H[:, 0, :])

    est = args.block * n * words * 8 / 1e6
    print(f"[exp5-dup] 分块 {args.block} 行，单块临时数组约 {est:.0f} MB", flush=True)
    if est > 400:
        sys.exit("分块过大，会在本机占用过多内存；请减小 --block")

    hist = np.zeros(maxd + 1, dtype=np.int64)
    hist_same = np.zeros(maxd + 1, dtype=np.int64)
    hist_xds = np.zeros(maxd + 1, dtype=np.int64)
    chunks = []
    t0 = time.time()
    for s in range(0, n, args.block):
        e = min(s + args.block, n)
        best = None
        for v in range(8):
            d = np.zeros((e - s, n), dtype=np.uint16)
            for w in range(words):
                d += np.bitwise_count(H[s:e, v, w][:, None] ^ ref[None, :, w]).astype(np.uint16)
            best = d if best is None else np.minimum(best, d)
        rr, cc = np.nonzero(best <= maxd)
        keep = (rr + s) < cc            # 只取上三角，排除自配对与重复计数
        rr, cc = rr[keep], cc[keep]
        dd = best[rr, cc].astype(np.int64)
        gi = rr + s
        same = cls[gi] == cls[cc]
        xds = ds[gi] != ds[cc]
        for k in range(maxd + 1):
            m = dd == k
            hist[k] += int(m.sum()); hist_same[k] += int((m & same).sum()); hist_xds[k] += int((m & xds).sum())
        chunks.append(np.stack([gi, cc, dd], axis=1))
        if (s // args.block) % 100 == 0:
            print(f"[exp5-dup] {e}/{n}  {time.time()-t0:.0f}s", flush=True)
    pairs = np.concatenate(chunks) if chunks else np.zeros((0, 3), dtype=np.int64)
    print(f"[exp5-dup] 扫描完成 {time.time()-t0:.0f}s，候选配对 {len(pairs)}\n", flush=True)

    print("距离   配对数    同类占比   跨数据集   累计")
    cum = 0
    for k in range(maxd + 1):
        if hist[k] == 0:
            continue
        cum += int(hist[k])
        print(f"{k:>3}  {int(hist[k]):>9}   {hist_same[k]/hist[k]:>7.3f}   {int(hist_xds[k]):>8}   {cum:>8}")

    def groups_at(thr):
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        sel = pairs[pairs[:, 2] <= thr]
        for i, j, _ in sel:
            a, b = find(int(i)), find(int(j))
            if a != b:
                parent[a] = b
        comp = collections.defaultdict(list)
        for i in range(n):
            comp[find(i)].append(i)
        return comp, sel

    summary = {"n_images": n, "bits": args.bits,
               "hist": {str(k): int(hist[k]) for k in range(maxd + 1) if hist[k]},
               "hist_same_class_fraction": {str(k): float(hist_same[k] / hist[k]) for k in range(maxd + 1) if hist[k]},
               "hist_cross_dataset": {str(k): int(hist_xds[k]) for k in range(maxd + 1) if hist[k]},
               "by_threshold": {}}
    print("\n阈值   配对   多元组  涉及图像  最大组  跨数据集  跨类别")
    for thr in range(0, maxd + 1, 2):
        comp, sel = groups_at(thr)
        multi = {k: v for k, v in comp.items() if len(v) > 1}
        big = max((len(v) for v in multi.values()), default=0)
        xds = int(sum(1 for i, j, _ in sel if ds[i] != ds[j]))
        xcl = int(sum(1 for i, j, _ in sel if cls[i] != cls[j]))
        summary["by_threshold"][str(thr)] = {"pairs": int(len(sel)), "groups_gt1": len(multi),
            "images_in_groups": sum(len(v) for v in multi.values()), "largest_group": big,
            "cross_dataset_pairs": xds, "cross_class_pairs": xcl}
        print(f"{thr:>4}  {len(sel):>7}  {len(multi):>6}  {sum(len(v) for v in multi.values()):>8}  "
              f"{big:>6}  {xds:>8}  {xcl:>6}")

    out = os.path.join(ws, "cache", "phash", f"near_dup_pairs_{tag}.csv.gz")
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset_a", "relpath_a", "class_a", "dataset_b", "relpath_b", "class_b", "dist"])
        for i, j, d in pairs[np.argsort(pairs[:, 2], kind="stable")]:
            w.writerow([ds[i], rows[i]["relpath"], cls[i], ds[j], rows[j]["relpath"], cls[j], int(d)])
    sj = os.path.join(ws, "cache", "phash", f"near_dup_summary_{tag}.json")
    json.dump(summary, open(sj, "w"), ensure_ascii=False, indent=2)
    print(f"\n[exp5-dup] 明细 {out}\n[exp5-dup] 汇总 {sj}", flush=True)


if __name__ == "__main__":
    main()
