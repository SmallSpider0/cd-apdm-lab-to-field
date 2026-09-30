#!/usr/bin/env python3
"""EXP-5 第一步：为防泄漏审计计算逐图像哈希。

产出两类哈希：
  sha256  文件字节的精确哈希 —— 识别完全相同的文件
  phash   感知哈希，同时算 64 位与 256 位两档，各自再计算 90/180/270 度旋转
          与水平翻转共 8 个变体 —— 识别经旋转/翻转/重压缩后的近重复

为什么要两档：64 位 phash 在「纯色背景 + 单片绿叶」这类低信息量图像上会
大量碰撞（实测 Soybean___healthy 与 PlantDoc 的 "green leaf isolated on
white" 在距离 4 上相连，但并非同一张照片）。256 位容量更大，可将这类
误配剔除。两档都保留，以便在报告中展示阈值选择的依据。

为什么要算旋转变体：PlantVillage 的采集流程会对同一枚叶片多角度拍摄，
仅比对原图哈希会漏掉这类近重复。审计应偏保守（宁可多报），
因此近重复判定取 8 个变体间的最小汉明距离。

本脚本只产出哈希，不做配对与分组（见 exp5_near_dup.py），
以便配对阈值可以反复调整而无需重新解码 4 万张图。
"""
import argparse, csv, gzip, hashlib, os, sys, time
import multiprocessing as mp

try:
    import imagehash
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"缺少依赖：{exc}. 请使用 agri-ctta 环境运行。")

Image.MAX_IMAGE_PIXELS = 200_000_000
HASH_SIZES = (8, 16)  # 8x8 -> 64 bit; 16x16 -> 256 bit


def variants(img, hash_size):
    """原图 + 3 个旋转 + 水平翻转及其 3 个旋转 = 8 个变体的 phash。

    这 8 个变体构成二面体群 D4 的轨道，对该群封闭；因此比对时只需
    展开一侧的变体，另一侧取原图即可覆盖全部旋转/翻转对应关系。
    """
    out = []
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    for base in (img, flipped):
        cur = base
        for _ in range(4):
            out.append(str(imagehash.phash(cur, hash_size=hash_size)))
            cur = cur.transpose(Image.ROTATE_90)
    return out


def hash_one(task):
    key, abspath = task
    try:
        h = hashlib.sha256()
        with open(abspath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()

        with Image.open(abspath) as im:
            # draft 让 JPEG 在解码阶段就降采样，显著降低内存与耗时
            try:
                im.draft("RGB", (128, 128))
            except Exception:
                pass
            im = im.convert("L").resize((64, 64), Image.BILINEAR)
            vs = {hs: variants(im, hs) for hs in HASH_SIZES}
        return (key, sha, vs, "")
    except Exception as exc:
        return (key, "", None, f"{type(exc).__name__}: {exc}")


def collect(dataset, root, exts):
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            ab = os.path.join(dirpath, fn)
            rel = os.path.relpath(ab, root)
            items.append((f"{dataset}\t{rel}", ab))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=os.environ.get("AGRI_WORKSPACE", os.path.expanduser("~/agri-cnz-workspace")))
    ap.add_argument("--classes", default="dataset/splits/plantvillage_lt_source.csv",
                    help="从该划分文件读取要纳入的源域类别（28 个可映射类）")
    ap.add_argument("--workers", type=int, default=4, help="进程数，默认保守取 4 以限制本机内存")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ws = os.path.abspath(os.path.expanduser(args.workspace))
    for bad in ("Mobile Documents", "CloudDocs", "Dropbox", "OneDrive"):
        if bad in ws:
            sys.exit(f"拒绝写入云同步目录：{ws}")

    keep = set()
    with open(args.classes) as f:
        for r in csv.DictReader(f):
            keep.add(r["class"])

    pv_root = os.path.join(ws, "data", "plantvillage", "raw", "color")
    pd_root = os.path.join(ws, "data", "plantdoc")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    items = [t for t in collect("plantvillage", pv_root, exts)
             if t[0].split("\t", 1)[1].split(os.sep)[0] in keep]
    items += collect("plantdoc", pd_root, exts)
    total = len(items)
    print(f"[exp5-hash] 待处理 {total} 张（plantvillage 28 类全量池 + plantdoc 全量）", flush=True)

    out = args.out or os.path.join(ws, "cache", "phash", "image_hashes.csv.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    done = 0
    errs = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        cols = ["dataset", "relpath", "sha256"]
        for hs in HASH_SIZES:
            cols += [f"p{hs*hs}_{i}" for i in range(8)]
        w.writerow(cols + ["error"])
        # maxtasksperchild 定期回收子进程，避免 PIL 解码器在长队列上累积内存
        with mp.Pool(args.workers, maxtasksperchild=2000) as pool:
            for key, sha, vs, err in pool.imap_unordered(hash_one, items, chunksize=32):
                ds, rel = key.split("\t", 1)
                flat = []
                for hs in HASH_SIZES:
                    flat += (vs[hs] if vs else [""] * 8)
                w.writerow([ds, rel, sha] + flat + [err])
                done += 1
                if err:
                    errs += 1
                    if errs <= 10:
                        print(f"[exp5-hash] 失败 {ds}/{rel}: {err}", flush=True)
                if done % 2000 == 0:
                    el = time.time() - t0
                    print(f"[exp5-hash] {done}/{total}  {el:.0f}s  {done/el:.0f} img/s", flush=True)

    print(f"[exp5-hash] 完成 {done} 张，失败 {errs} 张，用时 {time.time()-t0:.0f}s", flush=True)
    print(f"[exp5-hash] 输出：{out}", flush=True)
    if errs:
        print("[exp5-hash] 注意：存在解码失败图像，后续分组须显式处理，不得静默跳过", flush=True)


if __name__ == "__main__":
    main()
