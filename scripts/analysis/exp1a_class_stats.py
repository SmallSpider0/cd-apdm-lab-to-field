#!/usr/bin/env python3
"""EXP-1a —— 候选跨域基准的类别结构与长尾程度统计。

回答 R2-1（重叠类别数）与 R1-1（类别映射）的前置测量，并为基准方案选型
（rebuild-cross-domain-benchmark）提供读数。

只读取数据集的**元数据**（类别目录与文件计数），不下载任何图像。
数据源：GitHub tree API（经 gh 认证）。缓存置于 $AGRI_WORKSPACE/cache/exp1a。

用法：
    python3 scripts/analysis/exp1a_class_stats.py            # 用缓存，缺失则拉取
    python3 scripts/analysis/exp1a_class_stats.py --refresh  # 强制重新拉取
"""
import json, os, re, subprocess, sys, argparse, collections
from pathlib import Path

WS = Path(os.environ.get("AGRI_WORKSPACE", Path.home() / "agri-cnz-workspace"))
CACHE = WS / "cache" / "exp1a"
REPO = Path(__file__).resolve().parents[2]
IMG = re.compile(r"(?i)\.(jpg|jpeg|png)$")

PV_REPO, PD_REPO, IP_REPO = ("spMohanty/PlantVillage-Dataset",
                             "pratikkayal/PlantDoc-Dataset", "xpwu95/IP102")

# PlantDoc 类名 → PlantVillage 类名。依据：物种 + 病害名的字面对应。
# PlantDoc 本就是作为 PlantVillage 的田间对照集构建的，命名体系平行。
PD2PV = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    # PlantDoc 独有、PlantVillage 无对应：
    "Potato leaf": None,          # PlantVillage 有 Potato___healthy，但 PlantDoc test 才有
}


def gh(path, jq=None):
    cmd = ["gh", "api", path] + (["--jq", jq] if jq else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh api {path} 失败: {r.stderr[:200]}")
    return r.stdout


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if p.exists() and not ARGS.refresh:
        return json.loads(p.read_text())
    v = fn()
    p.write_text(json.dumps(v))
    return v


def fetch_plantvillage():
    root = json.loads(gh(f"repos/{PV_REPO}/git/trees/master"))
    raw = next(t for t in root["tree"] if t["path"] == "raw")
    rawt = json.loads(gh(f"repos/{PV_REPO}/git/trees/{raw['sha']}"))
    color = next(t for t in rawt["tree"] if t["path"] == "color")
    dirs = json.loads(gh(f"repos/{PV_REPO}/git/trees/{color['sha']}"))["tree"]
    jq = '[.tree[] | select(.type=="blob") | select(.path|test("(?i)\\\\.(jpg|jpeg|png)$"))] | length'
    out = {}
    for i, t in enumerate(dirs, 1):
        out[t["path"]] = int(gh(f"repos/{PV_REPO}/git/trees/{t['sha']}", jq).strip())
        sys.stderr.write(f"\r  PlantVillage {i}/{len(dirs)}")
    sys.stderr.write("\n")
    return out


def fetch_plantdoc():
    tree = json.loads(gh(f"repos/{PD_REPO}/git/trees/master?recursive=1"))
    assert not tree.get("truncated"), "PlantDoc 树被截断"
    sp = collections.defaultdict(collections.Counter)
    for e in tree["tree"]:
        if e["type"] == "blob" and IMG.search(e["path"]):
            p = e["path"].split("/")
            if len(p) >= 3:
                sp[p[0]][p[1]] += 1
    return {k: dict(v) for k, v in sp.items()}


def fetch_ip102():
    import urllib.request
    url = f"https://raw.githubusercontent.com/{IP_REPO}/master/classes.txt"
    txt = urllib.request.urlopen(url, timeout=30).read().decode("utf8")
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s+(.*)$", line)
        if m:
            out.append({"index": int(m.group(1)), "name": m.group(2).strip()})
    return out


def imbalance(counts):
    v = sorted((x for x in counts.values() if x), reverse=True)
    if not v:
        return None
    return {"n_classes": len(v), "total": sum(v), "max": v[0], "min": v[-1],
            "imbalance_factor": round(v[0] / v[-1], 1),
            "median": v[len(v) // 2],
            "n_lt100": sum(1 for x in v if x < 100),
            "n_lt20": sum(1 for x in v if x < 20)}


def main():
    pv = cached("counts_plantvillage.json", fetch_plantvillage)
    pd_ = cached("counts_plantdoc.json", fetch_plantdoc)
    ip = cached("classes_ip102.json", fetch_ip102)

    pd_all = collections.Counter()
    for split in pd_.values():
        pd_all.update(split)
    pd_all = dict(pd_all)

    mapped = {k: v for k, v in PD2PV.items() if v and k in pd_all}
    unmapped_pd = [k for k in pd_all if k not in mapped]
    pv_covered = set(mapped.values())
    pv_only = sorted(set(pv) - pv_covered)

    # IP102 与 PlantVillage：逐类字面/语义比对
    pv_tokens = " ".join(pv).lower()
    ip_hits = [c["name"] for c in ip
               if any(w in pv_tokens for w in re.findall(r"[a-z]{5,}", c["name"].lower()))]

    res = {
        "exp": "1a",
        "purpose": "候选跨域基准的类别结构与长尾程度；回应 R2-1 / R1-1 的前置测量",
        "method": "GitHub tree API 读取类别目录与文件计数，不下载图像",
        "datasets": {
            "PlantVillage": {**imbalance(pv), "per_class": pv},
            "PlantDoc": {**imbalance(pd_all), "per_split":
                         {k: {"n_classes": len(v), "total": sum(v.values())} for k, v in pd_.items()},
                         "per_class": pd_all},
            "IP102": {"n_classes": len(ip), "per_class_counts": None,
                      "note": "仓库仅含 classes.txt，图像托管于 Google Drive；"
                              "逐类计数需下载数据集后方可统计"},
        },
        "overlap": {
            "PlantVillage_x_PlantDoc": {
                "mapped_pairs": len(mapped),
                "plantdoc_classes": len(pd_all),
                "plantvillage_classes": len(pv),
                "plantvillage_covered": len(pv_covered),
                "plantvillage_only": pv_only,
                "plantdoc_unmapped": unmapped_pd,
                "mapping": mapped,
            },
            "PlantVillage_x_IP102": {
                "mapped_pairs": 0,
                "rationale": "PlantVillage 为叶片真菌/细菌/病毒病害与健康状态；"
                             "IP102 为昆虫害虫物种。两者标签语义不属同一分类体系，"
                             "不存在可对齐的类别对。",
                "lexical_probe_hits": ip_hits,
            },
        },
    }
    out = REPO / "results" / "exp-1a-class-stats.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n✓ 已写入 {out.relative_to(REPO)}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ARGS = ap.parse_args()
    main()
