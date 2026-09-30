#!/usr/bin/env python3
"""重算**原始**（修复前、按 10% 比例切分的）验证集的泄漏程度，写入审计记录。

稿件 4.1.3 与回复信 R1-1 引用「修复前 49/2,372 张验证图（2.07%）与训练集同叶片」。
该数字原由 exp5_apply_groups.py 在未修复的划分上算出，但脚本重跑后 source_before_repair
度量的是已修复的划分（为 0），原始数字因此从审计记录中消失。这里从 git 历史取回
原始划分文件（提交 f06c00b，即任务 5 修复之前），用同一套分组逻辑重算，
使该数字始终可复现。

    python scripts/analysis/exp5_original_split.py
"""
import contextlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMIT = "f06c00b"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp5_apply_groups  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "dataset/splits"
        d.mkdir(parents=True)
        for f in ("plantvillage_lt_source.csv", "plantdoc_target.csv"):
            blob = (REPO / "dataset/splits" / f.replace(".csv", f"_{COMMIT}.csv")).read_bytes()
            (d / f).write_bytes(blob)
            if f == "plantvillage_lt_source.csv":
                lt_csv = blob
        cwd = os.getcwd()
        buf = io.StringIO()
        try:
            os.chdir(tmp)
            sys.argv = ["exp5_apply_groups.py", "--dry-run"]
            with contextlib.redirect_stdout(buf):
                exp5_apply_groups.main()
        finally:
            os.chdir(cwd)
    text = buf.getvalue()
    report = json.loads(text[: text.rindex("}") + 1])
    rec = {"split_commit": COMMIT, "construction": "LT 保留集按 10% 比例切分（任务 5 修复前）",
           **report["source_before_repair"]}
    out = REPO / "results/exp-5-leakage-audit.json"
    audit = json.loads(out.read_text())
    audit["original_proportional_split"] = rec
    # 叶片编号覆盖率的两种分母（稿件 4.1、4.8 与 S2 用 PlantVillage-LT 口径）：
    # 修复前划分文件恰为长尾集 23,688 张；现行划分另含平衡验证集从长尾集外取的 1,200 张。
    import csv
    cov = {}
    for name, rows in (("plantvillage_lt", list(csv.DictReader(io.StringIO(lt_csv.decode())))),
                       ("source_split_incl_balanced_val",
                        list(csv.DictReader(open(REPO / "dataset/splits/plantvillage_lt_source.csv"))))):
        n = sum(1 for r in rows if exp5_apply_groups.leaf_key(r))
        cov[name] = {"images": len(rows), "with_leaf_id": n, "pct": round(100.0 * n / len(rows), 2)}
    cov["note"] = ("稿件引用 plantvillage_lt 口径（1,660 / 23,688 = 7.01%）；source.images_with_leaf_id 与 "
                   "leaf_id_coverage_pct 按现行划分文件全部 24,888 行计。")
    audit["leaf_id_coverage"] = cov
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k: rec[k] for k in ("val_total", "val_contaminated", "val_contaminated_pct",
                                          "caught_by_phash_alone")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
