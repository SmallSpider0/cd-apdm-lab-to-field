#!/usr/bin/env python3
"""作者代码审阅用的独立复核（只读，不改任何文件，本机 CPU 约半分钟）。

    $AGRI_PYTHON scripts/analysis/review_crosscheck.py      # 在仓库根目录运行

四项检查
  0  由逐图概率重跑 Table 1 汇总，结果与 results/exp-2-table1.json 完全相同
  1  Macro-F1 的实现与 sklearn 一致（含"某类从未被预测"的情形）
  2  Holm 校正的实现与教科书定义一致
  3  结果文件中的 95 个配对 t 检验 p 值可由逐 seed 数值用 scipy 复算
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from src.utils.metrics import macro_f1  # noqa: E402
from aggregate_table1 import holm  # noqa: E402

REF = ROOT / "results/exp-2-table1.json"

# 0. 重跑汇总
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "t1.json"
    remote = ROOT / "results/remote"
    subprocess.run([sys.executable, str(ROOT / "scripts/analysis/aggregate_table1.py"),
                    str(remote / "exp2"), str(remote / "exp2-ours"), str(remote / "exp2-adabn"),
                    "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    assert json.loads(out.read_text()) == json.loads(REF.read_text()), "重跑结果与 results/exp-2-table1.json 不同"
print("0 由逐图概率重跑 Table 1 汇总，与结果文件完全相同: OK")

rng = np.random.default_rng(0)

# 1. Macro-F1
for _ in range(200):
    y = rng.integers(0, 28, 500)
    p = rng.integers(0, 20, 500)          # 20–27 类从未被预测
    cls = list(range(27))
    ref = f1_score(y, p, labels=cls, average="macro", zero_division=0)
    assert abs(macro_f1(p, y, 28, classes=cls) - ref) < 1e-12
print("1 Macro-F1 与 sklearn 一致: OK")


# 2. Holm：第 r 小的 p 乘以 (m − r)，取累计最大并截断到 1
def holm_ref(pv):
    m, adj, run = len(pv), np.empty(len(pv)), 0.0
    for r, i in enumerate(np.argsort(pv)):
        run = max(run, min(1.0, (m - r) * pv[i]))
        adj[i] = run
    return adj


for _ in range(200):
    pv = rng.uniform(0, 0.2, 19)
    ours = holm(dict(enumerate(pv)))
    assert np.allclose([ours[i] for i in range(19)], holm_ref(pv))
print("2 Holm 校正与定义一致: OK")

# 3. 配对 t 检验（主口径：bestval、保留冲突图像、λ 按开发集选取）
v = json.loads(REF.read_text())["views"][0]
tab, tests, n = v["table"], v["paired_tests_vs_reference"], 0
for key, e in tests.items():
    if "p_t" not in e:
        continue
    m, k = key.split("|")
    seeds = sorted(tab["ours"][k]["per_seed"])
    a = [tab["ours"][k]["per_seed"][s] for s in seeds]
    b = [tab[m][k]["per_seed"][s] for s in seeds]
    assert abs(stats.ttest_rel(a, b).pvalue - e["p_t"]) < 1e-12
    n += 1
print(f"3 {n} 个配对 t 检验的 p 值可由逐 seed 数值复算: OK")
