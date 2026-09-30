#!/usr/bin/env python3
"""核验 Proposition 2 的 EMA 残差界。

稿件给出 `(1-η)·σ_c/√B_c`，并在正文代入 η=0.95 得 0.05σ_c/√B_c。
EMA 稳态方差的标准结果是 ((1-η)/(1+η))·σ²/B，故残差标准差应为
`√((1-η)/(1+η))·σ_c/√B_c`，η=0.95 时系数 0.160 而非 0.05。

差 3.2 倍不是小数点问题：稿件据此称阈值抖动约 ±0.01–0.02 并断言
"ensuring stable pseudo-label selection"，该结论的余量会被吃掉大半。

本脚本用蒙特卡洛直接测稳态标准差，判定哪个界成立。
"""
import argparse, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def run(eta, p, B, steps, reps, burn, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros(reps)
    tail = []
    for t in range(steps):
        x = rng.binomial(B, p, size=reps) / B
        a = eta * a + (1 - eta) * x
        if t >= burn:
            tail.append(a.copy())
    tail = np.asarray(tail)
    sigma_c = math.sqrt(p * (1 - p))          # 单样本指示变量标准差
    per_step = sigma_c / math.sqrt(B)          # 每步观测的标准差
    return {
        "eta": eta, "p": p, "batch": B, "steps": steps, "reps": reps, "seed": seed,
        "sigma_c": round(sigma_c, 6),
        "per_step_sd": round(per_step, 6),
        "empirical_sd": round(float(tail.std()), 6),
        "manuscript_bound": round((1 - eta) * per_step, 6),
        "standard_result": round(math.sqrt((1 - eta) / (1 + eta)) * per_step, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=400)
    args = ap.parse_args()

    cases = []
    for eta in (0.9, 0.95, 0.99):
        for p, B in ((0.8, 32), (0.6, 16), (0.95, 64)):
            r = run(eta, p, B, args.steps, args.reps, args.steps // 10, args.seed)
            r["ratio_emp_over_manuscript"] = round(r["empirical_sd"] / r["manuscript_bound"], 3)
            r["ratio_emp_over_standard"] = round(r["empirical_sd"] / r["standard_result"], 3)
            cases.append(r)

    print(f"{'η':>5}{'p':>6}{'B':>5}{'实测':>10}{'稿件界':>10}{'标准结果':>11}"
          f"{'实测/稿件':>11}{'实测/标准':>11}")
    for r in cases:
        print(f"{r['eta']:>5}{r['p']:>6}{r['batch']:>5}{r['empirical_sd']:>10.5f}"
              f"{r['manuscript_bound']:>10.5f}{r['standard_result']:>11.5f}"
              f"{r['ratio_emp_over_manuscript']:>11.2f}{r['ratio_emp_over_standard']:>11.3f}")

    worst = max(c["ratio_emp_over_manuscript"] for c in cases)
    dev = max(abs(c["ratio_emp_over_standard"] - 1) for c in cases)
    out = {
        "exp": "ACRM Proposition 2 残差界核验",
        "manuscript_claim": "(1-η)·σ_c/√B_c",
        "standard_result": "√((1-η)/(1+η))·σ_c/√B_c",
        "verdict": ("稿件界不成立：实测稳态标准差全面超出该界，"
                    f"最大超出 {worst:.2f} 倍；标准结果与实测吻合，最大偏差 {dev:.1%}"),
        "cases": cases,
    }
    p = ROOT / "results/exp-acrm-ema-variance.json"
    p.parent.mkdir(exist_ok=True)
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=2)
    print(f"\n{out['verdict']}")
    print(f"结果：{p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
