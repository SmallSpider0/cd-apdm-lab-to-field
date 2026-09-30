#!/usr/bin/env python3
"""离线组合：基线 logits ×（AdaBN 有 / 无）×（logit 调整 τ）×（ACRM 尾类重标定 γ），只在开发集上评估。

    python scripts/analysis/adabn_decompose.py <baseline_adabn.npz> [...] [--tau 0.5] [--gamma 0.005]

组合公式与 src/modules/ss_plam.logit_adjustment、src/modules/acrm.ACRM.rectify 一致：
    z' = z − τ·log π_c；p = softmax(z')；p̃_c ∝ p_c·(1 + γ·(n_max/n_c − 1)·1[c ∈ tail])
每个组合另存为 eval_probs.py 可读的 npz（probs / labels / image_ids），结果打印为 JSON 行。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_probs import evaluate  # noqa: E402


def combine(z, counts, tail, tau, gamma):
    counts = np.where(counts <= 0, 1, counts).astype(float)
    z = z - tau * np.log(counts / counts.sum())
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    mask = np.zeros(len(counts)); mask[tail] = 1
    p = p * (1 + gamma * (counts.max() / counts - 1) * mask)
    return p / p.sum(1, keepdims=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.005)
    a = ap.parse_args()
    for f in a.files:
        d = np.load(f, allow_pickle=False)
        tag = f"{d['method']}-s{d['seed']}-{d['ckpt']}"
        for bn in ("plain", "adabn"):
            for tau, gamma, name in ((0, 0, "none"), (a.tau, 0, f"LA{a.tau}"), (a.tau, a.gamma, f"LA{a.tau}+g{a.gamma}")):
                p = combine(d[f"logits_{bn}"], d["class_counts"], d["tail"], tau, gamma)
                out = Path(f).with_name(f"{Path(f).stem}__{bn}__{name}.npz")
                np.savez_compressed(out, probs=p.astype(np.float32), labels=d["labels"], image_ids=d["image_ids"])
                r = evaluate(out, "dev")
                print(json.dumps({"run": tag, "bn": bn, "post": name, **{k: r[k] for k in ("top1", "tail_recall", "macro_f1", "ece", "top3_pred_share")}}, ensure_ascii=False))
