"""Table 9：推理期计算开销实测（参数量、FLOPs、单图时延），以及 CD-APDM 的一次性 AdaBN 开销。

    python -m tools.compute_cost --ours <cd_apdm_bestval.pt> --out <json>

- 参数量与 FLOPs 只计推理时实际经过的模块：定稿配置关闭时序分支（TFEM、LSTM、MHSA 不参与），
  CD-APDM 的推理路径因此是 ResNet-50 骨干 + 分类头 + 温度；logit 调整与尾类重标定是逐类的标量运算，计入时延。
- FLOPs 用 torch.utils.flop_counter（乘加计为 2 FLOPs），输入 1×3×224×224。
- 时延：batch 1，预热 50 次后计时 1,000 次前向（CUDA 同步），报告中位数与均值；GPU 型号写入结果。
- AdaBN：在全部 2,566 张目标图上重估 BN 统计量的一次性耗时（batch 64，含数据加载），不计入单图时延。
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from src.baselines.net import BaselineNet
from src.datasets.splits import StripLabels, build_datasets
from src.inference import adabn, class_priors_from_counts, load_model
from src.modules.ss_plam import logit_adjustment
from src.utils.config import load_config


def n_params(modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def flops(fn, x):
    with FlopCounterMode(display=False) as fc:
        fn(x)
    return fc.get_total_flops()


@torch.no_grad()
def latency(fn, x, warmup=50, iters=1000):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(ts)), "mean_ms": float(np.mean(ts))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True); ap.add_argument("--config", default="configs/cd_apdm.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    _, _, tgt, ci = build_datasets(cfg, temporal=None)
    x = torch.randn(1, 3, 224, 224, device=dev)
    res = {"gpu": torch.cuda.get_device_name(0), "input": [1, 3, 224, 224], "rows": {}}

    for name, backbone in (("resnet50", "resnet50"), ("swin_b", "swin_b")):
        m = BaselineNet(ci.num_classes, backbone, pretrained=False).to(dev).eval()
        res["rows"][name] = {"params_M": n_params([m]) / 1e6, "gflops": flops(m, x) / 1e9, **latency(m, x)}

    model, ck = load_model(a.ours, cfg, 16, dev)
    model.use_temporal = False; model.eval()
    pri = class_priors_from_counts(ck["class_counts"]).to(dev)
    counts = torch.tensor(ck["class_counts"], dtype=torch.float32, device=dev).clamp_min(1)
    tail = torch.zeros_like(counts); tail[ck["tail_classes"]] = 1
    boost = 1 + cfg["acrm"]["gamma_boost"] * (counts.max() / counts - 1) * tail

    def ours(inp):   # 定稿推理路径：骨干 + 分类头 + 温度 → logit 调整 → softmax → 尾类重标定并归一化
        p = torch.softmax(logit_adjustment(model(inp), pri, cfg["logit_adjust"]["tau"]), -1) * boost
        return p / p.sum(-1, keepdim=True)

    active = [model.backbone, model.classifier, model.temperature]
    res["rows"]["cd_apdm"] = {"params_M": n_params(active) / 1e6, "gflops": flops(ours, x) / 1e9, **latency(ours, x),
                              "params_M_all_modules_incl_disabled_temporal": n_params([model]) / 1e6}
    loader = DataLoader(StripLabels(tgt), batch_size=64, num_workers=6)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.enable_grad():
        adabn(model, loader, dev)
    torch.cuda.synchronize()
    res["rows"]["cd_apdm"]["adabn_one_time_s"] = time.perf_counter() - t0
    res["rows"]["cd_apdm"]["adabn_images"] = len(tgt)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
