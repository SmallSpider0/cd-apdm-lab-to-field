"""诊断用（不产出论文数字）：同一 checkpoint 在目标域上，逐项叠加推理期处理后的指标。

    python -m tools.diagnose_adaptation --config configs/cd_apdm.yaml --checkpoint <ckpt>

输出：raw（仅模型，含其训练温度）；+logit 调整 τ（按源域类别先验）；预测类别分布。
"""
import argparse, json
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.datasets.splits import build_datasets
from src.inference import class_priors_from_counts, load_model
from src.modules.ss_plam import logit_adjustment
from src.utils.config import load_config
from src.utils.metrics import evaluate_all

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True)
a = ap.parse_args()
cfg = load_config(a.config); dev = torch.device("cuda")
_, _, tgt, ci = build_datasets(cfg, temporal=None)
model, ck = load_model(a.checkpoint, cfg, 16, dev); model.use_temporal = False; model.eval()
pri = class_priors_from_counts(ck["class_counts"]).to(dev)
L, Y = [], []
with torch.no_grad():
    for b in DataLoader(tgt, batch_size=128, num_workers=8):
        L.append(model(b["image"].to(dev), b["sequence"].to(dev)).float().cpu()); Y.append(b["label"])
L, Y = torch.cat(L), torch.cat(Y).numpy()
tail = ck["tail_classes"]; out = {}
for tau in (0.0, 0.5, 1.0):
    lg = logit_adjustment(L.to(dev), pri, tau=tau).cpu() if tau else L
    p = F.softmax(lg, -1).numpy(); pred = p.argmax(-1)
    m = evaluate_all(pred, Y, p, ci.num_classes, tail)
    share_tail = float(np.isin(pred, tail).mean()); true_tail = float(np.isin(Y, tail).mean())
    out[f"tau={tau}"] = {**{k: round(v, 3) for k, v in m.items()},
                         "pred_share_tail": round(share_tail, 3), "true_share_tail": round(true_tail, 3),
                         "top5_pred_classes": np.bincount(pred, minlength=ci.num_classes).argsort()[::-1][:5].tolist()}
print(json.dumps(out, indent=1))
