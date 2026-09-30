"""诊断：WB+DANN 第二阶段为何使源域验证 Top-1 从约 95% 跌到约 58%（exp2/wb_dann-s0）。

    python -m tools.wb_diag --stage1 <含 model_stage1.pt 的目录> --seed 0 --out <json>

起点是同一 seed 的第一阶段末权重（src.baselines.run --method wb_dann --save_stage1 产出）。
每个变体都从这份权重出发、用同一数据顺序跑完整个第二阶段（configs/baselines.yaml 的 stage2），
与 run.py 的实现相比**每次只改一个因素**：

    official    与 run.py 完全相同：train 模式、只用源域批、CB 损失、wd 0.1、MaxNorm
    frozen_bn   第二阶段整网处于 eval 模式（BN 运行统计量冻结、Dropout 关闭）
    mixed_bn    train 模式，但前向时拼入同样大小的目标域批（损失只算源域部分），
                使 BN 统计量与第一阶段 DANN 的混合批口径一致
    wd0         weight_decay 取 0
    ce          CB 损失换成普通交叉熵
另有两个不做梯度更新的对照，只在 train 模式下前向一遍源域训练集以更新 BN 运行统计量：
    bn_src      只用源域批（= official 第一个 epoch 中 BN 的变化，不含最后一层的更新）
    bn_mixed    源域批拼目标域批（第一阶段的口径）

只记录**源域**验证集指标与最后一层的权重范数、偏置范围 —— 不看任何目标域标签，
诊断结论因此不受测试集影响。
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.baselines.losses import ClassBalancedSoftmaxLoss, MaxNormPGD
from src.baselines.net import BaselineNet
from src.baselines.run import forever, metrics_of, predict, target_train_set
from src.datasets.image_dataset import build_train_transform
from src.datasets.splits import build_datasets
from src.utils.config import load_config
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed

VARIANTS = ("official", "frozen_bn", "mixed_bn", "wd0", "ce")


def layer_stats(layer):
    n = layer.weight.detach().norm(dim=1)
    b = layer.bias.detach()
    return {"w_norm_min": float(n.min()), "w_norm_med": float(n.median()), "w_norm_max": float(n.max()),
            "bias_min": float(b.min()), "bias_max": float(b.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="含 model_stage1.pt 的目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="configs/cd_apdm.yaml")
    ap.add_argument("--overlay", default="configs/baselines.yaml")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="冒烟：每个 epoch 只跑 2 个批、第二阶段只跑 1 个 epoch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.base)
    s2 = yaml.safe_load(open(a.overlay))["baselines"]["wb_dann"]["stage2"]
    dev = torch.device("cuda")
    bs, size = cfg["train"]["batch_size"], cfg["data"].get("image_size", 224)
    src_train, src_val, _, ci = build_datasets(cfg, temporal=None)
    C, tail = ci.num_classes, ci.tail_classes
    ck = torch.load(f"{a.stage1}/model_stage1.pt", map_location="cpu", weights_only=False)
    assert ck["method"] == "wb_dann" and ck["seed"] == a.seed, (ck["method"], ck["seed"])
    init = ck["model"]

    def loader(ds, shuffle, drop_last=False):
        g = torch.Generator().manual_seed(a.seed)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=a.num_workers,
                          drop_last=drop_last, pin_memory=True, generator=g)

    val_loader = loader(src_val, False)
    tgt_ds = target_train_set(cfg, ci, build_train_transform(size))

    def fresh():
        m = BaselineNet(C, "resnet50", pretrained=False).to(dev)
        m.load_state_dict(init)
        return m

    def src_val_metrics(m):
        p, y, _ = predict(m, val_loader, dev)
        r = metrics_of(p, y, C, tail)
        return {"top1": r["top1"], "tail_recall": r["tail_recall"], "macro_f1": r["macro_f1"]}

    report = {"stage1_dir": a.stage1, "seed": a.seed, "stage2_hparams": s2}
    m0 = fresh()
    report["stage1"] = {"src_val": src_val_metrics(m0), **layer_stats(m0.last_layer)}
    print(json.dumps({"stage1": report["stage1"]}, ensure_ascii=False), flush=True)

    # ---- 对照：只更新 BN 运行统计量，不做梯度更新
    for name, mixed in (("bn_src", False), ("bn_mixed", True)):
        set_seed(a.seed)
        m = fresh()
        m.train()
        tgt_iter = forever(loader(tgt_ds, True, drop_last=True)) if mixed else None
        with torch.no_grad():
            for i, b in enumerate(loader(src_train, True, drop_last=True)):
                if a.smoke and i >= 2:
                    break
                x = b["image"].to(dev, non_blocking=True)
                if mixed:
                    x = torch.cat([x, next(tgt_iter)["image"].to(dev, non_blocking=True)])
                m(x)
        report[name] = {"src_val": src_val_metrics(m)}
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)

    # ---- 第二阶段变体（除所改因素外与 run.py 的 WB 第二阶段逐行一致）
    for v in a.variants.split(","):
        set_seed(a.seed)
        m = fresh()
        for p in m.parameters():
            p.requires_grad_(False)
        layer = m.last_layer
        for p in layer.parameters():
            p.requires_grad_(True)
        pgd = MaxNormPGD(layer, thresh=s2["maxnorm_thresh"])
        cb = ClassBalancedSoftmaxLoss(src_train.class_counts, beta=s2["cb_beta"]).to(dev)
        opt = build_optimizer(list(layer.parameters()), s2["optim"], s2["lr"],
                              0.0 if v == "wd0" else s2["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, s2["epochs"], eta_min=0.0)
        train_loader = loader(src_train, True, drop_last=True)
        tgt_iter = forever(loader(tgt_ds, True, drop_last=True)) if v == "mixed_bn" else None
        rows, t0 = [], time.time()
        for k in range(1, (1 if a.smoke else s2["epochs"]) + 1):
            m.eval() if v == "frozen_bn" else m.train()
            run_loss, n = 0.0, 0
            for i, b in enumerate(train_loader):
                if a.smoke and i >= 2:
                    break
                x, y = b["image"].to(dev, non_blocking=True), b["label"].to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if v == "mixed_bn":
                    xt = next(tgt_iter)["image"].to(dev, non_blocking=True)
                    logits = m(torch.cat([x, xt]))[: x.size(0)]
                else:
                    logits = m(x)
                loss = F.cross_entropy(logits.float(), y) if v == "ce" else cb(logits.float(), y)
                loss.backward()
                opt.step()
                pgd.project()
                run_loss += float(loss.detach()); n += 1
            sch.step()
            row = {"epoch": k, "loss": run_loss / max(n, 1), "src_val": src_val_metrics(m),
                   **layer_stats(layer), "elapsed_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(json.dumps({v: row}, ensure_ascii=False), flush=True)
        report[v] = {"maxnorm_limit": pgd.limit, "epochs": rows}
        with open(a.out, "w") as f:           # 每个变体跑完即落盘，中途失败也保留已完成部分
            json.dump(report, f, indent=2, ensure_ascii=False)

    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
