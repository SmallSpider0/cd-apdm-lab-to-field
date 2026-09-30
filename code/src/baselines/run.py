"""Table 1 对比基线的统一入口。

    python -m src.baselines.run --method dann --seed 1
    python -m src.baselines.run --method cotta --seed 1 --init_from <source_only 产物目录> [--init_ckpt final]

训练预算（epochs、batch、数据、增强）与 CD-APDM 取自同一份
``configs/cd_apdm.yaml``；方法专属超参取自 ``configs/baselines.yaml``。

产物（写入 $RUN_DIR，本机则写入 checkpoints/baselines/<method>-s<seed>）：
    history.json          每个 epoch 的源域验证与目标域指标
    probs_final.npz       最后一个 epoch 的目标域概率、标签、图像 ID
    probs_bestval.npz     源域验证 Top-1 最高的 epoch 的同上
    summary.json          两种选型规则下的指标、超参、与官方实现的偏离
    model_final.pt        最终权重
    model_bestval.pt      源域验证 Top-1 最高的 epoch 的权重

选型协议（fix-experiment-protocol，2026-09-21 定）：主结果按**源域验证集**选型，
附录给"取最后一个 epoch"。两种规则都记录，排除标签冲突图像等评测口径也可以
从保存的概率离线重算。CoTTA 是测试时方法，本身不选 epoch；它的起点
（source_only 的哪份权重）随规则走：主结果用 model_bestval.pt，附录用 model_final.pt。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from ..datasets.image_dataset import build_train_transform
from ..datasets.splits import SplitFileDataset, StripLabels, build_datasets  # noqa: F401
from ..utils.config import load_config
from ..utils.metrics import evaluate_all
from ..utils.optim import backbone_param_groups, build_optimizer, build_scheduler
from ..utils.resume import atomic_save, clear_resume, load_resume, rng_state, set_rng_state
from ..utils.seed import pick_device, set_seed
from .losses import ClassBalancedSoftmaxLoss, FocalLoss, MaxNormPGD, dann_lambda
from .net import BaselineNet, DomainDiscriminator, grad_reverse
from .semi import CoTTA, FlexMatchThreshold, flexmatch_views

METHODS = ("source_only", "focal", "dann", "wb_dann", "flexmatch", "swin_b", "cotta")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ data utils
def forever(loader):
    while True:
        for b in loader:
            yield b


def target_train_set(cfg, class_index, transform):
    d = cfg["data"]
    ds = SplitFileDataset(d.get("target_split", "plantdoc_target.csv"), "plantdoc", 1,
                          class_index=class_index, split=None, train=True,
                          image_size=d.get("image_size", 224), transform=transform)
    return StripLabels(ds)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y, ids = [], [], []
    for b in loader:
        logits = model(b["image"].to(device, non_blocking=True))
        P.append(F.softmax(logits.float(), dim=-1).cpu())
        Y.append(b["label"])
        ids.extend(b["image_id"])
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), ids


def metrics_of(probs, labels, C, tail):
    return evaluate_all(probs.argmax(-1), labels, probs, C, tail)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", default="configs/cd_apdm.yaml")
    ap.add_argument("--overlay", default="configs/baselines.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="仅冒烟用；正式运行取共享配置")
    ap.add_argument("--batch_size", type=int, default=None, help="仅冒烟用")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_train_batches", type=int, default=None, help="仅冒烟用")
    ap.add_argument("--max_eval_batches", type=int, default=None, help="仅冒烟用")
    ap.add_argument("--init_from", default=None, help="CoTTA：source_only 同 seed 的产物目录")
    ap.add_argument("--init_ckpt", default="bestval", choices=["bestval", "final"],
                    help="CoTTA 的起点：bestval（主结果，源域验证选型）或 final（附录）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_stage1", action="store_true",
                    help="诊断用（tools/wb_diag.py）：WB 第一阶段结束时另存 model_stage1.pt；不影响其余行为")
    args = ap.parse_args()

    cfg = load_config(args.base)
    ov = yaml.safe_load(open(args.overlay))["baselines"]
    hp = ov[args.method]
    smoke = any(v is not None for v in (args.epochs, args.batch_size, args.max_train_batches))

    set_seed(args.seed)
    device = pick_device(cfg["device"])
    epochs = args.epochs or cfg["train"]["epochs"]
    bs = args.batch_size or cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"] if args.num_workers is None else args.num_workers
    size = cfg["data"].get("image_size", 224)
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"

    out = Path(args.out or os.environ.get("RUN_DIR") or f"checkpoints/baselines/{args.method}-s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and (out / "probs_final.npz").exists():
        print(f"[baseline] {out} 已完成（summary.json 存在），跳过")
        return
    print(f"[baseline] method={args.method} seed={args.seed} device={device} out={out}"
          + ("  [冒烟：数字不可用]" if smoke else ""))

    src_train, src_val, tgt_eval, ci = build_datasets(cfg, temporal=None)
    C, tail = ci.num_classes, ci.tail_classes
    print(f"[baseline] source train={len(src_train)} val={len(src_val)} target={len(tgt_eval)} "
          f"classes={C} tail={len(tail)}")

    def loader(ds, shuffle, drop_last=False, batch=None):
        g = torch.Generator().manual_seed(args.seed)
        return DataLoader(ds, batch_size=batch or bs, shuffle=shuffle, num_workers=nw,
                          drop_last=drop_last, pin_memory=device.type == "cuda", generator=g)

    val_loader = loader(src_val, False)
    tgt_loader = loader(tgt_eval, False)

    def evaluate_both(model):
        vb = args.max_eval_batches
        if vb:
            from itertools import islice

            class _Cap:
                def __init__(s, l):
                    s.l = l

                def __iter__(s):
                    return islice(iter(s.l), vb)
            pv = predict(model, _Cap(val_loader), device)
            pt = predict(model, _Cap(tgt_loader), device)
        else:
            pv = predict(model, val_loader, device)
            pt = predict(model, tgt_loader, device)
        return metrics_of(pv[0], pv[1], C, tail), metrics_of(pt[0], pt[1], C, tail), pt

    history, best = [], {"val": -1.0}
    t0 = time.time()

    def record(epoch, stage, loss, extra, model):
        mv, mt, pt = evaluate_both(model)
        row = {"epoch": epoch, "stage": stage, "loss": loss, "src_val": mv, "tgt": mt,
               "cross_domain_gap": mv["top1"] - mt["top1"], "elapsed_s": round(time.time() - t0, 1), **extra}
        history.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in
                          {"epoch": epoch, "stage": stage, "loss": loss, "src_top1": mv["top1"],
                           "tgt_top1": mt["top1"], "tgt_tail": mt["tail_recall"], "tgt_ece": mt["ece"],
                           **extra}.items()}))
        return mv, mt, pt

    def save_probs(name, pt, epoch):
        tmp = out / (name + ".part.npz")      # 先写临时文件再改名，重启时不留半截文件
        np.savez_compressed(tmp, probs=pt[0].astype(np.float16), labels=pt[1],
                            image_ids=np.array(pt[2]), epoch=epoch)
        os.replace(tmp, out / name)

    # ================================================================ CoTTA
    if args.method == "cotta":
        if not args.init_from:
            raise SystemExit("CoTTA 需要 --init_from 指向同 seed 的 source_only 产物目录")
        src_dir = Path(args.init_from)
        ck = torch.load(src_dir / f"model_{args.init_ckpt}.pt", map_location="cpu")
        if ck["method"] != "source_only" or ck["seed"] != args.seed:
            raise SystemExit(f"--init_from 必须是同 seed 的 source_only：得到 {ck['method']} seed={ck['seed']}")
        model = BaselineNet(C, "resnet50", pretrained=False).to(device)
        model.load_state_dict(ck["model"])
        mv, _, _ = evaluate_both(model)
        tta = CoTTA(model, MEAN, STD, lr=hp["lr"], mt=hp["mt"], rst=hp["rst"], ap=hp["ap"],
                    n_aug=hp["n_aug"], seed=args.seed)
        # 流式适配：目标域按 seed 打乱成流。划分文件按类别排序，原序会让每批
        # 几乎只含一个类，既破坏 BN 批统计，也不是现实的数据流。
        stream = loader(tgt_eval, True, batch=bs)
        P, Y, ids = [], [], []
        for i, b in enumerate(stream):
            if args.max_eval_batches and i >= args.max_eval_batches:
                break
            logits = tta.step(b["image"].to(device))
            P.append(F.softmax(logits.float(), dim=-1).detach().cpu())
            Y.append(b["label"])       # 只用于评测：标签不进入 tta.step
            ids.extend(b["image_id"])
        pt = (torch.cat(P).numpy(), torch.cat(Y).numpy(), ids)
        mt = metrics_of(pt[0], pt[1], C, tail)
        history.append({"epoch": 1, "stage": "online-tta", "src_val": mv, "tgt": mt,
                        "cross_domain_gap": mv["top1"] - mt["top1"],
                        "aug_average_triggered": f"{tta.triggered}/{tta.batches}"})
        print(json.dumps({"tgt_top1": mt["top1"], "tgt_tail": mt["tail_recall"], "tgt_ece": mt["ece"],
                          "aug_average_triggered": f"{tta.triggered}/{tta.batches}"}))
        save_probs("probs_final.npz", pt, 1)
        save_probs("probs_bestval.npz", pt, 1)
        _finish(out, args, hp, history, best_epoch=1, final_epoch=1, smoke=smoke, t0=t0,
                extra={"init_from": str(src_dir), "init_ckpt": args.init_ckpt,
                       "init_epoch": ck.get("epoch")})
        return

    # ================================================= 训练式方法的公共部分
    backbone = "swin_b" if args.method == "swin_b" else "resnet50"
    model = BaselineNet(C, backbone).to(device)

    def save_model(name, epoch):
        atomic_save({"model": model.state_dict(), "method": args.method, "seed": args.seed,
                     "backbone": backbone, "num_classes": C, "epoch": epoch}, out / name)
    train_loader = loader(src_train, True, drop_last=True)
    steps = len(train_loader) if args.max_train_batches is None else min(args.max_train_batches, len(train_loader))
    total_steps = steps * epochs

    disc = None
    params = list(model.parameters())
    if args.method in ("dann", "wb_dann"):
        disc = DomainDiscriminator(model.feature_dim, hp.get("disc_hidden", 1024)).to(device)
        params += list(disc.parameters())
    tgt_iter = None
    if args.method in ("dann", "wb_dann"):
        tgt_iter = forever(loader(target_train_set(cfg, ci, build_train_transform(size)), True, drop_last=True))
    flex = None
    if args.method == "flexmatch":
        tds = target_train_set(cfg, ci, flexmatch_views(size, MEAN, STD))
        tgt_iter = forever(loader(tds, True, drop_last=True, batch=bs * hp["uratio"]))
        flex = FlexMatchThreshold(len(tds), C, hp["p_cutoff"], hp["thresh_warmup"], device=device)

    opt_cfg = hp.get("optim", {})
    base_lr = opt_cfg.get("lr", cfg["train"]["lr"])
    # 判别式微调只用于 ResNet-50 方法（决策记录 4）；Swin-B 保持其标准 AdamW 设置
    mult = cfg["train"].get("backbone_lr_mult", 1.0) if backbone == "resnet50" else 1.0
    optim = build_optimizer(backbone_param_groups(model.backbone, params, base_lr, mult),
                            opt_cfg.get("name", cfg["train"].get("optimizer", "adam")), base_lr,
                            opt_cfg.get("weight_decay", cfg["train"]["weight_decay"]))
    print("[baseline] lr: " + ", ".join(f"{g['name']}={g['lr']:g}" for g in optim.param_groups))
    sched = build_scheduler(optim, cfg["train"].get("scheduler", "cosine"), epochs,
                            int(cfg["train"].get("warmup_epochs", 0)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    focal = FocalLoss(hp.get("gamma", 2.0)) if args.method == "focal" else None

    step_g, start_epoch, resumed_at = 0, 1, []
    ck = load_resume(out, method=args.method, seed=args.seed, epochs=epochs)
    if ck is not None:
        model.load_state_dict(ck["model"])
        if disc is not None:
            disc.load_state_dict(ck["disc"])
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        if flex is not None:
            flex.selected.copy_(ck["flex"]["selected"].to(flex.selected.device))
            flex.classwise_acc.copy_(ck["flex"]["classwise_acc"].to(flex.classwise_acc.device))
        train_loader.generator.set_state(ck["train_gen"])
        set_rng_state(ck["rng"])
        history, best, step_g = ck["history"], ck["best"], ck["step_g"]
        last_pt, last_epoch = ck["last_pt"], ck["epoch"]
        resumed_at = ck["resumed_at"] + [ck["epoch"] + 1]
        start_epoch = ck["epoch"] + 1
        print(f"[baseline] 从 epoch {start_epoch} 续跑（resume.pt）")

    def save_resume(epoch):
        atomic_save({"method": args.method, "seed": args.seed, "epochs": epochs, "epoch": epoch,
                     "model": model.state_dict(), "disc": disc.state_dict() if disc is not None else None,
                     "optim": optim.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "flex": ({"selected": flex.selected.cpu(), "classwise_acc": flex.classwise_acc.cpu()}
                              if flex is not None else None),
                     "train_gen": train_loader.generator.get_state(), "rng": rng_state(),
                     "history": history, "best": best, "step_g": step_g, "last_pt": last_pt,
                     "resumed_at": resumed_at}, out / "resume.pt")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if disc is not None:
            disc.train()
        run_loss, extra_acc = 0.0, {}
        for i, b in enumerate(train_loader):
            if i >= steps:
                break
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
            progress = step_g / max(total_steps - 1, 1)
            optim.zero_grad(set_to_none=True)
            n_s = x.size(0)

            if args.method in ("dann", "wb_dann"):
                # 与官方实现（如 Transfer-Learning-Library）一致：源域与目标域拼成
                # 一个批做前向，使 BN 统计量来自混合批，而不是各域各自一套。
                xt = next(tgt_iter)["image"].to(device, non_blocking=True)
                lam = dann_lambda(progress)
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits_all, feat_all = model(torch.cat([x, xt]), return_features=True)
                    d_all = disc(grad_reverse(feat_all, lam))
                logits = logits_all[:n_s]
                loss = F.cross_entropy(logits.float(), y)
                d_all = d_all.float()
                dom = torch.cat([torch.zeros(n_s, device=device), torch.ones(xt.size(0), device=device)])
                d_loss = F.binary_cross_entropy_with_logits(d_all, dom)
                loss = loss + hp.get("trade_off", 1.0) * d_loss
                with torch.no_grad():
                    acc = ((d_all > 0).float() == dom).float().mean()
                extra_acc.setdefault("domain_acc", []).append(float(acc))
                extra_acc.setdefault("grl_lambda", []).append(lam)

            elif args.method == "flexmatch":
                # 与 USB 的 use_cat=True 一致：有标注、弱视图、强视图拼成一个批前向；
                # 弱视图的输出 detach 后用于生成伪标签。
                tb = next(tgt_iter)
                xw, xs = tb["image"]
                idx = tb["index"].to(device)
                xw, xs = xw.to(device, non_blocking=True), xs.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits_all = model(torch.cat([x, xw, xs]))
                logits_all = logits_all.float()
                logits = logits_all[:n_s]
                lw, ls = logits_all[n_s:].chunk(2)
                loss = F.cross_entropy(logits, y)
                mask, pseudo = flex.mask(F.softmax(lw.detach(), dim=-1), idx)
                u_loss = (F.cross_entropy(ls, pseudo, reduction="none") * mask).mean()
                loss = loss + hp["lambda_u"] * u_loss
                extra_acc.setdefault("util_ratio", []).append(float(mask.mean()))

            else:  # source_only / focal / swin_b
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits = model(x)
                loss = focal(logits.float(), y) if focal is not None else F.cross_entropy(logits.float(), y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            run_loss += float(loss.detach())
            step_g += 1
        sched.step()
        extra = {k: round(float(np.mean(v)), 4) for k, v in extra_acc.items()}
        if flex is not None:
            extra.update({k: round(v, 4) for k, v in flex.status().items()})
        mv, mt, pt = record(epoch, "train" if args.method != "wb_dann" else "wb-stage1",
                            run_loss / max(steps, 1), extra, model)
        if args.method != "wb_dann" and mv["top1"] > best["val"]:
            best = {"val": mv["top1"], "epoch": epoch}
            save_probs("probs_bestval.npz", pt, epoch)
            save_model("model_bestval.pt", epoch)
        last_pt, last_epoch = pt, epoch
        save_resume(epoch)

    # ================================================= WB 第二阶段
    # resume.pt 只在第一阶段写入：第二阶段仅 10 个 epoch，中断时从第一阶段末的状态整段重跑
    if args.method == "wb_dann":
        if args.save_stage1:
            save_model("model_stage1.pt", epochs)
        s2 = hp["stage2"]
        for p in model.parameters():
            p.requires_grad_(False)
        layer = model.last_layer
        for p in layer.parameters():
            p.requires_grad_(True)
        pgd = MaxNormPGD(layer, thresh=s2["maxnorm_thresh"])
        cb = ClassBalancedSoftmaxLoss(src_train.class_counts, beta=s2["cb_beta"]).to(device)
        opt2 = build_optimizer(list(layer.parameters()), s2["optim"], s2["lr"], s2["weight_decay"])
        e2 = s2["epochs"] if not smoke else 1
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, e2, eta_min=0.0)
        for k in range(1, e2 + 1):
            model.train()     # 官方 train_model 在第二阶段同样处于 train 模式
            run_loss = 0.0
            for i, b in enumerate(train_loader):
                if i >= steps:
                    break
                x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
                opt2.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits = model(x)
                loss = cb(logits.float(), y)
                loss.backward()
                opt2.step()
                pgd.project()
                run_loss += float(loss.detach())
            sch2.step()
            ep = epochs + k
            mv, mt, pt = record(ep, "wb-stage2", run_loss / max(steps, 1),
                                {"maxnorm_limit": round(pgd.limit, 5)}, model)
            # WB 的产出是第二阶段后的模型，选型只在第二阶段的 epoch 中进行
            if mv["top1"] > best["val"]:
                best = {"val": mv["top1"], "epoch": ep}
                save_probs("probs_bestval.npz", pt, ep)
                save_model("model_bestval.pt", ep)
            last_pt, last_epoch = pt, ep

    save_probs("probs_final.npz", last_pt, last_epoch)
    save_model("model_final.pt", last_epoch)
    _finish(out, args, hp, history, best_epoch=best.get("epoch"), final_epoch=last_epoch,
            smoke=smoke, t0=t0, extra={"resumed_at_epochs": resumed_at} if resumed_at else None)
    clear_resume(out)


def _finish(out, args, hp, history, best_epoch, final_epoch, smoke, t0, extra=None):
    by = {h["epoch"]: h for h in history}
    summary = {
        "method": args.method, "seed": args.seed, "smoke": smoke,
        "usable_for_paper": not smoke,
        "selection": {
            "final": {"epoch": final_epoch, "tgt": by[final_epoch]["tgt"], "src_val": by[final_epoch]["src_val"]},
            "best_source_val": ({"epoch": best_epoch, "tgt": by[best_epoch]["tgt"],
                                 "src_val": by[best_epoch]["src_val"]} if best_epoch else None),
        },
        "hyperparameters": hp,
        "wall_clock_s": round(time.time() - t0, 1),
        **(extra or {}),
    }
    (out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[baseline] 完成 -> {out}  用时 {summary['wall_clock_s']} s")


if __name__ == "__main__":
    main()
