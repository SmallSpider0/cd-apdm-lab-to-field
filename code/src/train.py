"""CD-APDM Training Phase.

Implements the training loop sketched in §3.2 of the paper:
    1. Pre-fit the ResNet-18 domain classifier (MDIWM).
    2. For each batch:
       a. compute MDIWM weights (d_i, u_i),
       b. fuse image features with TFEM temporal features via MHSA,
       c. optimise the CIWLM-weighted classifier loss.
    3. Save checkpoints + per-epoch evaluation metrics.
"""
from __future__ import annotations

import argparse
import os
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
try:
    from tqdm import tqdm
except ImportError:  # 服务器环境无 tqdm 且无外网；进度条不是功能，缺了就退化
    class tqdm:  # noqa: N801  最小替身：可迭代、可调用 set_postfix/update/close
        def __init__(self, iterable=None, total=None, desc=None, **kw):
            self.iterable = iterable if iterable is not None else []
            self.total, self.desc, self.n = total, desc, 0

        def __iter__(self):
            for x in self.iterable:
                self.n += 1
                yield x

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_postfix(self, *a, **kw):
            pass

        def set_description(self, *a, **kw):
            pass

        def update(self, n=1):
            self.n += n

        def close(self):
            pass


from .datasets.agrinet import AgriNetTFEM, AgriNetTFEMConfig
from .datasets.splits import StripLabels, build_datasets as build_split_datasets
from .models.backbone import CDAPDMModel
from .modules.ciwlm import CIWLMLoss
from .modules.mdiwm import HeadTailWeightedSampler, MDIWM, MDIWMConfig, compute_instance_weights
from .utils.config import load_config
from .utils.metrics import evaluate_all
from .utils.optim import backbone_param_groups, build_optimizer, build_scheduler
from .utils.resume import atomic_save, clear_resume, load_resume, rng_state, set_rng_state
from .utils.seed import pick_device, set_seed


# --------------------------------------------------------------------- helpers
def temporal_dim(cfg: dict) -> int:
    return len(cfg["tfem"]["feature_columns"]) + len(cfg["tfem"]["event_columns"])


def make_datasets(cfg: dict):
    """按划分文件构建数据集。

    时序模态默认关闭 —— PlantVillage 与 PlantDoc 均无可信的采集时间戳与
    地块标识，无法诚实配对时序数据（R2-2；见 dataset/AgriNet/ALIGNMENT.md）。
    仅当 ``tfem.enabled`` 为真时才构造 AgriNetTFEM，且其 strict 模式会在
    对齐失败时抛错，不再静默退化。
    """
    tfem = None
    if cfg.get("tfem", {}).get("enabled", False):
        tfem = AgriNetTFEM(AgriNetTFEMConfig(
            window_csv=cfg["data"]["agrinet_window_csv"],
            align_csv=cfg["data"]["agrinet_align_csv"],
            feature_columns=cfg["tfem"]["feature_columns"],
            event_columns=cfg["tfem"]["event_columns"],
            window_days=cfg["tfem"]["window_days"],
            forward_fill_max_gap=cfg["tfem"]["forward_fill_max_gap"],
            strict=cfg["tfem"].get("strict", True),
        ))
    src_train, src_val, target, class_index = build_split_datasets(cfg, temporal=tfem)
    return tfem, src_train, src_val, target, class_index


# --------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device, num_classes, tail_classes):
    model.eval()
    preds, labels, all_probs = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        seq = batch["sequence"].to(device, non_blocking=True)
        logits = model(img, seq)
        probs = F.softmax(logits, dim=-1)
        preds.append(probs.argmax(dim=-1).cpu().numpy())
        labels.append(batch["label"].numpy())
        all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    probs = np.concatenate(all_probs)
    return evaluate_all(preds, labels, probs, num_classes, tail_classes), preds, labels, probs


def guard_local_resources(cfg: dict, device) -> None:
    """本机（MPS / CPU）运行时的资源护栏。

    M4 / 16 GB 是统一内存，CPU 与 GPU 共享。完整流水线会同时驻留
    ResNet-50 学生 + 教师深拷贝 + CC-GANM 生成器与判别器，极易吃满内存
    并拖垮整机。此处在启动前给出明确提示，而不是等到 OOM。
    """
    if str(device) not in ("mps", "cpu"):
        return
    import shutil
    bs = cfg["train"]["batch_size"]
    heavy = []
    if bs > 16:
        heavy.append(f"batch_size={bs}（本机建议 <=16）")
    if cfg.get("cc_ganm", {}).get("enabled") and cfg["cc_ganm"].get("num_iters", 0) > 50:
        heavy.append(f"CC-GANM {cfg['cc_ganm']['num_iters']} 轮（本机建议 <=50 或关闭）")
    if cfg["train"]["epochs"] > 2:
        heavy.append(f"epochs={cfg['train']['epochs']}（本机仅作接线验证，建议 1）")
    if heavy:
        print("=" * 68)
        print(f"[CD-APDM] ⚠ 在 {device} 上以较重的设置运行：")
        for h in heavy:
            print(f"    · {h}")
        print("    本机仅用于小规模验证；完整实验请在服务器执行（见 WORKSPACE.md）。")
        print("    轻量冒烟配置：configs/local_smoke.yaml")
        print("=" * 68)


# --------------------------------------------------------------------- main
TRAIN_ABLATIONS = ["full", "w/o-MDIWM", "w/o-CIWLM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖配置中的 seed；多 seed 实验必须经由此参数，而不是改配置文件")
    ap.add_argument("--epochs", type=int, default=None, help="override config")
    ap.add_argument("--max_train_batches", type=int, default=None, help="for quick smoke tests")
    ap.add_argument("--ablation", default="full", choices=TRAIN_ABLATIONS,
                    help="训练期消融（Table 2）：w/o-MDIWM 去掉域实例加权与头尾采样；"
                         "w/o-CIWLM 以普通交叉熵替代 CIWLM（含其可学习温度）。"
                         "推理期消融见 inference.py --ablation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.ablation == "w/o-CIWLM":
        # 温度是 CIWLM 的组成部分（配置位于 ciwlm 段）：去掉 CIWLM 即固定 T=1。
        # 改写 cfg 而非另设变量：cfg 随 checkpoint 保存，inference.py 据此重建同构模型
        cfg["ciwlm"]["init_temperature"] = 1.0
        cfg["ciwlm"]["learnable_temperature"] = False
    cfg["train_ablation"] = args.ablation
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    print(f"[CD-APDM] device = {device}")
    guard_local_resources(cfg, device)

    # ---- 断点续跑：已完成则跳过；有 resume.pt 则从下一个 epoch 继续
    run_dir = Path(os.environ.get("RUN_DIR") or cfg["train"]["checkpoint_dir"])
    if (run_dir / "selection.json").exists() and (run_dir / "cd_apdm_final.pt").exists():
        print(f"[CD-APDM] {run_dir} 训练已完成（selection.json 存在），跳过")
        return
    n_epochs = args.epochs or cfg["train"]["epochs"]
    resume = load_resume(run_dir, seed=cfg["seed"], ablation=args.ablation, epochs=n_epochs)

    # ----------------------------------------------- data
    # 划分来自 dataset/splits/*.csv（固定种子生成、随仓库公开），
    # 不再于运行时随机划分 —— 随机划分不可复现，且每次运行都不同。
    tfem, source_ds, source_val_ds, target_ds, class_index = make_datasets(cfg)
    use_temporal = tfem is not None
    print(f"[CD-APDM] source train={len(source_ds)}  val={len(source_val_ds)}  "
          f"target={len(target_ds)}  temporal={use_temporal}")

    num_classes = class_index.num_classes
    if num_classes != cfg["data"]["num_classes"]:
        raise ValueError(
            f"划分文件为 {num_classes} 类，配置写的是 {cfg['data']['num_classes']} 类"
        )
    bs, nw = cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    head_th = cfg["mdiwm"]["head_threshold"]
    tail_th = cfg["mdiwm"]["tail_threshold"]
    # 尾类身份由源域构造分布的 OLTR 分桶确定（Tail = Medium ∪ Few），
    # 不按运行时计数重判 —— 训练集计数已因切出验证集而偏离构造分布。
    tail_classes = class_index.tail_classes
    head_classes = class_index.bucket_classes("many")
    print(f"[CD-APDM] buckets: many={len(head_classes)} tail={len(tail_classes)} "
          f"-> tail_classes={tail_classes}")

    # ----------------------------------------------- model
    model = CDAPDMModel(
        num_classes=num_classes,
        temporal_input_dim=(tfem.feature_dim if use_temporal else temporal_dim(cfg)),
        temporal_hidden=cfg["tfem"]["lstm_hidden"],
        temporal_layers=cfg["tfem"].get("lstm_layers", 1),
        mhsa_heads=cfg["mhsa"]["num_heads"],
        mhsa_dropout=cfg["mhsa"]["dropout"],
        init_T=cfg["ciwlm"]["init_temperature"],
        learnable_T=cfg["ciwlm"]["learnable_temperature"],
        pretrained_backbone=True,
        use_temporal=use_temporal,
    ).to(device)

    use_mdiwm = args.ablation != "w/o-MDIWM"
    if use_mdiwm:
        # ----------------------------------------------- MDIWM (domain classifier)
        mdiwm_cfg = MDIWMConfig(
            alpha=cfg["mdiwm"]["alpha"],
            beta=cfg["mdiwm"]["beta"],
            mc_dropout_p=cfg["mdiwm"]["mc_dropout_p"],
            mc_passes=cfg["mdiwm"]["mc_passes"],
            head_threshold=head_th,
            tail_threshold=tail_th,
            min_tail_ratio=cfg["mdiwm"]["min_tail_ratio"],
        )
        mdiwm = MDIWM(mdiwm_cfg, device=device)

        if resume is not None:
            # 续跑：复用首次运行算出的实例权重（域分类器训练与 MC dropout 均有随机性，重算会不一致）
            weights_train = resume["weights_train"]
            print("[CD-APDM] 续跑：复用保存的 MDIWM 实例权重")
        else:
            print("[CD-APDM] Fitting domain classifier (MDIWM)…")
            # 域分类器只需少量样本即可拟合；取固定前缀保证可复现
            n_dc = cfg["mdiwm"].get("domain_classifier_samples", 1024)
            src_loader_dc = DataLoader(Subset(source_ds, range(min(n_dc, len(source_ds)))),
                                       batch_size=bs, shuffle=True, num_workers=nw)
            tgt_loader_dc = DataLoader(StripLabels(Subset(target_ds, range(min(n_dc, len(target_ds))))),
                                       batch_size=bs, shuffle=True, num_workers=nw)
            mdiwm.fit_domain_classifier(
                src_loader_dc, tgt_loader_dc,
                epochs=cfg["mdiwm"]["domain_classifier_pretrain_epochs"],
            )

            # ----------------------------------------------- pre-compute MDIWM weights for source train
            print("[CD-APDM] Pre-computing MDIWM instance weights…")
            src_train_eval_loader = DataLoader(source_ds, batch_size=bs, shuffle=False, num_workers=nw)
            d_scores, u_scores = [], []
            with torch.no_grad():
                for batch in tqdm(src_train_eval_loader, desc="MDIWM scoring"):
                    img = batch["image"].to(device)
                    d = mdiwm.domain_scores(img).cpu().numpy()
                    u = mdiwm.uncertainties(model, img).cpu().numpy()
                    d_scores.append(d)
                    u_scores.append(u)
            d_scores = np.concatenate(d_scores)
            u_scores = np.concatenate(u_scores)
            weights_train = compute_instance_weights(d_scores, u_scores, mdiwm_cfg.alpha, mdiwm_cfg.beta)

        # source_ds 现已是 train 划分本身，权重与索引一一对应，无需再映射
        train_labels = source_ds.labels
        sampler = HeadTailWeightedSampler(
            labels=train_labels,
            class_counts=source_ds.class_counts,
            weights=weights_train,
            batch_size=cfg["train"]["batch_size"],
            num_samples=len(train_labels),
            head_threshold=head_th,
            tail_threshold=tail_th,
            head_classes=head_classes,
            tail_classes=tail_classes,
            min_tail_ratio=mdiwm_cfg.min_tail_ratio,
            seed=cfg["seed"],
        )
    else:
        # w/o MDIWM：实例权重全为 1，普通随机打乱（无头尾重采样）
        print("[CD-APDM] ablation w/o-MDIWM：跳过域分类器与实例加权，使用均匀随机采样")
        weights_train = np.ones(len(source_ds), dtype=np.float32)
        sampler = None

    # ----------------------------------------------- loaders
    train_loader = DataLoader(source_ds, batch_size=bs, sampler=sampler, shuffle=sampler is None,
                              num_workers=nw, drop_last=True, pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]) if sampler is None else None)
    # 源域验证集 —— 供不使用目标域标签的选型协议使用
    val_loader = DataLoader(source_val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)
    # 目标域全量评估（transductive）。目标域的划分用法（transductive / inductive）
    # 属 fix-experiment-protocol 的待决项，此处不做二次切分。
    target_loader = DataLoader(target_ds, batch_size=bs, shuffle=False,
                               num_workers=nw, pin_memory=True)

    # ----------------------------------------------- loss + optim
    if args.ablation == "w/o-CIWLM":
        # 普通交叉熵：无类别权重、无 focal 项；MDIWM 的实例权重照常生效
        def criterion(logits, targets, instance_weights=None):
            loss = F.cross_entropy(logits, targets, reduction="none")
            if instance_weights is not None:
                loss = loss * instance_weights
            return loss.mean()
    else:
        criterion = CIWLMLoss(
            class_counts=source_ds.class_counts,
            beta_e=cfg["ciwlm"]["beta_e"],
            focal_gamma=cfg["ciwlm"]["focal_gamma"],
        ).to(device)

    epochs = args.epochs or cfg["train"]["epochs"]
    optim = build_optimizer(
        backbone_param_groups(model.backbone, model.parameters(), cfg["train"]["lr"],
                              cfg["train"].get("backbone_lr_mult", 1.0)),
        cfg["train"].get("optimizer", "adam"), cfg["train"]["lr"], cfg["train"]["weight_decay"])
    print("[CD-APDM] lr: " + ", ".join(f"{g['name']}={g['lr']:g}" for g in optim.param_groups))
    sched = build_scheduler(optim, cfg["train"].get("scheduler", "cosine"), epochs,
                            int(cfg["train"].get("warmup_epochs", 0)))

    amp_enabled = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 服务器上由 scripts/remote.sh 设置 RUN_DIR，产物随 run 目录一并取回
    ckpt_dir = Path(os.environ.get("RUN_DIR") or cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_val = {"src_top1": -1.0, "epoch": None}

    def save_ckpt(path: Path, extra: Dict | None = None):
        atomic_save({
            "model": model.state_dict(),
            "tail_classes": tail_classes,
            "class_counts": source_ds.class_counts,
            "class_names": class_index.names,
            "class_buckets": class_index.buckets,
            "config": cfg,
            "history": history,
            **(extra or {}),
        }, path)

    # ----------------------------------------------- 续跑状态
    start_epoch, resumed_at = 0, []
    if resume is not None:
        model.load_state_dict(resume["model"])
        optim.load_state_dict(resume["optim"])
        sched.load_state_dict(resume["sched"])
        scaler.load_state_dict(resume["scaler"])
        if sampler is not None:
            sampler.rng.bit_generator.state = resume["sampler_rng"]
        else:
            train_loader.generator.set_state(resume["loader_gen"])
        set_rng_state(resume["rng"])
        history, best_val = resume["history"], resume["best_val"]
        start_epoch = resume["epoch"]
        resumed_at = resume["resumed_at"] + [start_epoch + 1]
        print(f"[CD-APDM] 从 epoch {start_epoch + 1} 续跑（resume.pt）")

    def save_resume(done_epochs):
        atomic_save({"seed": cfg["seed"], "ablation": args.ablation, "epochs": epochs, "epoch": done_epochs,
                     "model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "weights_train": weights_train,
                     "sampler_rng": sampler.rng.bit_generator.state if sampler is not None else None,
                     "loader_gen": train_loader.generator.get_state() if sampler is None else None,
                     "rng": rng_state(), "history": history, "best_val": best_val,
                     "resumed_at": resumed_at}, ckpt_dir / "resume.pt")

    # ----------------------------------------------- training loop
    weights_train_t = torch.from_numpy(weights_train).float()
    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoch {epoch+1}/{epochs}")
        for step, batch in pbar:
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            img = batch["image"].to(device, non_blocking=True)
            seq = batch["sequence"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            sub_idx = batch["index"]
            inst_w = weights_train_t[sub_idx].to(device)

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(img, seq)
                loss = criterion(logits, lbl, instance_weights=inst_w)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item() * img.size(0)
            n_seen += img.size(0)
            if step % cfg["train"]["log_every"] == 0:
                pbar.set_postfix(loss=f"{running/max(n_seen,1):.4f}")
        sched.step()

        # ---- 源域验证集（选型用，不触碰目标域标签）+ 目标域全量（报告用）
        src_metrics, *_ = evaluate(model, val_loader, device, num_classes, tail_classes)
        tgt_metrics, *_ = evaluate(model, target_loader, device, num_classes, tail_classes)
        gap = src_metrics["top1"] - tgt_metrics["top1"]
        epoch_summary = {
            "epoch": epoch + 1,
            "loss": running / max(n_seen, 1),
            "src_top1": src_metrics["top1"],
            "tgt_top1": tgt_metrics["top1"],
            "tgt_tail_recall": tgt_metrics["tail_recall"],
            "tgt_macro_f1": tgt_metrics["macro_f1"],
            "tgt_ece": tgt_metrics["ece"],
            "cross_domain_gap": gap,
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, indent=2))
        # 选型只看源域验证集（R1-3）：目标域指标仅记录、不参与任何决策
        if src_metrics["top1"] > best_val["src_top1"]:
            best_val = {"src_top1": src_metrics["top1"], "epoch": epoch + 1}
            save_ckpt(ckpt_dir / "cd_apdm_bestval.pt", {"selected_by": "source_val_top1", **best_val})
        save_resume(epoch + 1)

    # final checkpoint（"取最后一个 epoch" 规则，附录用）
    final_ckpt = ckpt_dir / "cd_apdm_final.pt"
    save_ckpt(final_ckpt, {"selected_by": "last_epoch", "epoch": len(history)})
    (ckpt_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    (ckpt_dir / "selection.json").write_text(json.dumps(
        {"best_source_val": best_val, "last_epoch": len(history), "train_ablation": args.ablation,
         **({"resumed_at_epochs": resumed_at} if resumed_at else {})}, indent=2))
    clear_resume(ckpt_dir)
    print(f"[CD-APDM] training complete -> {final_ckpt}")


if __name__ == "__main__":
    main()
