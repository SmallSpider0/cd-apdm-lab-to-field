"""CD-APDM Inference + Target-Domain Adaptation.

Implements §3.3 of the paper:
    1. Build student / teacher classifiers (teacher = EMA copy).
    2. (Optional) train CC-GANM for class-conditional style transfer of
       source images into target style.
    3. SS-PLAM target-encoder pretraining (rotation + SimCLR).
    4. Mean-Teacher pseudo-labeling pass with logit adjustment.
    5. **ACRM** confidence rectification → produces final calibrated
       predictions and the metric table.

The same script supports the ablations reported in Table 2 via CLI flags
(``--ablation w/o-ACRM`` etc.).
"""
from __future__ import annotations

import argparse
import os
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from .models.cc_gan import CCGANConfig, CCGANM
from .modules.acrm import ACRM, ACRMConfig
from .modules.ss_plam import (
    SSPLAMHelper,
    ema_update,
    logit_adjustment,
    nt_xent_loss,
    rotation_self_supervision,
)
from .utils.config import load_config
from .utils.metrics import evaluate_all
from .utils.seed import pick_device, set_seed


ABLATIONS = {"full", "w/o-ACRM", "w/o-CC-GANM", "w/o-SS-PLAM", "w/o-TFEM"}


# --------------------------------------------------------------------- helpers
def load_model(checkpoint_path: str, cfg: dict, tfem_dim: int, device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    # 温度设置取自 checkpoint 自身的配置：训练期消融 w/o-CIWLM 固定 T=1，
    # 若按命令行配置重建，会得到可学习温度、与训练时不同构
    ciwlm = ckpt.get("config", cfg)["ciwlm"]
    model = CDAPDMModel(
        num_classes=cfg["data"]["num_classes"],
        temporal_input_dim=tfem_dim,
        temporal_hidden=cfg["tfem"]["lstm_hidden"],
        mhsa_heads=cfg["mhsa"]["num_heads"],
        mhsa_dropout=cfg["mhsa"]["dropout"],
        init_T=ciwlm["init_temperature"],
        learnable_T=ciwlm["learnable_temperature"],
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def class_priors_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.where(counts <= 0, 1.0, counts)
    return torch.tensor(counts / counts.sum(), dtype=torch.float32)


# --------------------------------------------------------------------- adaptation
def cc_ganm_train(ccgan: CCGANM, src_loader, tgt_loader, device, iters: int = 300,
                  labeler=None):
    """Lightweight CycleGAN finetune — enough to expose the API, not a full GAN training.

    目标域一侧的类别条件取自 ``labeler``（源域模型的伪标签），**不使用目标域真实标签**。
    此前这里读取 ``tgt_b["label"]`` 作为 G_T2S 的条件，属目标域标签泄漏（R1-3），
    已改正；目标域 loader 经 StripLabels 包装，误用会当场抛 KeyError。
    """
    g_params = list(ccgan.G_S2T.parameters()) + list(ccgan.G_T2S.parameters())
    d_params = list(ccgan.D_T.parameters()) + list(ccgan.D_S.parameters())
    opt_g = torch.optim.Adam(g_params, lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d_params, lr=2e-4, betas=(0.5, 0.999))
    ccgan.train()
    pbar = tqdm(zip(src_loader, tgt_loader), total=iters, desc="CC-GANM")
    for step, (src_b, tgt_b) in enumerate(pbar):
        if step >= iters:
            break
        src = src_b["image"].to(device)
        tgt = tgt_b["image"].to(device)
        src_label = src_b["label"].to(device)
        with torch.no_grad():
            tgt_label = labeler(tgt, tgt_b["sequence"].to(device))

        # Generator update
        opt_g.zero_grad()
        g_loss, fake_t, fake_s = ccgan.generator_step(src, tgt, src_label, tgt_label)
        g_loss.backward()
        opt_g.step()

        # Discriminator update
        opt_d.zero_grad()
        d_loss = ccgan.discriminator_step(src, tgt, fake_t, fake_s)
        d_loss.backward()
        opt_d.step()
        pbar.set_postfix(g=f"{g_loss.item():.3f}", d=f"{d_loss.item():.3f}")


def ss_plam_pretrain(model, helper: SSPLAMHelper, loader, device, iters: int = 300,
                     temp: float = 0.1, rot_w: float = 0.5, con_w: float = 0.5):
    """Rotation + SimCLR head training on the target encoder."""
    helper.to(device)
    params = list(model.backbone.parameters()) + list(helper.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    pbar = tqdm(enumerate(loader), total=min(iters, len(loader)), desc="SS-PLAM pretrain")
    for step, batch in pbar:
        if step >= iters:
            break
        img = batch["image"].to(device)
        # rotation
        rot_img, rot_lbl = rotation_self_supervision(img)
        feat_rot = model.extract_image_feature(rot_img)
        loss_rot = F.cross_entropy(helper.rotation_head(feat_rot), rot_lbl)
        # SimCLR pair (two augmented views via flip + jitter realised in dataloader transforms)
        feat1 = model.extract_image_feature(img)
        feat2 = model.extract_image_feature(torch.flip(img, dims=(-1,)))
        z1 = helper.projection_head(feat1)
        z2 = helper.projection_head(feat2)
        loss_con = nt_xent_loss(z1, z2, temperature=temp)
        loss = rot_w * loss_rot + con_w * loss_con
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")


def mean_teacher_adaptation(
    student: CDAPDMModel,
    teacher: CDAPDMModel,
    tgt_loader: DataLoader,
    device,
    class_priors: torch.Tensor,
    tau_adj: float,
    teacher_tau: float,
    pseudo_conf_threshold: float,
    iters: int,
    ccgan: CCGANM | None = None,
    src_loader: DataLoader | None = None,
    selection: str = "fixed",
    cb_keep: float = 0.5,
    cb_refresh: int = 100,
    diversity_weight: float = 0.0,
    lr: float = 5e-5,
    acrm: "ACRM | None" = None,
):
    """Mean-Teacher 自训练；CC-GANM 提供类条件迁移样本作增广。

    每一步：
      * 教师（eval 模式）对真实目标域图像给出 logit 调整后的伪标签，保留置信度 ≥ 阈值者；
      * 若启用 CC-GANM：取一批**带标签的源域图像**，经 G_S2T 以其真实类别为条件
        迁移为目标域风格（论文 3.x："class-conditional migrations"，增广目标域稀有类多样性）；
      * 学生（train 模式）在「真实目标图 + 伪标签」与「迁移源图 + 源标签」拼成的一个批上
        做交叉熵（同批前向，BN 统计来自混合批），随后以 EMA 更新教师。

    2026-09-22 更正的三处实现缺陷（诊断见 results/exp-2-diag-s0.json）：
      1. 此前把 G_S2T 施加于**目标域**图像、并用其输出替换学生的输入 —— 生成器从未见过
         这种输入，且原真实目标图因此不再参与训练；现改为迁移源域图像并作为增广样本。
      2. 教师此前处于 train 模式生成伪标签：dropout 生效、BN 用批统计，伪标签额外带噪；现固定 eval。
      3. 生成器输出范围与真实图像不一致（见 models/cc_gan.py）。
    """
    """（续）伪标签筛选方式 ``selection``（2026-09-22 起可配置，默认与此前一致）：
      * ``fixed``          置信度 ≥ pseudo_conf_threshold（此前的唯一实现）
      * ``acrm``           置信度 ≥ ACRM 的逐类阈值 τ_c —— 论文规格："ACRM is applied during both
                           pseudo-label generation and final inference"，此前实现未照做
      * ``class_balanced`` 逐类保留教师置信度最高的 cb_keep 比例（CBST, Zou et al. 2018），
                           阈值每 cb_refresh 步在全部目标图（无标签）上重算，防止向少数类坍缩
    ``diversity_weight`` > 0 时加 SHOT（Liang et al. 2020）的多样性项：最大化批内平均预测的熵，
    直接惩罚预测集中到少数类。
    """
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    cb_thr = None

    @torch.no_grad()
    def class_thresholds():
        confs, preds = [], []
        for b in tgt_loader:
            lg = logit_adjustment(teacher(b["image"].to(device), b["sequence"].to(device)), class_priors, tau=tau_adj)
            c, pr = F.softmax(lg, dim=-1).max(dim=-1)
            confs.append(c); preds.append(pr)
        confs, preds = torch.cat(confs), torch.cat(preds)
        thr = torch.ones(class_priors.numel(), device=device)          # 没有样本被预测为该类 → 不取
        for k in preds.unique():
            ck = confs[preds == k]
            thr[k] = torch.quantile(ck.float(), 1.0 - cb_keep)
        return thr
    src_iter = None
    if ccgan is not None:
        if src_loader is None:
            raise ValueError("启用 CC-GANM 时须提供带标签的源域 loader")
        ccgan.eval()

        def _forever(loader):
            while True:
                for b in loader:
                    yield b
        src_iter = _forever(src_loader)
    pbar = tqdm(enumerate(tgt_loader), total=min(iters, len(tgt_loader)), desc="Mean-Teacher")
    for step, batch in pbar:
        if step >= iters:
            break
        img = batch["image"].to(device)
        seq = batch["sequence"].to(device)
        with torch.no_grad():
            teacher_logits = teacher(img, seq)
            adj = logit_adjustment(teacher_logits, class_priors, tau=tau_adj)
            probs = F.softmax(adj, dim=-1)
            conf, pseudo = probs.max(dim=-1)
            if selection == "acrm" and acrm is not None:
                keep = conf >= acrm.thresholds().to(device)[pseudo]
            elif selection == "class_balanced":
                if cb_thr is None or step % cb_refresh == 0:
                    cb_thr = class_thresholds()
                keep = conf >= cb_thr[pseudo]
            else:
                keep = conf >= pseudo_conf_threshold
        n_all_t = img.size(0) if diversity_weight > 0 else None
        if n_all_t is not None:      # 多样性项需要全部目标图的学生输出：全部目标图在前，交叉熵只取保留者
            xs, ys, ss = [img], [pseudo], [seq]
        else:
            xs, ys, ss = [img[keep]], [pseudo[keep]], [seq[keep]]
        if src_iter is not None:
            sb = next(src_iter)
            s_img, s_lbl = sb["image"].to(device), sb["label"].to(device)
            with torch.no_grad():
                xs.append(ccgan.G_S2T(s_img, s_lbl))
            ys.append(s_lbl)
            ss.append(sb["sequence"].to(device))
        x, y, sq = torch.cat(xs), torch.cat(ys), torch.cat(ss)
        if x.size(0) < 2:        # 无可用样本（BN 在训练模式下至少需要 2 个样本）
            continue
        if acrm is not None and selection == "acrm":   # ACRM：学生与教师在同一批目标图上的一致率与熵
            student.eval()
            with torch.no_grad():
                sp = F.softmax(logit_adjustment(student(img, seq), class_priors, tau=tau_adj), dim=-1)
            acrm.update_stats(pseudo_labels=sp.argmax(-1), teacher_labels=pseudo, probs=sp)
        student.train()
        out = student(x, sq)
        if n_all_t is not None:
            ce_mask = torch.cat([keep, torch.ones(x.size(0) - n_all_t, dtype=torch.bool, device=device)])
            loss = F.cross_entropy(out[ce_mask], y[ce_mask]) if ce_mask.any() else out.sum() * 0
            p_mean = F.softmax(out[:n_all_t], dim=-1).mean(0)
            loss = loss + diversity_weight * (p_mean * torch.log(p_mean + 1e-8)).sum()   # = −H(平均预测)
        else:
            loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(student, teacher, tau=teacher_tau)
        pbar.set_postfix(loss=f"{loss.item():.3f}", kept=int(keep.sum().item()))


@torch.no_grad()
def adabn(model: nn.Module, loader: DataLoader, device) -> None:
    """AdaBN（Li et al. 2017）：用目标域图像重估全部 BatchNorm 的运行统计量，其余参数不变。
    不使用标签，也不使用伪标签。统计量取全部目标图上的累计平均（momentum=None）。"""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats(); m.momentum = None
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    for b in loader:
        model(b["image"].to(device), b["sequence"].to(device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()


# --------------------------------------------------------------------- predict
@torch.no_grad()
def predict_with_acrm(model, loader, device, acrm: ACRM | None, class_priors: torch.Tensor, tau_adj: float):
    model.eval()
    all_probs, all_lbl, all_ids = [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        seq = batch["sequence"].to(device)
        logits = model(img, seq)
        logits = logit_adjustment(logits, class_priors, tau=tau_adj)
        probs = F.softmax(logits, dim=-1)
        if acrm is not None:
            probs = acrm.rectify(probs)
        all_probs.append(probs.cpu().numpy())
        all_lbl.append(batch["label"].numpy())
        all_ids.extend(batch["image_id"])
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_lbl, axis=0)
    preds = probs.argmax(axis=-1)
    predict_with_acrm.last_image_ids = all_ids   # 供保存逐图概率使用，不改变返回签名
    return preds, labels, probs


# --------------------------------------------------------------------- main
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖配置中的 seed；多 seed 实验必须经由此参数，而不是改配置文件")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    # 默认值一律取自配置文件（default=None），命令行仅用于临时覆盖。
    # 此前这三项硬编码默认值，而配置里的同名键从不被读取 ——
    # 配置文件看似权威实则是摆设，且它会被原样存进 checkpoint 作为
    # 「本次运行如何配置」的记录，造成记录与实际不符。
    ap.add_argument("--ssplam_iters", type=int, default=None)
    ap.add_argument("--ccgan_iters", type=int, default=None)
    ap.add_argument("--mt_iters", type=int, default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="覆盖配置项（可重复），仅供开发集实验；正式运行的配置以配置文件为准")
    ap.add_argument("--dump_stages", action="store_true",
                    help="诊断：在自适应各阶段后各保存一份逐图概率（stage_*.npz），用于在开发集上定位性能在哪一步下降")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set or []:           # 开发实验用：--set ss_plam.pseudo_selection=class_balanced
        key, val = kv.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
        print(f"[CD-APDM] 覆盖配置 {key} = {node[last]!r}")
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    guard_local_resources(cfg, device)
    # 断点续跑：eval_<tag>.json 最后写入，存在即表示本次推理已完成
    done = Path(os.environ.get("RUN_DIR") or cfg["train"]["checkpoint_dir"]) / f"eval_{args.ablation.replace('/', '-')}.json"
    if done.exists():
        print(f"[CD-APDM] {done} 已存在，推理已完成，跳过")
        return

    # 命令行 > 配置文件 > 内置兜底
    ccgan_iters = args.ccgan_iters if args.ccgan_iters is not None else cfg["cc_ganm"].get("num_iters", 200)
    ssplam_iters = args.ssplam_iters if args.ssplam_iters is not None else cfg["ss_plam"].get("iters", 300)
    mt_iters = args.mt_iters if args.mt_iters is not None else cfg["ss_plam"].get("mean_teacher_iters", 300)
    print(f"[CD-APDM] iters: ccgan={ccgan_iters} ssplam={ssplam_iters} mt={mt_iters}")
    num_classes = cfg["data"]["num_classes"]

    # -------------------------------------------- datasets
    # 划分来自 dataset/splits/*.csv，与训练侧同源；不再运行时随机划分。
    # 时序模态默认关闭，理由见 dataset/AgriNet/ALIGNMENT.md。
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
    source_ds, _src_val, target_ds, class_index = build_split_datasets(cfg, temporal=tfem)
    num_classes = class_index.num_classes
    temporal_dim = (tfem.feature_dim if tfem is not None
                    else len(cfg["tfem"]["feature_columns"]) + len(cfg["tfem"]["event_columns"]))

    bs, nw = cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    src_train_loader = DataLoader(source_ds, batch_size=bs, shuffle=True, num_workers=nw)
    # 目标域自适应与评估使用同一全量集合（transductive，UDA 常规做法）。
    # 是否改为留出式（inductive）属 fix-experiment-protocol 的待决项。
    # 参与自适应的目标域数据不带标签；标签只在 tgt_test_loader 的评测路径上出现
    tgt_train_loader = DataLoader(StripLabels(target_ds), batch_size=bs, shuffle=True, num_workers=nw)
    tgt_test_loader = DataLoader(target_ds, batch_size=bs, shuffle=False, num_workers=nw)

    # -------------------------------------------- model + teacher
    student, ckpt = load_model(args.checkpoint, cfg, temporal_dim, device)
    if tfem is None:
        student.use_temporal = False
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    if args.ablation == "w/o-TFEM":
        student.use_temporal = False
        teacher.use_temporal = False
    elif args.ablation != "full" and tfem is None and args.ablation == "w-TFEM":
        raise ValueError("请求 TFEM 消融但 tfem.enabled 为 false")

    class_counts = ckpt["class_counts"]
    class_priors = class_priors_from_counts(class_counts).to(device)
    tail_classes = ckpt["tail_classes"]

    # -------------------------------------------- CC-GANM (optional)
    ccgan = None
    adapt_mode = cfg["ss_plam"].get("adapt_mode", "mean_teacher")   # mean_teacher | adabn | adabn_mt | none
    uses_mt = adapt_mode in ("mean_teacher", "adabn_mt")
    if adapt_mode == "adabn_mt":      # 先以目标域统计量重估 BN，再在此基础上自训练
        adabn(student, tgt_train_loader, device)
        teacher.load_state_dict(student.state_dict())
    if cfg["cc_ganm"]["enabled"] and args.ablation != "w/o-CC-GANM" and uses_mt:
        ccgan_cfg = CCGANConfig(
            hidden_dim=cfg["cc_ganm"]["hidden_dim"],
            cycle_weight=cfg["cc_ganm"]["cycle_weight"],
            adv_weight=cfg["cc_ganm"]["adv_weight"],
            num_classes=num_classes,
        )
        ccgan = CCGANM(ccgan_cfg).to(device)
        def source_model_pseudo_labels(img, seq):
            was_training = student.training   # 不改变后续 SS-PLAM / Mean-Teacher 看到的模式
            student.eval()
            logits = logit_adjustment(student(img, seq), class_priors, cfg["logit_adjust"]["tau"])
            student.train(was_training)
            return logits.argmax(-1)
        cc_ganm_train(ccgan, src_train_loader, tgt_train_loader, device, iters=ccgan_iters,
                      labeler=source_model_pseudo_labels)

    def dump(name, model):
        if not args.dump_stages:
            return
        was_training = model.training     # 快照不得改变后续阶段看到的训练/评估模式
        _, lbl, pr = predict_with_acrm(model, tgt_test_loader, device, None, class_priors, cfg["logit_adjust"]["tau"])
        _, _, pr0 = predict_with_acrm(model, tgt_test_loader, device, None, class_priors, 0.0)
        d = Path(os.environ.get("RUN_DIR") or cfg["train"]["checkpoint_dir"])
        d.mkdir(parents=True, exist_ok=True)
        ids = np.array(predict_with_acrm.last_image_ids)
        np.savez_compressed(d / f"stage_{name}.npz", probs=pr.astype(np.float16), labels=lbl, image_ids=ids)
        np.savez_compressed(d / f"stage_{name}_noLA.npz", probs=pr0.astype(np.float16), labels=lbl, image_ids=ids)
        model.train(was_training)

    dump("0_source", student)          # 此时学生仍是未经任何自适应的源模型（CC-GANM 不改动分类网络）

    # -------------------------------------------- SS-PLAM pretrain (optional)
    if cfg["ss_plam"]["enabled"] and args.ablation != "w/o-SS-PLAM" and uses_mt and cfg["ss_plam"].get("pretrain", True):
        helper = SSPLAMHelper(feat_dim=student.backbone.feature_dim).to(device)
        ss_plam_pretrain(student, helper, tgt_train_loader, device, iters=ssplam_iters,
                         temp=cfg["ss_plam"]["contrastive_temp"],
                         rot_w=cfg["ss_plam"]["rotation_weight"],
                         con_w=cfg["ss_plam"]["contrastive_weight"])
        teacher.load_state_dict(student.state_dict())

    if cfg["ss_plam"]["enabled"] and args.ablation != "w/o-SS-PLAM" and uses_mt and cfg["ss_plam"].get("pretrain", True):
        dump("1_after_ssplam", student)

    sp_cfg = cfg["ss_plam"]
    selection = sp_cfg.get("pseudo_selection", "fixed")
    acrm = None
    if args.ablation != "w/o-ACRM":
        acrm = ACRM(num_classes=num_classes, class_counts=class_counts, cfg=ACRMConfig(
            eta=cfg["acrm"]["eta"], tau_base=cfg["acrm"]["tau_base"],
            lambda_acc=cfg["acrm"]["lambda_acc"], lambda_ent=cfg["acrm"]["lambda_ent"],
            gamma_boost=cfg["acrm"]["gamma_boost"], tail_threshold=cfg["mdiwm"]["tail_threshold"],
        ), device=device)
    elif selection == "acrm":
        selection = "fixed"      # 消融 w/o-ACRM 时无 ACRM 可用，退回固定阈值

    # -------------------------------------------- Mean-Teacher adaptation
    if adapt_mode == "adabn":
        adabn(student, tgt_train_loader, device)
        teacher.load_state_dict(student.state_dict())
    if uses_mt:
        mean_teacher_adaptation(
            student=student,
            teacher=teacher,
            tgt_loader=tgt_train_loader,
            device=device,
            class_priors=class_priors,
            tau_adj=cfg["logit_adjust"]["tau"],
            teacher_tau=cfg["ss_plam"]["teacher_tau"],
            pseudo_conf_threshold=cfg["ss_plam"]["pseudo_conf_threshold"],
            iters=mt_iters,
            ccgan=ccgan if args.ablation != "w/o-CC-GANM" else None,
            src_loader=src_train_loader,
            selection=selection,
            cb_keep=sp_cfg.get("cb_keep", 0.5),
            cb_refresh=sp_cfg.get("cb_refresh", 100),
            diversity_weight=sp_cfg.get("diversity_weight", 0.0),
            lr=sp_cfg.get("mean_teacher_lr", 5e-5),
            acrm=acrm if selection == "acrm" else None,
        )

    dump("2_after_mt_student", student)
    dump("2_after_mt_teacher", teacher)

    # -------------------------------------------- ACRM warm-up
    # 伪标签筛选用 ACRM 时，其统计量已在自适应过程中更新，不再另做 warm-up
    if acrm is not None and selection != "acrm":
        # populate EMA buffers using a short pass over target train data
        student.eval(); teacher.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(tgt_train_loader, desc="ACRM warm-up")):
                if step >= 50:
                    break
                img = batch["image"].to(device); seq = batch["sequence"].to(device)
                s_logits = student(img, seq)
                t_logits = teacher(img, seq)
                s_logits = logit_adjustment(s_logits, class_priors, cfg["logit_adjust"]["tau"])
                t_logits = logit_adjustment(t_logits, class_priors, cfg["logit_adjust"]["tau"])
                probs = F.softmax(s_logits, dim=-1)
                acrm.update_stats(
                    pseudo_labels=probs.argmax(-1),
                    teacher_labels=t_logits.argmax(-1),
                    probs=probs,
                )

    # -------------------------------------------- evaluate on target test
    final_model = teacher if sp_cfg.get("predict_with", "student") == "teacher" else student
    preds, labels, probs = predict_with_acrm(
        final_model, tgt_test_loader, device, acrm,
        class_priors=class_priors,
        tau_adj=cfg["logit_adjust"]["tau"],
    )
    metrics = evaluate_all(preds, labels, probs, num_classes, tail_classes)
    metrics["ablation"] = args.ablation
    if acrm is not None:
        # EXP-10：ACRM 在自适应过程中跟踪的逐类师生一致率（命题 1 中的量），与真实准确率离线对照
        metrics["acrm_agree_c"] = [float(v) for v in acrm.agree_c.detach().cpu()]

    # 产物写入 $RUN_DIR（服务器由 scripts/remote.sh 设置），否则写入 checkpoint_dir。
    # 与基线一致地保存逐图目标域概率：评测口径（是否排除标签冲突图像、
    # 不可评测类别等）属待决项，定下后可离线重算，无需重跑。
    out_dir = Path(os.environ.get("RUN_DIR") or cfg["train"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.ablation.replace("/", "-")
    np.savez_compressed(out_dir / f"probs_{tag}.npz", probs=probs.astype(np.float16), labels=labels,
                        image_ids=np.array(predict_with_acrm.last_image_ids))
    out_path = out_dir / f"eval_{tag}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print("\n=== Target-domain results ===")
    print(json.dumps(metrics, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
