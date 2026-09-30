"""优化器与学习率调度的唯一构造入口。

CD-APDM（train.py）与全部基线（baselines/run.py）都从这里构造，
保证 Table 1 各行的优化设置出自同一段代码，而不是各自一份可能漂移的拷贝。
"""
from __future__ import annotations

import torch


def backbone_param_groups(backbone, rest, lr: float, backbone_lr_mult: float = 1.0):
    """预训练骨干与新初始化层分组：骨干 lr = lr × backbone_lr_mult。

    2026-09-21 作者决策（fix-experiment-protocol 决策记录 4）：原设定对 ImageNet 预训练的
    ResNet-50 全量使用 Adam lr 1e-3，诊断显示源模型对实验室图像过拟合、跨域迁移能力被破坏
    （results/exp-2-diag-s0.json）。改用领域通行的判别式微调：骨干取头部学习率的 0.1 倍
    （Transfer-Learning-Library 中 DANN 等方法的默认做法）。该倍率取自文献，未在目标域上搜索。
    调度器按各组的初始 lr 等比缩放，两组的 warmup/cosine 形状相同。
    """
    if backbone_lr_mult == 1.0:
        # 不分组：参数组结构与顺序与引入本函数前完全相同（Swin-B 等不受影响，已有 resume.pt 可续跑）
        return [{"params": [p for p in rest if p.requires_grad], "lr": lr, "name": "all"}]
    bb = [p for p in backbone.parameters() if p.requires_grad]
    ids = {id(p) for p in bb}
    other = [p for p in rest if id(p) not in ids and p.requires_grad]
    groups = [{"params": other, "lr": lr, "name": "head"}]
    if bb:
        groups.append({"params": bb, "lr": lr * backbone_lr_mult, "name": "backbone"})
    return groups


def build_optimizer(params, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "sgd_plain":   # WB 官方：momentum 0.9，无 nesterov
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"未知的 optimizer: {name}")


def build_scheduler(optim, name: str, epochs: int, warmup: int = 0):
    """按 epoch 步进的调度器。warmup>0 时先线性升温再接主调度。"""
    name = name.lower()
    if name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs - warmup, 1))
    elif name == "step":
        main = torch.optim.lr_scheduler.StepLR(optim, step_size=max(epochs // 3, 1), gamma=0.1)
    elif name == "none":
        main = torch.optim.lr_scheduler.ConstantLR(optim, factor=1.0, total_iters=1)
    else:
        raise ValueError(f"未知的 scheduler: {name}")
    if warmup > 0:
        return torch.optim.lr_scheduler.SequentialLR(
            optim,
            [torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=warmup), main],
            milestones=[warmup],
        )
    return main
