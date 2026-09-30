"""基线关键组件与官方实现的数值等价性测试。

"能跑通"不能说明实现正确：CB 损失、MaxNorm 投影、FlexMatch 阈值这类组件
写错了照样能跑、照样产出看似合理的数字。这里逐一与 tests/official/ 下
未经修改的官方代码比对。

    cd code && python -m pytest tests/test_baselines.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "official"))
sys.path.insert(0, str(HERE.parent))

from src.baselines.losses import (ClassBalancedSoftmaxLoss, FocalLoss,  # noqa: E402
                                  MaxNormPGD, dann_lambda)
from src.baselines.net import BaselineNet, grad_reverse  # noqa: E402
from src.baselines.run import StripLabels  # noqa: E402
from src.baselines.semi import FlexMatchThreshold  # noqa: E402

import usb_flexmatch_utils as usb  # noqa: E402
import wb_class_balanced_loss as wb_cb  # noqa: E402
import wb_regularizers as wb_reg  # noqa: E402

COUNTS = [5357, 3000, 1200, 400, 107, 85, 43, 21, 17, 11]


# ------------------------------------------------------------------ WB
def test_cb_loss_matches_official():
    torch.manual_seed(0)
    logits = torch.randn(32, len(COUNTS))
    y = torch.randint(0, len(COUNTS), (32,))
    ours = ClassBalancedSoftmaxLoss(COUNTS, beta=0.9999)(logits, y)
    ref = wb_cb.CB_loss(y, logits, COUNTS, len(COUNTS), "softmax", 0.9999, 2.0, "cpu")
    assert torch.allclose(ours, ref.float(), atol=1e-6), (ours, ref)


def test_maxnorm_matches_official():
    torch.manual_seed(0)
    layer = nn.Linear(64, len(COUNTS))
    with torch.no_grad():                   # 造出范数差异明显的行
        layer.weight.mul_(torch.linspace(0.2, 3.0, len(COUNTS))[:, None])

    class Stub(nn.Module):                  # 官方代码写死了 model.encoder.fc
        def __init__(self, fc):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.fc = fc

    ref_layer = copy.deepcopy(layer)
    ref = wb_reg.MaxNorm_via_PGD(thresh=0.1)
    ref.setPerLayerThresh(Stub(ref_layer))
    ours = MaxNormPGD(layer, thresh=0.1)
    assert abs(ours.limit - float(ref.perLayerThresh[0])) < 1e-6

    for _ in range(3):                      # 模拟若干次更新后的投影
        with torch.no_grad():
            noise = torch.randn_like(layer.weight)
            layer.weight.add_(noise)
            ref_layer.weight.add_(noise)
        ours.project()
        ref.PGD(Stub(ref_layer))
        assert torch.allclose(layer.weight, ref_layer.weight, atol=1e-6)
        assert torch.allclose(layer.bias, ref_layer.bias)


# ------------------------------------------------------------------ FlexMatch
@pytest.mark.parametrize("warmup", [True, False])
def test_flexmatch_threshold_matches_usb(warmup):
    N, C, steps, B = 200, 7, 40, 16
    g = torch.Generator().manual_seed(1)
    ours = FlexMatchThreshold(N, C, p_cutoff=0.95, thresh_warmup=warmup)
    ref = usb.FlexMatchThresholdingHook(ulb_dest_len=N, num_classes=C, thresh_warmup=warmup)

    class Alg:
        p_cutoff = 0.95

    for t in range(steps):
        # 置信度随步数上升，使选中集逐步扩大、阈值逐步变化
        sharp = 1.0 + 0.4 * t
        logits = torch.randn(B, C, generator=g) * sharp
        probs = logits.softmax(-1)
        idx = torch.randperm(N, generator=g)[:B]
        m_ours, _ = ours.mask(probs, idx)
        try:
            m_ref = ref.masking(Alg, probs, idx, softmax_x_ulb=False)
        except ValueError:
            # 官方在 warmup=False 且尚无任何选中样本时对空字典取 max 会抛错；
            # 我们的实现在该情形下保持阈值不变。此分支只在最初几步可能出现。
            assert not warmup
            continue
        assert torch.equal(m_ours, m_ref.float()), f"step {t}"
        assert torch.equal(ours.selected, ref.selected_label), f"step {t}"
        assert torch.allclose(ours.classwise_acc, ref.classwise_acc), f"step {t}"
    assert (ours.selected >= 0).any(), "测试未覆盖到阈值变化的阶段"


# ------------------------------------------------------------------ DANN
def test_grl_reverses_and_scales_gradient():
    x = torch.randn(4, 3, requires_grad=True)
    grad_reverse(x, 0.7).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.7))


def test_dann_lambda_schedule():
    assert dann_lambda(0.0) == pytest.approx(0.0)
    assert dann_lambda(1.0) == pytest.approx(2 / (1 + np.exp(-10)) - 1)
    vals = [dann_lambda(p) for p in np.linspace(0, 1, 11)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


# ------------------------------------------------------------------ Focal
def test_focal_reduces_to_ce_at_gamma_zero():
    torch.manual_seed(0)
    logits, y = torch.randn(16, 5), torch.randint(0, 5, (16,))
    assert torch.allclose(FocalLoss(0.0)(logits, y), F.cross_entropy(logits, y), atol=1e-6)


def test_focal_matches_formula():
    torch.manual_seed(0)
    logits, y = torch.randn(16, 5), torch.randint(0, 5, (16,))
    p = logits.softmax(-1).gather(1, y[:, None]).squeeze(1)
    ref = (-(1 - p) ** 2 * p.log()).mean()
    assert torch.allclose(FocalLoss(2.0)(logits, y), ref, atol=1e-6)


# ------------------------------------------------------------------ 目标域标签不可达
def test_target_labels_unreachable_in_training():
    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, i):
            return {"image": torch.zeros(1), "label": 5, "index": i}

    item = StripLabels(DS())[0]
    assert "label" not in item and item["index"] == 0
    with pytest.raises(KeyError):
        item["label"]


def test_head_matches_cdapdm_classifier():
    """基线分类头与 CD-APDM 同构 —— Table 1 的差异不应来自分类头容量。"""
    from src.models.backbone import CDAPDMModel
    a = BaselineNet(28, "resnet50", pretrained=False).classifier
    b = CDAPDMModel(28, temporal_input_dim=4, pretrained_backbone=False, use_temporal=False).classifier
    assert [type(m) for m in a] == [type(m) for m in b]
    assert [tuple(p.shape) for p in a.parameters()] == [tuple(p.shape) for p in b.parameters()]


# ------------------------------------------------------------------ CD-APDM 自身的目标域标签隔离
def test_ccganm_adaptation_never_reads_target_labels():
    """CC-GANM 曾以目标域真实标签作生成器条件（泄漏）。改正后目标域 batch 不含 label，
    类别条件只能来自 labeler；若代码回退到读 label，这里会抛 KeyError。"""
    from src.datasets.splits import StripLabels
    from src.inference import cc_ganm_train
    from src.models.cc_gan import CCGANConfig, CCGANM

    C = 4

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {"image": torch.rand(3, 32, 32), "sequence": torch.zeros(2), "label": i % C, "index": i}

    src = torch.utils.data.DataLoader(DS(), batch_size=2)
    tgt = torch.utils.data.DataLoader(StripLabels(DS()), batch_size=2)
    seen = []

    def labeler(img, seq):
        seen.append(img.shape[0])
        return torch.zeros(img.shape[0], dtype=torch.long)

    gan = CCGANM(CCGANConfig(hidden_dim=8, cycle_weight=10.0, adv_weight=1.0, num_classes=C))
    cc_ganm_train(gan, src, tgt, "cpu", iters=2, labeler=labeler)
    assert seen == [2, 2]


# ------------------------------------------------------------------ 评测指标
def test_macro_f1_matches_sklearn_with_never_predicted_class():
    from sklearn.metrics import f1_score
    from src.utils.metrics import macro_f1
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, 300)
    preds = labels.copy()
    noise = rng.random(300) < 0.4
    preds[noise] = rng.integers(0, 6, int(noise.sum()))
    preds[preds >= 4] = 0          # 类 4、5 从未被预测
    ours = macro_f1(preds, labels, 6)
    ref = f1_score(labels, preds, average="macro", labels=np.unique(labels), zero_division=0)
    assert ours == pytest.approx(ref)
    sub = [0, 1, 2, 3, 5]
    assert macro_f1(preds, labels, 6, classes=sub) == pytest.approx(
        f1_score(labels, preds, average="macro", labels=sub, zero_division=0))


# ------------------------------------------------------------------ 判别式微调
def test_backbone_lr_groups():
    from src.utils.optim import backbone_param_groups, build_optimizer, build_scheduler
    m = BaselineNet(28, "resnet50", pretrained=False)
    disc = torch.nn.Linear(2048, 1)
    params = list(m.parameters()) + list(disc.parameters())
    g = backbone_param_groups(m.backbone, params, 1e-3, 0.1)
    opt = build_optimizer(g, "adam", 1e-3, 5e-4)
    lrs = {x["name"]: x["lr"] for x in opt.param_groups}
    assert lrs == {"head": 1e-3, "backbone": pytest.approx(1e-4)}
    n_bb = sum(p.numel() for p in m.backbone.parameters())
    assert sum(p.numel() for p in opt.param_groups[1]["params"]) == n_bb
    assert sum(p.numel() for x in opt.param_groups for p in x["params"]) == sum(p.numel() for p in params)
    sch = build_scheduler(opt, "cosine", 10, 2)   # 调度器按组等比缩放
    for _ in range(4):
        opt.step(); sch.step()
    a, b = (x["lr"] for x in opt.param_groups)
    assert a == pytest.approx(10 * b)
    # 倍率为 1 时不分组，保持原结构（已有 resume.pt 可续跑）
    g1 = backbone_param_groups(m.backbone, params, 1e-4, 1.0)
    assert len(g1) == 1 and [id(p) for p in g1[0]["params"]] == [id(p) for p in params]


# ------------------------------------------------------------------ CD-APDM 自适应阶段（2026-09-22 更正）
def test_generator_output_in_imagenet_normalized_range():
    """生成图须与真实图同处 ImageNet 标准化空间；此前输出 Tanh∈[-1,1]，范围不一致。"""
    from src.models.cc_gan import IMAGENET_MEAN, IMAGENET_STD, CCGANConfig, ConditionalGenerator
    g = ConditionalGenerator(CCGANConfig(hidden_dim=8, num_classes=4))
    with torch.no_grad():
        for p in g.parameters():          # 放大权重使 Tanh 饱和，覆盖两端
            p.mul_(20)
        out = g(torch.randn(4, 3, 32, 32) * 3, torch.arange(4))
    lo = (0 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    hi = (1 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    for c in range(3):
        assert out[:, c].min() >= lo[c] - 1e-4 and out[:, c].max() <= hi[c] + 1e-4
    assert out.min() < -1.2 and out.max() > 1.2    # 确实超出了旧的 [-1, 1]


def test_mean_teacher_uses_eval_teacher_and_translated_labelled_source():
    from src.inference import mean_teacher_adaptation

    C = 4

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = nn.BatchNorm1d(3)
            s.fc = nn.Linear(3, C)
            s.modes, s.batches = [], []

        def forward(s, img, seq):
            s.modes.append(s.training)
            s.batches.append(img.shape[0])
            return s.fc(s.bn(img.mean(dim=(2, 3))))

    student, teacher = Net(), Net()
    teacher.load_state_dict(student.state_dict())
    teacher.train()                                   # 即使传入时处于 train，也须被切到 eval
    with torch.no_grad():
        teacher.fc.bias.copy_(torch.tensor([10.0, 0, 0, 0]))   # 教师高置信，伪标签全部保留

    class Tgt(torch.utils.data.Dataset):
        def __len__(s):
            return 6

        def __getitem__(s, i):
            return {"image": torch.rand(3, 8, 8), "sequence": torch.zeros(2), "index": i}

    class Src(Tgt):
        def __getitem__(s, i):
            return {**super().__getitem__(i), "label": 3}

    seen = {}

    class FakeGAN:
        def eval(s):
            return s

        def G_S2T(s, x, y):
            seen["labels"] = y.tolist()
            return x + 100.0                              # 可辨认的"迁移后"图像

    tgt = torch.utils.data.DataLoader(Tgt(), batch_size=3)
    src = torch.utils.data.DataLoader(Src(), batch_size=2)
    mean_teacher_adaptation(student, teacher, tgt, "cpu", torch.full((C,), 1 / C), 0.5, 0.99,
                            pseudo_conf_threshold=0.5, iters=2, ccgan=FakeGAN(), src_loader=src)
    assert teacher.modes and not any(teacher.modes), "教师须在 eval 模式下生成伪标签"
    assert all(student.modes), "学生须在 train 模式下更新"
    assert student.batches == [5, 5], "学生批 = 3 张真实目标图 + 2 张迁移源图"
    assert seen["labels"] == [3, 3], "迁移以源域真实标签为条件"


def test_mean_teacher_selection_variants():
    """开发集实验用的伪标签筛选方式：class_balanced 逐类只保留高置信部分；diversity 项可运行；
    acrm 模式按 ACRM 阈值筛选并更新其统计量。默认（fixed）行为由上一测试覆盖。"""
    from src.inference import mean_teacher_adaptation
    from src.modules.acrm import ACRM, ACRMConfig
    C = 4

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = nn.BatchNorm1d(3)
            s.fc = nn.Linear(3, C)

        def forward(s, img, seq):
            return s.fc(s.bn(img.mean(dim=(2, 3))))

    class Tgt(torch.utils.data.Dataset):
        def __len__(s):
            return 12

        def __getitem__(s, i):
            g = torch.Generator().manual_seed(i)
            return {"image": torch.rand(3, 8, 8, generator=g), "sequence": torch.zeros(2), "index": i}

    tgt = torch.utils.data.DataLoader(Tgt(), batch_size=4)
    pri = torch.full((C,), 1 / C)
    for kw in ({"selection": "class_balanced", "cb_keep": 0.5},
               {"selection": "fixed", "diversity_weight": 1.0},
               {"selection": "class_balanced", "cb_keep": 0.5, "diversity_weight": 0.5}):
        s, t = Net(), Net(); t.load_state_dict(s.state_dict())
        before = [p.clone() for p in s.parameters()]
        mean_teacher_adaptation(s, t, tgt, "cpu", pri, 0.5, 0.99, pseudo_conf_threshold=0.0, iters=3, **kw)
        assert any(not torch.equal(a, b) for a, b in zip(before, s.parameters())), kw
    acrm = ACRM(num_classes=C, class_counts=[100, 50, 10, 5], device="cpu",
                cfg=ACRMConfig(eta=0.9, tau_base=0.0, lambda_acc=0.1, lambda_ent=0.1, gamma_boost=0.0, tail_threshold=20))
    s, t = Net(), Net(); t.load_state_dict(s.state_dict())
    mean_teacher_adaptation(s, t, tgt, "cpu", pri, 0.5, 0.99, pseudo_conf_threshold=0.99, iters=3,
                            selection="acrm", acrm=acrm)
    assert acrm._initialized, "acrm 模式须在自适应过程中更新 ACRM 统计量"


def test_adabn_reestimates_bn_stats_only():
    from src.inference import adabn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = nn.BatchNorm2d(3)
            s.fc = nn.Linear(3, 2)

        def forward(s, img, seq):
            return s.fc(s.bn(img).mean(dim=(2, 3)))

    m = Net()
    w = [p.clone() for p in m.parameters()]
    data = [{"image": torch.randn(4, 3, 8, 8) * 2 + 5, "sequence": torch.zeros(4, 2)} for _ in range(3)]
    adabn(m, data, "cpu")
    allx = torch.cat([d["image"] for d in data])
    assert torch.allclose(m.bn.running_mean, allx.mean(dim=(0, 2, 3)), atol=1e-4)   # 累计平均 = 全体均值
    assert all(torch.equal(a, b) for a, b in zip(w, m.parameters())), "AdaBN 不得改动可学习参数"
    assert m.bn.momentum == 0.1 and not m.training
