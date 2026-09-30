"""Figs 2–3 的数据：Grad-CAM 热力图与 t-SNE 所用特征（绘图在本机完成，服务器无 matplotlib）。

    python -m tools.figure_data --ours <cd_apdm_bestval.pt> --baseline <source_only 产物目录> \
        --ours_probs <ours probs_full.npz> --baseline_probs <probs_bestval.npz> --out <npz>

对比对象：源域训练的 ResNet-50（不自适应）与定稿 CD-APDM（AdaBN 后）。同一 seed。
- 特征：分类头之前的 2,048 维特征，目标域测试集全部图像 + 源域验证集全部图像（t-SNE 在本机计算）。
- Grad-CAM（Selvaraju et al. 2017）：以骨干 layer4 输出为目标层，对各模型自身预测类别的 logit 求梯度。
  类别取各模型的最终预测（CD-APDM 含 logit 调整与尾类重标定）。选图规则写死、不看图挑选：测试集中尾类（Medium/Few-shot）样本里，CD-APDM 预测正确而基线预测错误者，
  每个真实类别取置信度最高的 1 张，最多 6 张；另按同一规则取两者都预测错误的样本最多 2 张，避免只展示成功案例。
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.baselines.net import BaselineNet
from src.datasets.splits import SPLIT_DIR, StripLabels, build_datasets
from src.inference import adabn, load_model
from src.utils.config import load_config


@torch.no_grad()
def features(fwd, ds, dev):
    F_, Y, ids = [], [], []
    for b in DataLoader(ds, batch_size=128, num_workers=6):
        F_.append(fwd(b["image"].to(dev)).float().cpu()); Y.append(b["label"]); ids += list(b["image_id"])
    return torch.cat(F_).numpy(), torch.cat(Y).numpy(), np.array(ids)


def gradcam(model_logits, layer, img, dev, c):
    """对给定类别 c（取各模型最终预测，CD-APDM 含 logit 调整与尾类重标定）的 logit 求 Grad-CAM。"""
    acts = {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: acts.__setitem__("g", go[0]))
    x = img.unsqueeze(0).to(dev).requires_grad_(True)
    logits = model_logits(x)
    logits[0, c].backward()
    h1.remove(); h2.remove()
    w = acts["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True); ap.add_argument("--baseline", required=True)
    ap.add_argument("--ours_probs", required=True); ap.add_argument("--baseline_probs", required=True)
    ap.add_argument("--config", default="configs/cd_apdm.yaml"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = torch.device("cuda")
    _, src_val, tgt, ci = build_datasets(cfg, temporal=None)

    ck = torch.load(f"{a.baseline}/model_bestval.pt", map_location="cpu", weights_only=False)
    base = BaselineNet(ck["num_classes"], ck["backbone"], pretrained=False).to(dev).eval()
    base.load_state_dict(ck["model"])
    ours, ock = load_model(a.ours, cfg, 16, dev)
    ours.use_temporal = False
    ours_src = {k: v.clone() for k, v in ours.state_dict().items()}   # AdaBN 前的状态：源域特征用它提取
    adabn(ours, DataLoader(StripLabels(tgt), batch_size=64, num_workers=6), dev)
    ours.eval()

    out = {}
    f, y, ids = features(lambda x: base(x, return_features=True)[1], tgt, dev)
    out.update(base_tgt_feat=f.astype(np.float16), tgt_labels=y, tgt_ids=ids)
    f, y, sids = features(lambda x: base(x, return_features=True)[1], src_val, dev)
    out.update(base_src_feat=f.astype(np.float16), src_labels=y, src_ids=sids)
    f, _, _ = features(lambda x: ours(x, return_features=True)[1], tgt, dev)
    out.update(ours_tgt_feat=f.astype(np.float16))
    adapted = {k: v.clone() for k, v in ours.state_dict().items()}
    ours.load_state_dict(ours_src); ours.eval()
    f, _, _ = features(lambda x: ours(x, return_features=True)[1], src_val, dev)
    out.update(ours_src_feat=f.astype(np.float16))
    ours.load_state_dict(adapted); ours.eval()

    # ---- Grad-CAM 选图（规则见文件头）
    split = {r["path"]: r["eval_split"] for r in csv.DictReader(open(SPLIT_DIR / "plantdoc_target.csv"))}
    po, pb = np.load(a.ours_probs), np.load(a.baseline_probs)
    assert (po["image_ids"] == pb["image_ids"]).all()
    lab, pid = po["labels"], [str(i) for i in po["image_ids"]]
    tail = set(ock["tail_classes"])
    test_tail = np.array([split[i] == "test" and int(l) in tail for i, l in zip(pid, lab)])
    co, cb = po["probs"].argmax(1) == lab, pb["probs"].argmax(1) == lab
    conf = po["probs"].max(1)
    chosen = []
    for cond, k in ((test_tail & co & ~cb, 6), (test_tail & ~co & ~cb, 2)):
        seen = set()
        for i in np.argsort(-conf):
            if cond[i] and lab[i] not in seen and len(seen) < k:
                seen.add(lab[i]); chosen.append((pid[i], "ours_correct_base_wrong" if k == 6 else "both_wrong"))
    idx_of = {str(i): n for n, i in enumerate(ids)}
    pos = {i: n for n, i in enumerate(pid)}
    cams = []
    for image_id, kind in chosen:
        item = tgt[idx_of[image_id]]
        cb_c, co_c = int(pb["probs"][pos[image_id]].argmax()), int(po["probs"][pos[image_id]].argmax())
        cb_ = gradcam(lambda x: base(x), base.backbone.layer4, item["image"], dev, cb_c)
        co_ = gradcam(lambda x: ours(x), ours.backbone.layer4, item["image"], dev, co_c)
        cams.append({"image_id": image_id, "kind": kind, "label": int(item["label"]),
                     "base_pred": cb_c, "ours_pred": co_c, "base_cam": cb_, "ours_cam": co_})
    out.update(cam_image_ids=np.array([c["image_id"] for c in cams]), cam_kind=np.array([c["kind"] for c in cams]),
               cam_label=np.array([c["label"] for c in cams]), cam_base_pred=np.array([c["base_pred"] for c in cams]),
               cam_ours_pred=np.array([c["ours_pred"] for c in cams]),
               cam_base=np.stack([c["base_cam"] for c in cams]) if cams else np.zeros((0, 224, 224), np.float16),
               cam_ours=np.stack([c["ours_cam"] for c in cams]) if cams else np.zeros((0, 224, 224), np.float16),
               tail_classes=np.array(sorted(tail)))
    np.savez_compressed(a.out, **out)
    print(f"已写入 {a.out}：目标 {len(ids)} 张、源域验证 {len(sids)} 张特征；Grad-CAM {len(cams)} 张")


if __name__ == "__main__":
    main()
