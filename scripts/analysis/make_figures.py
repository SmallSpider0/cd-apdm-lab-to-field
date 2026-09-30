"""重绘稿件 Figs 2–5（revise-cdapdm-method 9.3），只读已取回的运行产物，输出到 submission/v1/figures/。
同时导出 PDF（矢量，字体内嵌）供投稿系统单独上传；稿件内仍嵌 PNG。

    python scripts/analysis/make_figures.py

Fig. 2  Grad-CAM：源域训练 ResNet-50 与 CD-APDM（seed 0）在同一批尾类测试图像上的热力图
        （results/remote/exp-final/figdata-s0/figdata.npz；选图规则见 code/tools/figure_data.py，不看图挑选）
Fig. 3  t-SNE：两模型分类头前特征，源域验证集与目标测试集（同上）
Fig. 4  尾类样本可靠性图：ResNet-50（源域训练）、Focal Loss + AdaBN + LA、CD-APDM，6 seed 合并（results/exp-offline-inference.json）
Fig. 5  尾类混淆矩阵（按行归一化）：Focal Loss + AdaBN + LA 与 CD-APDM，6 seed 合并（同上）
"""
import csv
import textwrap
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
FD = ROOT / "results/remote/exp-final/figdata-s0"
OFF = json.load(open(ROOT / "results/exp-offline-inference.json"))
NAMES = OFF["class_names_pretty"]
plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans", "axes.linewidth": 0.5,
                     "pdf.fonttype": 42})     # PDF 内嵌 TrueType 字体（Elsevier 要求矢量图嵌入字体）


def test_mask(ids):
    sp = {r["path"]: r["eval_split"] for r in csv.DictReader(open(ROOT / "dataset/splits/plantdoc_target.csv"))}
    return np.array([sp[str(i)] == "test" for i in ids])


def fig2(z):
    n = len(z["cam_image_ids"])
    fig, ax = plt.subplots(2, n, figsize=(6.5, 2.6))
    wrap = lambda t: "\n".join(textwrap.wrap(t, 16))
    for j in range(n):
        f = str(z["cam_image_ids"][j]).replace("/", "_").replace(" ", "_")
        img = np.asarray(Image.open(FD / "images" / f).convert("RGB").resize((224, 224)))
        for i, (cam, pred) in enumerate(((z["cam_base"][j], z["cam_base_pred"][j]), (z["cam_ours"][j], z["cam_ours_pred"][j]))):
            a = ax[i, j]
            a.imshow(img); a.imshow(cam.astype(np.float32), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ok = int(pred) == int(z["cam_label"][j])
            a.set_xlabel(wrap(("✓ " if ok else "✗ ") + NAMES[int(pred)]), fontsize=4.5, color="black" if ok else "firebrick", labelpad=1.5)
            a.set_xticks([]); a.set_yticks([])
        ax[0, j].set_title(wrap(f"({chr(97 + j)}) {NAMES[int(z['cam_label'][j])]}"), fontsize=4.8)
    ax[0, 0].set_ylabel("ResNet-50\n(source only)", fontsize=6)
    ax[1, 0].set_ylabel("CD-APDM", fontsize=6)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.6)
    return fig


def fig3(z):
    tail = set(int(c) for c in z["tail_classes"])
    tm = test_mask(z["tgt_ids"])
    fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for a, (name, src, tgt) in zip(ax, (("ResNet-50 (source only)", z["base_src_feat"], z["base_tgt_feat"]),
                                        ("CD-APDM (after AdaBN)", z["ours_src_feat"], z["ours_tgt_feat"]))):
        X = np.concatenate([src, tgt[tm]]).astype(np.float32)
        y = np.concatenate([z["src_labels"], z["tgt_labels"][tm]])
        dom = np.r_[np.zeros(len(src)), np.ones(tm.sum())]
        E = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(PCA(50, random_state=0).fit_transform(X))
        is_tail = np.array([int(c) in tail for c in y])
        for d, mk, lab in ((0, "o", "source"), (1, "^", "target")):
            for t, col in ((False, "#4a78b5"), (True, "#c0392b")):
                m = (dom == d) & (is_tail == t)
                a.scatter(E[m, 0], E[m, 1], s=1.2 if d == 0 else 2.0, marker=mk, c=col, alpha=0.35 if d == 0 else 0.8,
                          linewidths=0, label=f"{lab}, {'tail' if t else 'head'}")
        a.set_title(name); a.set_xticks([]); a.set_yticks([])
    ax[1].legend(fontsize=5, markerscale=3, loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    return fig


def fig4():
    rel = OFF["fig4_reliability_tail"]
    fig, ax = plt.subplots(1, 3, figsize=(5.0, 1.75), sharey=True)
    for a, (k, name) in zip(ax, (("resnet50_source_only", "ResNet-50 (source only)"), ("focal_adabn_la", "Focal Loss + AdaBN + LA"),
                                 ("cd_apdm", "CD-APDM"))):
        b = [x for x in rel[k]["bins10"] if x["n"] > 0]
        centers = [(x["lo"] + x["hi"]) / 2 for x in b]
        a.bar(centers, [x["acc"] for x in b], width=0.09, color="#4a78b5", edgecolor="white", linewidth=0.3, label="accuracy")
        a.plot([0, 1], [0, 1], "k--", linewidth=0.6)
        a.set_title(f"{name}\ntail ECE = {rel[k]['tail_ece_15bins']:.3f}")
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("confidence")
    ax[0].set_ylabel("accuracy")
    fig.tight_layout(pad=0.3)
    return fig


def fig5():
    c = OFF["fig5_confusion_tail"]
    labels = [NAMES[i] for i in c["rows_true_tail"]]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.4))
    for a, (k, name) in zip(ax, (("focal_adabn_la", "Focal Loss + AdaBN + LA"), ("cd_apdm", "CD-APDM"))):
        M = np.array(c["counts"][k], float)
        M = M / M.sum(1, keepdims=True).clip(1)
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.05:
                    a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4, color="white" if M[i, j] > 0.5 else "black")
        a.set_xticks(range(M.shape[1])); a.set_xticklabels(labels + ["other (non-tail)"], rotation=60, ha="right", fontsize=4.5)
        a.set_yticks(range(M.shape[0])); a.set_yticklabels(labels if a is ax[0] else [], fontsize=4.5)
        a.set_title(f"{name}  (tail recall {np.trace(M[:, :-1]) / M.shape[0] * 100:.1f}%)")
        a.set_xlabel("predicted")
    ax[0].set_ylabel("true (tail classes)")
    fig.subplots_adjust(left=0.16, right=0.9, bottom=0.27, top=0.93, wspace=0.05)
    fig.colorbar(im, cax=fig.add_axes([0.915, 0.27, 0.012, 0.66]))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(FD / "figdata.npz")
    for name, f in (("fig2_gradcam", lambda: fig2(z)), ("fig3_tsne", lambda: fig3(z)), ("fig4_reliability", fig4), ("fig5_confusion", fig5)):
        fig = f()
        fig.savefig(OUT / f"{name}.png", dpi=300)
        fig.savefig(OUT / f"{name}.pdf", dpi=300)       # 矢量版供单独上传（位图部分如 Grad-CAM 按 300 dpi 嵌入）
        plt.close(fig)
        w, h = Image.open(OUT / f"{name}.png").size
        print(f"{name}.png  {w}×{h}")


if __name__ == "__main__":
    main()
