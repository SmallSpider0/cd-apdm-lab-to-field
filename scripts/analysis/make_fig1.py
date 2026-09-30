"""重画稿件 Fig. 1（框架图），输出 submission/v1/figures/fig1_framework.{png,pdf}。

    ~/miniconda3/envs/agri-ctta/bin/python scripts/analysis/make_fig1.py

原图（figures/framework_architecture.*，matplotlib 生成，源脚本不在仓库）把 TFEM、MHSA 融合、CC-GANM、SS-PLAM
画在主流程上，没有 AdaBN。2026-09-30 按报告配置重画（二审满意度评估，编辑意见 E4 与 R2-2）：
实线彩色框为报告配置实际运行的步骤；灰色虚线框为原设计中已停用的组件（负面结果见 Table 2 与补充材料 S6）。
按印刷尺寸 6.5 × 2.5 in 排版，最小字号 6.5 pt；PNG 600 dpi，PDF 内嵌 TrueType 字体。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/fig1_framework"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})

COL = {"input": ("#e3f0fb", "#2f77c7"), "train": ("#e8f4e8", "#3d8b40"), "infer": ("#fff1e0", "#e07b12"),
       "core": ("#f1e6f6", "#7b3aa0"), "output": ("#fdeaea", "#c73a3a"), "off": ("#f2f2f2", "#9a9a9a")}
INK, MUTED = "#1a1a1a", "#6b6b6b"


def box(ax, x0, x1, y0, y1, text, kind, bold=False):
    fc, ec = COL[kind]
    off = kind == "off"
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=0.8, ls=(0, (3, 2)) if off else "-"))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=6.5,
            color=MUTED if off else INK, fontweight="bold" if bold else "normal", linespacing=1.15)
    return (x0, x1, y0, y1)


def arrow(ax, p, q, kind="data", rad=0.0):
    style = {"data": dict(color="#444444", lw=0.9, ls="-"), "off": dict(color="#a8a8a8", lw=0.8, ls=(0, (3, 2))),
             }[kind]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", **style))


def main():
    fig = plt.figure(figsize=(6.5, 2.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 50); ax.axis("off")
    X = [(2, 20), (24, 42), (46, 64), (68, 86), (90, 108)]      # 五列框的横向位置

    for y0, y1, name in ((25.5, 49.5, "Training phase"), (0.5, 24.5, "Inference phase")):
        ax.add_patch(FancyBboxPatch((0.5, y0), 109, y1 - y0, boxstyle="round,pad=0,rounding_size=1",
                                    fc="#fafafa", ec="#d0d0d0", lw=0.5))
        ax.text(1.8, y1 - 1.6, name, fontsize=7, fontweight="bold", color=INK, va="center")

    # 训练阶段：报告配置（实线）与已停用组件（灰色虚线）
    y0, y1, z0, z1 = 35.5, 44, 27, 33.5
    top = [box(ax, *X[0], y0, y1, "Source images\nPlantVillage-LT", "input"),
           box(ax, *X[1], y0, y1, "MDIWM\ninstance weights", "train"),
           box(ax, *X[2], y0, y1, "CIWLM\nweighted loss", "train"),
           box(ax, *X[3], y0, y1, "ResNet-50\nbackbone", "core", bold=True),
           box(ax, *X[4], y0, y1, "Classifier\n+ temperature T", "output")]
    for a, b in zip(top, top[1:]):
        arrow(ax, (a[1], (y0 + y1) / 2), (b[0], (y0 + y1) / 2))
    ts = box(ax, *X[0], z0, z1, "Time series\n(disabled)", "off")
    tfem = box(ax, *X[1], z0, z1, "TFEM\n(disabled)", "off")
    mhsa = box(ax, *X[3], z0, z1, "MHSA fusion\n(disabled)", "off")
    zm = (z0 + z1) / 2
    arrow(ax, (ts[1], zm), (tfem[0], zm), "off")
    arrow(ax, (tfem[1], zm), (mhsa[0], zm), "off")
    arrow(ax, (sum(X[3]) / 2, z1), (sum(X[3]) / 2, y0), "off")

    # 推理阶段
    u0, u1, v0, v1 = 11.5, 20, 2.5, 9
    bot = [box(ax, *X[0], u0, u1, "Target images\n(unlabeled)", "input"),
           box(ax, *X[1], u0, u1, "AdaBN on the\ntrained model", "infer", bold=True),
           box(ax, *X[2], u0, u1, "Logit adjustment\n$z/T-\\lambda\\log\\pi_s$", "infer"),
           box(ax, *X[3], u0, u1, "ACRM tail-class\nrectification", "infer"),
           box(ax, *X[4], u0, u1, "Prediction\n$\\hat{p}$", "output")]
    for a, b in zip(bot, bot[1:]):
        arrow(ax, (a[1], (u0 + u1) / 2), (b[0], (u0 + u1) / 2))
    ax.text(X[0][0], (v0 + v1) / 2, "Originally proposed,\ndisabled (Table 2):", fontsize=6.5, color=MUTED,
            va="center", linespacing=1.15)
    box(ax, *X[1], v0, v1, "CC-GANM\nstyle transfer", "off")
    box(ax, *X[2], v0, v1, "SS-PLAM +\nMean Teacher", "off")
    box(ax, *X[3], v0, v1, "ACRM adaptive\nthreshold", "off")

    # 图例
    items = [("input", "Input"), ("train", "Training"), ("core", "Backbone"), ("infer", "Inference"),
             ("output", "Output"), ("off", "Disabled")]
    for k, (kind, lab) in enumerate(items):
        yy = 45 - k * 5.5
        fc, ec = COL[kind]
        ax.add_patch(FancyBboxPatch((112.5, yy - 1.3), 3.4, 2.6, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=fc, ec=ec, lw=0.7, ls=(0, (2, 1.5)) if kind == "off" else "-"))
        ax.text(117, yy, lab, fontsize=6.5, va="center", color=INK)
    arrow(ax, (112.5, 10), (116, 10))
    ax.text(117, 10, "Data flow", fontsize=6.5, va="center", color=INK)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=600)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"已生成 {OUT.relative_to(ROOT)}.png / .pdf")


if __name__ == "__main__":
    main()
