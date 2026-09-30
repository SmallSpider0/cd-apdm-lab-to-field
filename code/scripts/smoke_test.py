"""Lightweight smoke test that exercises every CD-APDM module on a tiny
synthetic config — useful to verify the code runs end-to-end before
launching a full GPU training job."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.datasets.agrinet import AgriNetTFEM, AgriNetTFEMConfig  # noqa: E402
from src.datasets.image_dataset import AgriImageDataset, build_eval_transform  # noqa: E402
from src.models.backbone import CDAPDMModel  # noqa: E402
from src.modules.acrm import ACRM, ACRMConfig  # noqa: E402
from src.modules.ciwlm import CIWLMLoss  # noqa: E402
from src.modules.mdiwm import MDIWM, MDIWMConfig, compute_instance_weights  # noqa: E402
from src.utils.seed import pick_device, set_seed  # noqa: E402


def main():
    set_seed(0)
    device = torch.device("cpu")
    num_classes = 6

    # ---------------- TFEM
    tfem = AgriNetTFEM(AgriNetTFEMConfig(
        window_csv=str(ROOT / "../dataset/AgriNet/AgriNet_sliding_window_7d.csv"),
        align_csv=str(ROOT / "../dataset/AgriNet/AgriNet_image_alignment.csv"),
        feature_columns=[
            "temp_7d_mean", "temp_7d_std", "humidity_7d_mean",
            "soil_moisture_7d_mean", "precip_7d_total",
        ],
        event_columns=["fertilization_event", "extreme_weather_event"],
        window_days=7,
    ))
    print(f"TFEM data loaded: {tfem.has_data()}  feature_dim={tfem.feature_dim}")

    # ---------------- datasets (synthetic fallback)
    src = AgriImageDataset(root="__missing__", domain_id=0, num_classes=num_classes,
                           tfem=tfem, train=True, image_size=64,
                           synthetic_when_missing=True, synthetic_head=24, synthetic_tail=4)
    tgt = AgriImageDataset(root="__missing__", domain_id=1, num_classes=num_classes,
                           tfem=tfem, train=False, image_size=64,
                           synthetic_when_missing=True, synthetic_head=12, synthetic_tail=2)
    print(f"src counts={src.class_counts}  tgt counts={tgt.class_counts}")
    sample = src[0]
    print(f"sample image {sample['image'].shape}, sequence {sample['sequence'].shape}")

    # ---------------- model
    model = CDAPDMModel(
        num_classes=num_classes,
        temporal_input_dim=tfem.feature_dim,
        temporal_hidden=32,
        mhsa_heads=2,
        pretrained_backbone=False,
    ).to(device)
    img = sample["image"].unsqueeze(0)
    seq = sample["sequence"].unsqueeze(0)
    logits = model(img, seq)
    print("forward ok, logits shape =", logits.shape)

    # ---------------- CIWLM
    loss = CIWLMLoss(class_counts=src.class_counts).to(device)
    out = loss(logits, torch.tensor([sample["label"]]))
    print("CIWLM loss =", out.item())

    # ---------------- MDIWM weights
    w = compute_instance_weights(
        domain_scores=torch.rand(4).numpy(),
        uncertainties=torch.rand(4).numpy(),
        alpha=0.7, beta=0.3,
    )
    print("MDIWM weights =", w)

    # ---------------- ACRM
    acrm = ACRM(num_classes=num_classes, class_counts=src.class_counts,
                cfg=ACRMConfig(), device=device)
    probs = torch.softmax(torch.randn(4, num_classes), dim=-1)
    acrm.update_stats(probs.argmax(-1), probs.argmax(-1), probs)
    rect = acrm.rectify(probs)
    print("ACRM thresholds =", acrm.thresholds())
    print("ACRM rectified sample sum =", rect.sum(-1))


if __name__ == "__main__":
    main()
