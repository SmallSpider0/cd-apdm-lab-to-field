"""epoch 级断点续跑的公共部分。

服务器会被计划性重启（2026-09-22），长任务（Swin-B 约 3 小时）若只能整任务重跑，
代价过大。训练脚本每个 epoch 结束写一份 resume 文件，启动时若发现则从下一个 epoch 继续。

可复现性说明
  * 恢复模型、优化器、调度器、AMP scaler 以及 python/numpy/torch/CUDA 的随机状态，
    源域数据顺序由保存的 generator / sampler 状态精确接续。
  * cudnn.benchmark 开启，GPU 计算本就不是逐位确定的，因此续跑与不中断运行不是逐位一致，
    而是同分布的；被续跑过的运行在其产物中记录 ``resumed_at_epochs``，可供核查。
  * 不中断的运行行为与加入本功能前完全相同。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

RESUME_NAME = "resume.pt"


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path: Path) -> None:
    """先写临时文件再改名：重启恰好发生在写入中途时，不会留下损坏的 resume 文件。"""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_resume(out_dir: Path, **expect):
    """读取 resume 文件；``expect`` 中的键（如 method、seed）必须一致，否则拒绝续跑。"""
    p = Path(out_dir) / RESUME_NAME
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k, v in expect.items():
        if ck.get(k) != v:
            raise SystemExit(f"{p} 的 {k}={ck.get(k)!r} 与本次运行 {v!r} 不符，拒绝续跑；请确认后手动删除")
    return ck


def clear_resume(out_dir: Path) -> None:
    for name in (RESUME_NAME, RESUME_NAME + ".part"):
        (Path(out_dir) / name).unlink(missing_ok=True)
