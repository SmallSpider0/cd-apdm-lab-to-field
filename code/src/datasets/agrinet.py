"""AgriNet 时序特征加载器（TFEM 数据侧）。

- 载入预计算的滑动窗口特征
- 前向填充缺失槽位，上限 ``max_gap``（论文：6 槽 = 3 小时）
- 按对齐表（station + window_end_date）提供 ``image_id -> 特征序列``

**重要：AgriNet 是模拟数据。** 其自述为 12 个 virtual IoT 设备，基础数据来自
NASA POWER (MERRA-2) 再分析产品，日温由正弦模型生成。对齐表中的 ``image_id``
形如 ``IMG_S01_2025-05-01_00``，是生成的占位标识，**不对应任何真实图像文件名**。

PlantVillage 与 PlantDoc 均无可信的采集时间戳与地块标识，因此在主基准上
**无法诚实地建立图像与时序的对应关系**（审稿意见 R2-2 所问）。主基准的默认
做法是不启用 TFEM（``use_temporal=False``）。

``strict=True``（默认）时，对齐失败即抛错。此前的实现会在查不到对齐记录时
静默返回全零向量，调用方再据此**注入一条随机的 AgriNet 时序** —— 等于凭空
编造模型输入，且在真实数据上必然触发（占位标识永不匹配真实文件名）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
# pandas 只在 TFEM 启用时才需要。主基准按 T-2 决议关闭 TFEM
# （configs/cd_apdm.yaml 的 tfem.enabled: false），因此不应把它作为硬依赖 ——
# 服务器环境无外网、装不了包，而主实验根本用不到它。
try:
    import pandas as pd
except ImportError as _e:  # pragma: no cover
    pd = None
    _PANDAS_ERR = _e
import torch


@dataclass
class AgriNetTFEMConfig:
    window_csv: str
    align_csv: str
    feature_columns: Sequence[str]
    event_columns: Sequence[str]
    window_days: int = 7
    forward_fill_max_gap: int = 6
    strict: bool = True      # 对齐失败时抛错而非静默返回零向量


class AgriNetTFEM:
    """Provides sliding-window temporal features for an image_id.

    The returned tensor has shape ``(window_days, n_features)`` and is
    ready to be fed into the TFEM LSTM encoder.
    """

    def __init__(self, cfg: AgriNetTFEMConfig):
        self.cfg = cfg
        self.window_df = self._load_window(cfg.window_csv)
        self.align_df = self._load_alignment(cfg.align_csv)
        self._feature_dim = len(cfg.feature_columns) + len(cfg.event_columns)
        self._cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def _load_window(path: str):
        if pd is None:
            raise RuntimeError(
                "TFEM 需要 pandas，但当前环境未安装。"
                "主基准按 T-2 决议关闭 TFEM（tfem.enabled: false），"
                "若确需启用请先安装 pandas。"
            )
        if not Path(path).exists():
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=["date"])
        # forward-fill within a station, capped by max gap (handled below)
        df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
        return df

    @staticmethod
    def _load_alignment(path: str):
        if pd is None:
            raise RuntimeError("TFEM 需要 pandas，但当前环境未安装。")
        if not Path(path).exists():
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=["window_end_date", "window_start_date"])
        return df

    # ------------------------------------------------------------------ API
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def has_data(self) -> bool:
        return not self.window_df.empty and not self.align_df.empty

    def get_sequence(self, image_id: str) -> torch.Tensor:
        if image_id in self._cache:
            return self._cache[image_id]

        if not self.has_data():
            if self.cfg.strict:
                raise RuntimeError(
                    "AgriNet 窗口特征或对齐表为空，无法提供时序序列。\n"
                    "若本次实验不使用时序模态，请以 temporal=None 构建数据集、"
                    "并以 use_temporal=False 构建模型，而不是让其静默退化为零向量。"
                )
            seq = torch.zeros(self.cfg.window_days, self.feature_dim)
            self._cache[image_id] = seq
            return seq

        align_row = self.align_df.loc[self.align_df.image_id == image_id]
        if align_row.empty:
            if self.cfg.strict:
                raise KeyError(
                    f"对齐表中没有 image_id={image_id!r}。\n"
                    "AgriNet 对齐表的标识形如 IMG_S01_2025-05-01_00，是生成的占位标识，"
                    "不对应任何真实图像文件名；PlantVillage / PlantDoc 也不带可信的采集"
                    "时间戳与地块标识。主基准应以 use_temporal=False 运行。\n"
                    "如确需时序模态，须先建立并公开一套可追溯的对齐规则。"
                )
            seq = torch.zeros(self.cfg.window_days, self.feature_dim)
            self._cache[image_id] = seq
            return seq

        station = align_row.iloc[0]["station_id"]
        end_date = pd.Timestamp(align_row.iloc[0]["window_end_date"])
        start_date = end_date - pd.Timedelta(days=self.cfg.window_days - 1)

        sub = self.window_df[
            (self.window_df.station_id == station)
            & (self.window_df.date >= start_date)
            & (self.window_df.date <= end_date)
        ].copy()

        # forward-fill bounded gaps for the requested feature columns
        cols = list(self.cfg.feature_columns) + list(self.cfg.event_columns)
        sub = sub.set_index("date").reindex(
            pd.date_range(start_date, end_date, freq="D")
        )
        sub[cols] = sub[cols].ffill(limit=self.cfg.forward_fill_max_gap)
        sub[cols] = sub[cols].fillna(0.0)

        arr = sub[cols].to_numpy(dtype=np.float32)
        if arr.shape[0] < self.cfg.window_days:
            pad = np.zeros((self.cfg.window_days - arr.shape[0], len(cols)), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)

        seq = torch.from_numpy(arr)
        self._cache[image_id] = seq
        return seq

    def random_sequence(self, rng: np.random.Generator | None = None) -> torch.Tensor:
        """随机抽取一条站点-日序列。

        **不得用于真实数据的对齐回退。** 随机时序与图像无任何对应关系，
        用它填补缺失的对齐等于编造模型输入。仅保留给
        EXP-6 置换检验（刻意打乱对齐关系以检测时序增益是否为伪影）使用。
        """
        if not self.has_data():
            raise RuntimeError("AgriNet 数据为空，无法抽样")
        rng = rng or np.random.default_rng()
        ids = self.align_df["image_id"].to_numpy()
        prev, self.cfg.strict = self.cfg.strict, True
        try:
            return self.get_sequence(str(rng.choice(ids)))
        finally:
            self.cfg.strict = prev
