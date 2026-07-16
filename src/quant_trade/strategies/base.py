from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    display_name: str
    description: str
    frequency: str = "1d"


@dataclass
class SignalResult:
    as_of: pd.Timestamp
    target_weights: pd.Series
    diagnostics: pd.DataFrame
    summary: str


class Strategy(ABC):
    metadata: StrategyMetadata

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def generate_targets(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return close-generated target weights indexed by signal time."""

    def latest_signal(self, bars: pd.DataFrame) -> SignalResult:
        targets = self.generate_targets(bars)
        if targets.empty:
            raise ValueError("没有足够数据生成信号")
        as_of = pd.Timestamp(targets.index[-1])
        weights = targets.iloc[-1]
        selected = weights[weights > 0]
        summary = (
            "空仓" if selected.empty else "、".join(f"{k} {v:.1%}" for k, v in selected.items())
        )
        return SignalResult(as_of, weights, pd.DataFrame(), summary)
