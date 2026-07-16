from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import pandas as pd


class AssetType(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    CONVERTIBLE_BOND = "convertible_bond"


class Frequency(StrEnum):
    MIN1 = "1min"
    MIN5 = "5min"
    MIN15 = "15min"
    MIN30 = "30min"
    MIN60 = "60min"
    DAY = "1d"


class Dataset(StrEnum):
    BARS = "bars"
    DAILY_BASIC = "daily_basic"
    ADJ_FACTOR = "adj_factor"
    TRADE_CALENDAR = "trade_calendar"


@dataclass(frozen=True)
class DataRequest:
    dataset: Dataset
    symbols: tuple[str, ...] = ()
    start: date | datetime | None = None
    end: date | datetime | None = None
    frequency: Frequency = Frequency.DAY
    asset_type: AssetType = AssetType.STOCK
    adjustment: str = "none"
    provider: str = "auto"
    fields: tuple[str, ...] = ()


@dataclass
class DataBatch:
    data: pd.DataFrame
    provider: str
    request: DataRequest
    fetched_at: datetime = field(default_factory=datetime.now)
    as_of: datetime | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
]
