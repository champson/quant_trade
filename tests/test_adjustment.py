from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.data.base import PermanentProviderError
from quant_trade.data.providers.tushare import TushareProvider
from quant_trade.data.storage import DataStore
from quant_trade.models import DataRequest, Dataset


def _rows(symbol: str, closes: list[float], adjustment: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "symbol": symbol,
            "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
            "bar_time": pd.NaT, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 100.0, "amount": 1000.0, "source": "fake",
        }
        for i, close in enumerate(closes)
    ])
    if adjustment is not None:
        frame["adjustment"] = adjustment
    return frame


def test_write_and_read_daily_segregate_adjustments(app_config):
    store = DataStore(app_config)
    # Legacy rows without the column count as unadjusted.
    store.write_daily(_rows("510300.SH", [10.0, 11.0]), "etf")
    store.write_daily(_rows("510300.SH", [20.0, 21.0], "hfq"), "etf")
    raw = store.read_daily(["510300.SH"])
    hfq = store.read_daily(["510300.SH"], adjustment="hfq")
    assert raw["close"].tolist() == [10.0, 11.0]
    assert hfq["close"].tolist() == [20.0, 21.0]


def test_adjusted_rows_do_not_overwrite_unadjusted_rows(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", [10.0], "none"), "etf")
    store.write_daily(_rows("510300.SH", [99.0], "hfq"), "etf")
    raw = store.read_daily(["510300.SH"], adjustment="none")
    assert raw["close"].tolist() == [10.0]


def test_tushare_rejects_adjusted_bar_requests():
    provider = TushareProvider()
    request = DataRequest(
        dataset=Dataset.BARS, symbols=("510300.SH",),
        start=date(2024, 1, 1), end=date(2024, 1, 31), adjustment="hfq",
    )
    assert not provider.supports(request)
    with pytest.raises(PermanentProviderError):
        provider.fetch(request)
