from __future__ import annotations

from datetime import date

import pandas as pd

from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType, DataBatch
from quant_trade.services import update_bars


def _rows(symbol: str, days) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "symbol": symbol, "trade_date": day, "bar_time": pd.NaT,
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
            "volume": 100.0, "amount": 1000.0, "source": "fake",
        }
        for day in days
    ])


class RecordingRouter:
    def __init__(self):
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        days = pd.bdate_range(request.start, request.end)
        return DataBatch(_rows(request.symbols[0], days), "fake", request)


def test_update_bars_refetches_missing_history(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("000001.SZ", pd.bdate_range("2024-06-03", "2024-06-14")), "stock")
    router = RecordingRouter()
    update_bars(
        app_config, router, store, ["000001.SZ"],
        date(2024, 1, 2), date(2024, 6, 14), AssetType.STOCK,
    )
    # A recent-only cache must not hide the missing months before it.
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 1, 2)


def test_update_bars_skips_when_range_covered(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("000001.SZ", pd.bdate_range("2024-01-02", "2024-06-14")), "stock")
    router = RecordingRouter()
    frame = update_bars(
        app_config, router, store, ["000001.SZ"],
        date(2024, 1, 2), date(2024, 6, 14), AssetType.STOCK,
    )
    assert router.requests == []
    assert frame.empty


def test_update_bars_cache_is_adjustment_scoped(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", pd.bdate_range("2024-01-02", "2024-06-14")), "etf")
    router = RecordingRouter()
    update_bars(
        app_config, router, store, ["510300.SH"],
        date(2024, 1, 2), date(2024, 6, 14), AssetType.ETF, adjustment="hfq",
    )
    # Unadjusted cache must not satisfy an hfq request.
    assert len(router.requests) == 1
    assert router.requests[0].adjustment == "hfq"


def test_update_bars_extends_tail_incrementally(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("000001.SZ", pd.bdate_range("2024-01-02", "2024-06-07")), "stock")
    router = RecordingRouter()
    update_bars(
        app_config, router, store, ["000001.SZ"],
        date(2024, 1, 2), date(2024, 6, 14), AssetType.STOCK,
    )
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 6, 8)
