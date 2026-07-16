from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.data.base import PermanentProviderError
from quant_trade.data.providers.baostock import BaoStockProvider
from quant_trade.data.providers.tushare import TushareProvider
from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType, DataRequest, Dataset


def _rows(symbol: str, closes: list[float], adjustment: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
                "bar_time": pd.NaT,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 100.0,
                "amount": 1000.0,
                "source": "fake",
            }
            for i, close in enumerate(closes)
        ]
    )
    if adjustment is not None:
        frame["adjustment"] = adjustment
    return frame


def test_write_and_read_daily_segregate_adjustments(app_config):
    store = DataStore(app_config)
    # Legacy rows without the column count as unadjusted.
    store.write_daily(_rows("510300.SH", [10.0, 11.0]), "etf")
    store.write_daily(_rows("510300.SH", [20.0, 21.0], "hfq"), "etf")
    raw = store.read_daily(["510300.SH"], asset_type=AssetType.ETF)
    hfq = store.read_daily(["510300.SH"], asset_type=AssetType.ETF, adjustment="hfq")
    assert raw["close"].tolist() == [10.0, 11.0]
    assert hfq["close"].tolist() == [20.0, 21.0]


def test_adjusted_rows_do_not_overwrite_unadjusted_rows(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", [10.0], "none"), "etf")
    store.write_daily(_rows("510300.SH", [99.0], "hfq"), "etf")
    raw = store.read_daily(["510300.SH"], asset_type=AssetType.ETF, adjustment="none")
    assert raw["close"].tolist() == [10.0]


def test_daily_paths_are_segregated_by_asset_and_adjustment(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", [10.0], "none"), AssetType.STOCK)
    store.write_daily(_rows("510300.SH", [20.0], "hfq"), AssetType.ETF)
    assert store.daily_path(AssetType.STOCK, "510300.SH", "none").exists()
    assert store.daily_path(AssetType.ETF, "510300.SH", "hfq").exists()
    assert "adjustment=hfq" in str(store.daily_path(AssetType.ETF, "510300.SH", "hfq"))
    stock = store.read_daily(["510300.SH"], asset_type=AssetType.STOCK)
    etf = store.read_daily(["510300.SH"], asset_type=AssetType.ETF, adjustment="hfq")
    assert stock["close"].tolist() == [10.0]
    assert etf["close"].tolist() == [20.0]


def test_legacy_unpartitioned_daily_file_is_not_reused(app_config):
    store = DataStore(app_config)
    legacy = store.root / "daily" / "etf" / "510300_SH.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    _rows("510300.SH", [99.0]).to_parquet(legacy, index=False)
    assert store.read_daily(["510300.SH"], asset_type=AssetType.ETF, adjustment="none").empty


def test_legacy_snapshot_marker_is_not_reused(app_config):
    store = DataStore(app_config)
    legacy = store.root / "snapshots" / "stock" / "2024-01-02.complete"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.touch()
    assert not store.market_snapshot_complete(AssetType.STOCK, "2024-01-02")
    store.mark_market_snapshot(AssetType.STOCK, "2024-01-02", "none")
    assert store.market_snapshot_complete(AssetType.STOCK, "2024-01-02", "none")


def test_tushare_rejects_adjusted_bar_requests():
    provider = TushareProvider()
    request = DataRequest(
        dataset=Dataset.BARS,
        symbols=("510300.SH",),
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        adjustment="hfq",
    )
    assert not provider.supports(request)
    with pytest.raises(PermanentProviderError):
        provider.fetch(request)


def test_data_request_rejects_unknown_adjustment():
    with pytest.raises(ValueError):
        DataRequest(dataset=Dataset.BARS, adjustment="mystery")


def test_baostock_validates_adjustment_echo(monkeypatch):
    class Result:
        error_code = "0"
        error_msg = ""
        fields = [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adjustflag",
        ]

        def __init__(self):
            self._rows = iter(
                [
                    [
                        "2024-01-02",
                        "sh.600000",
                        "10",
                        "11",
                        "9",
                        "10.5",
                        "100",
                        "1000",
                        "3",
                    ]
                ]
            )

        def next(self):
            try:
                self._current = next(self._rows)
                return True
            except StopIteration:
                return False

        def get_row_data(self):
            return self._current

    class Api:
        def query_history_k_data_plus(self, *args, **kwargs):
            return Result()

    provider = BaoStockProvider(interval_seconds=0)
    monkeypatch.setattr(provider, "_api", lambda: Api())
    request = DataRequest(
        dataset=Dataset.BARS,
        symbols=("600000.SH",),
        start=date(2024, 1, 1),
        end=date(2024, 1, 2),
        adjustment="hfq",
    )
    with pytest.raises(PermanentProviderError, match="复权回显"):
        provider.fetch(request)
