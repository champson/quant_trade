from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.data.base import PermanentProviderError
from quant_trade.data.quality import DataQualityError, validate_bars
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


def test_read_daily_dates_filters_inside_parquet_scan(app_config, monkeypatch):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", [10.0, 11.0, 12.0]), AssetType.ETF)
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("pandas scan"))

    selected = store.read_daily_dates(
        ["510300.SH"], [date(2024, 1, 2), date(2024, 1, 4)], asset_type=AssetType.ETF
    )

    assert selected["close"].tolist() == [10.0, 12.0]


def test_adjusted_rows_do_not_overwrite_unadjusted_rows(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", [10.0], "none"), "etf")
    store.write_daily(_rows("510300.SH", [99.0], "hfq"), "etf")
    raw = store.read_daily(["510300.SH"], asset_type=AssetType.ETF, adjustment="none")
    assert raw["close"].tolist() == [10.0]


def test_failed_daily_atomic_write_preserves_previous_file(app_config, monkeypatch):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", [10.0], "none"), AssetType.ETF)
    path = store.daily_path(AssetType.ETF, "510300.SH")
    original = pd.read_parquet(path)
    real_to_parquet = pd.DataFrame.to_parquet

    def fail_temporary_write(frame, target, *args, **kwargs):
        if str(target).endswith(".tmp"):
            raise OSError("disk full")
        return real_to_parquet(frame, target, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_temporary_write)
    with pytest.raises(OSError, match="disk full"):
        store.write_daily(_rows("510300.SH", [99.0], "none"), AssetType.ETF)
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


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
    assert not store.market_snapshot_complete(AssetType.STOCK, "2024-01-02", "none")


def test_tushare_rejects_qfq_bar_requests_because_chunk_scales_would_differ():
    provider = TushareProvider()
    request = DataRequest(
        dataset=Dataset.BARS,
        symbols=("510300.SH",),
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        asset_type=AssetType.ETF,
        adjustment="qfq",
    )
    assert not provider.supports(request)
    with pytest.raises(PermanentProviderError):
        provider.fetch(request)


def test_data_request_rejects_unknown_adjustment():
    with pytest.raises(ValueError):
        DataRequest(dataset=Dataset.BARS, adjustment="mystery")


@pytest.mark.parametrize("column", ["volume", "amount"])
def test_daily_quality_rejects_negative_trade_values(column):
    frame = _rows("000001.SZ", [10.0])
    frame.loc[0, column] = -1
    with pytest.raises(DataQualityError, match="负成交量或成交额"):
        validate_bars(frame)


def test_daily_quality_rejects_missing_volume():
    frame = _rows("000001.SZ", [10.0])
    frame.loc[0, "volume"] = None
    with pytest.raises(DataQualityError, match="关键字段为空"):
        validate_bars(frame)


@pytest.mark.parametrize(
    ("column", "value"),
    [("close", float("inf")), ("volume", float("inf")), ("amount", "not-a-number")],
)
def test_daily_quality_rejects_non_finite_or_malformed_numeric_values(column, value):
    frame = _rows("000001.SZ", [10.0])
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataQualityError):
        validate_bars(frame)


def test_minute_quality_requires_bar_time():
    frame = _rows("000001.SZ", [10.0]).drop(columns="bar_time")
    with pytest.raises(DataQualityError, match="bar_time"):
        validate_bars(frame, minute=True)


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
