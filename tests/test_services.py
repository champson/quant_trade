from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant_trade.data.base import EmptyDataError
from quant_trade.data.quality import DataQualityError
from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType, DataBatch, Dataset
from quant_trade.services import (
    run_strategy_backtest,
    run_strategy_signal,
    update_bars,
    update_daily_basic,
    update_market_history,
)


def _rows(symbol: str, days) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": day,
                "bar_time": pd.NaT,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100.0,
                "amount": 1000.0,
                "source": "fake",
            }
            for day in days
        ]
    )


def test_market_snapshot_member_fingerprints_migrate_from_legacy_schema(app_config):
    app_config.ensure_directories()
    with duckdb.connect(str(app_config.paths.database)) as con:
        con.execute(
            """
            CREATE TABLE market_snapshot_members (
                asset_type VARCHAR, adjustment VARCHAR, trade_date DATE, symbol VARCHAR,
                PRIMARY KEY (asset_type, adjustment, trade_date, symbol)
            )
            """
        )

    store = DataStore(app_config)
    with store.connect() as con:
        columns = {
            row[1] for row in con.execute("PRAGMA table_info('market_snapshot_members')").fetchall()
        }
    assert {"file_size", "file_mtime_ns"} <= columns


class RecordingRouter:
    def __init__(self, omitted_days=()):
        self.requests = []
        self.calendar_requests = []
        self.omitted_days = {pd.Timestamp(day) for day in omitted_days}

    def fetch(self, request):
        if request.dataset == Dataset.TRADE_CALENDAR:
            self.calendar_requests.append(request)
            calendar_days = pd.date_range(request.start, request.end, freq="D")
            return DataBatch(
                pd.DataFrame(
                    {
                        "cal_date": calendar_days.strftime("%Y%m%d"),
                        "is_open": (calendar_days.weekday < 5).astype(int),
                    }
                ),
                "fake",
                request,
            )
        days = pd.bdate_range(request.start, request.end)
        self.requests.append(request)
        days = days[~days.isin(self.omitted_days)]
        data = _rows(request.symbols[0], days)
        data["adjustment"] = str(request.adjustment)
        return DataBatch(data, "fake", request)


def test_update_bars_refetches_missing_history(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("000001.SZ", pd.bdate_range("2024-06-03", "2024-06-14")), "stock")
    router = RecordingRouter()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 6, 14),
        AssetType.STOCK,
    )
    # A recent-only cache must not hide the missing months before it.
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 1, 2)


def test_update_bars_skips_when_range_covered(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("000001.SZ", pd.bdate_range("2024-01-02", "2024-06-14")), "stock")
    router = RecordingRouter()
    frame = update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 6, 14),
        AssetType.STOCK,
    )
    assert router.requests == []
    assert frame.empty
    assert len(router.calendar_requests) == 1
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 6, 14),
        AssetType.STOCK,
    )
    # A durable calendar cache keeps a fully cached update offline.
    assert len(router.calendar_requests) == 1


def test_update_bars_cache_is_adjustment_scoped(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("510300.SH", pd.bdate_range("2024-01-02", "2024-06-14")), "etf")
    router = RecordingRouter()
    update_bars(
        app_config,
        router,
        store,
        ["510300.SH"],
        date(2024, 1, 2),
        date(2024, 6, 14),
        AssetType.ETF,
        adjustment="hfq",
    )
    # Unadjusted cache must not satisfy an hfq request.
    assert len(router.requests) == 1
    assert router.requests[0].adjustment == "hfq"


def test_qfq_refetches_the_full_window_to_avoid_mixed_anchor_scales(app_config):
    days = pd.bdate_range("2024-01-02", "2024-01-10")
    cached = _rows("000001.SZ", days)
    cached["adjustment"] = "qfq"
    store = DataStore(app_config)
    store.write_daily(cached, AssetType.STOCK)
    router = RecordingRouter()

    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
        adjustment="qfq",
    )

    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 1, 2)
    assert router.requests[0].end == date(2024, 1, 10)


def test_qfq_refresh_expands_to_all_cached_dates_and_replaces_scale(app_config):
    cached = _rows("000001.SZ", pd.bdate_range("2024-01-02", "2024-01-05"))
    cached["adjustment"] = "qfq"
    store = DataStore(app_config)
    store.write_daily(cached, AssetType.STOCK)

    class Router(RecordingRouter):
        def fetch(self, request):
            batch = super().fetch(request)
            if request.dataset == Dataset.BARS:
                batch.data["close"] = 20.0
                batch.data["high"] = 21.0
            return batch

    router = Router()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 8),
        date(2024, 1, 10),
        AssetType.STOCK,
        adjustment="qfq",
    )
    assert router.requests[0].start == date(2024, 1, 2)
    refreshed = store.read_daily(
        ["000001.SZ"], None, None, asset_type=AssetType.STOCK, adjustment="qfq"
    )
    assert set(refreshed["close"]) == {20.0}


def test_partial_qfq_refresh_preserves_original_cache(app_config):
    days = pd.bdate_range("2024-01-02", "2024-01-10")
    cached = _rows("000001.SZ", days)
    cached["adjustment"] = "qfq"
    store = DataStore(app_config)
    store.write_daily(cached, AssetType.STOCK)
    router = RecordingRouter(omitted_days=["2024-01-05"])

    with pytest.raises(DataQualityError, match="保留原缓存"):
        update_bars(
            app_config,
            router,
            store,
            ["000001.SZ"],
            date(2024, 1, 8),
            date(2024, 1, 10),
            AssetType.STOCK,
            adjustment="qfq",
        )
    after = store.read_daily(
        ["000001.SZ"], None, None, asset_type=AssetType.STOCK, adjustment="qfq"
    )
    assert list(after["trade_date"]) == list(days)
    assert set(after["close"]) == {10.5}


def test_market_history_batches_daily_file_merges(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {"stock": 1}
    app_config.providers.market_history_batch_days = 2
    days = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]

    class Router:
        def fetch(self, request):
            data = _rows("000001.SZ", [pd.Timestamp(request.end)])
            data["adjustment"] = "none"
            return DataBatch(data, "market", request)

    store = DataStore(app_config)
    writes = 0
    original = store.write_daily

    def counted_write(frame, asset_type):
        nonlocal writes
        writes += 1
        return original(frame, asset_type)

    monkeypatch.setattr(store, "write_daily", counted_write)
    assert update_market_history(app_config, Router(), store, days, include_basic=False) == 3
    assert writes == 2
    assert store.incomplete_market_snapshot_dates(AssetType.STOCK, days) == []


def test_update_bars_extends_tail_incrementally(app_config):
    store = DataStore(app_config)
    store.write_daily(_rows("000001.SZ", pd.bdate_range("2024-01-02", "2024-06-07")), "stock")
    router = RecordingRouter()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 6, 14),
        AssetType.STOCK,
    )
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 6, 10)


def test_update_bars_fetches_internal_calendar_gap_only(app_config):
    store = DataStore(app_config)
    days = pd.bdate_range("2024-01-02", "2024-01-15")
    cached = days[~days.isin(pd.to_datetime(["2024-01-08", "2024-01-09"]))]
    store.write_daily(_rows("000001.SZ", cached), AssetType.STOCK)
    router = RecordingRouter()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 15),
        AssetType.STOCK,
    )
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 1, 8)
    assert router.requests[0].end == date(2024, 1, 9)


def test_today_without_data_is_not_marked_covered(app_config, monkeypatch):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2024, 1, 10)

    monkeypatch.setattr("quant_trade.services.date", FakeDate)
    store = DataStore(app_config)
    # The provider has not published 2024-01-10 ("today") yet.
    router = RecordingRouter(omitted_days=["2024-01-10"])
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
    )
    assert len(router.requests) == 1
    router.requests.clear()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
    )
    # Unlike a suspended day in the past, today stays uncovered and is retried.
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 1, 10)
    assert router.requests[0].end == date(2024, 1, 10)


def test_future_end_is_clamped_without_future_empty_requests(app_config, monkeypatch):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2024, 1, 10)

    monkeypatch.setattr("quant_trade.services.date", FakeDate)
    store = DataStore(app_config)
    router = RecordingRouter()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 15),
        AssetType.STOCK,
    )
    assert router.calendar_requests[0].end == date(2024, 1, 10)
    assert router.requests[0].end == date(2024, 1, 10)

    router.calendar_requests.clear()
    router.requests.clear()
    result = update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 11),
        date(2024, 1, 15),
        AssetType.STOCK,
    )
    assert result.empty
    assert router.calendar_requests == []
    assert router.requests == []


def test_unpublished_today_empty_error_is_retried(app_config, monkeypatch):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2024, 1, 10)

    monkeypatch.setattr("quant_trade.services.date", FakeDate)

    class EmptyTodayRouter(RecordingRouter):
        def fetch(self, request):
            if request.dataset == Dataset.BARS and pd.Timestamp(request.start) == pd.Timestamp(
                "2024-01-10"
            ):
                self.requests.append(request)
                raise EmptyDataError("今日行情尚未发布")
            return super().fetch(request)

    store = DataStore(app_config)
    router = EmptyTodayRouter(omitted_days=["2024-01-10"])
    for _ in range(3):
        update_bars(
            app_config,
            router,
            store,
            ["000001.SZ"],
            date(2024, 1, 2),
            date(2024, 1, 10),
            AssetType.STOCK,
        )
    # The empty answer for today is never marked covered, so every later run
    # queries today again until data appears.
    assert [request.start for request in router.requests] == [
        date(2024, 1, 2),
        date(2024, 1, 10),
        date(2024, 1, 10),
    ]


def test_partial_success_does_not_hide_omitted_day(app_config):
    store = DataStore(app_config)
    router = RecordingRouter(omitted_days=["2024-01-08"])
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
    )
    assert len(router.requests) == 1
    router.requests.clear()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
    )
    assert len(router.requests) == 1
    assert router.requests[0].start == date(2024, 1, 8)
    assert router.requests[0].end == date(2024, 1, 8)


def test_empty_suspension_gap_is_marked_covered(app_config):
    store = DataStore(app_config)
    days = pd.bdate_range("2024-01-02", "2024-01-10")
    missing = pd.Timestamp("2024-01-08")
    store.write_daily(_rows("000001.SZ", days[days != missing]), AssetType.STOCK)

    class EmptyGapRouter(RecordingRouter):
        def fetch(self, request):
            if request.dataset == Dataset.BARS:
                self.requests.append(request)
                raise EmptyDataError("停牌日无行情")
            return super().fetch(request)

    router = EmptyGapRouter()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
    )
    assert len(router.requests) == 1
    router.requests.clear()
    update_bars(
        app_config,
        router,
        store,
        ["000001.SZ"],
        date(2024, 1, 2),
        date(2024, 1, 10),
        AssetType.STOCK,
    )
    assert router.requests == []


def test_confirmed_empty_range_is_cached_without_parquet(app_config):
    class AlwaysEmptyRouter(RecordingRouter):
        def fetch(self, request):
            if request.dataset == Dataset.BARS:
                self.requests.append(request)
                raise EmptyDataError("代码无行情")
            return super().fetch(request)

    store = DataStore(app_config)
    router = AlwaysEmptyRouter()
    for _ in range(2):
        update_bars(
            app_config,
            router,
            store,
            ["999999.SZ"],
            date(2024, 1, 2),
            date(2024, 1, 10),
            AssetType.STOCK,
        )
    assert len(router.requests) == 1
    assert not store.daily_path(AssetType.STOCK, "999999.SZ").exists()


def test_partial_market_snapshot_is_retried_until_symbol_threshold_is_met(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 3}

    class SnapshotRouter:
        def __init__(self):
            self.calls = 0

        def fetch(self, request):
            self.calls += 1
            symbols = ["000001.SZ", "000002.SZ"]
            if self.calls > 1:
                symbols.append("000003.SZ")
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(request.end)]) for symbol in symbols],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    router = SnapshotRouter()
    with pytest.raises(DataQualityError, match="快照不完整"):
        update_bars(
            app_config,
            router,
            store,
            [],
            date(2024, 1, 8),
            date(2024, 1, 8),
            AssetType.STOCK,
        )
    for _ in range(2):
        update_bars(
            app_config,
            router,
            store,
            [],
            date(2024, 1, 8),
            date(2024, 1, 8),
            AssetType.STOCK,
        )
    assert router.calls == 2
    assert store.market_snapshot_complete(AssetType.STOCK, date(2024, 1, 8))
    with store.connect() as con:
        state = con.execute(
            "SELECT symbol_count, expected_symbols, status FROM market_snapshots"
        ).fetchone()
    assert state == (3, 3, "complete")


def test_force_partial_refresh_preserves_existing_complete_snapshot(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2}
    day = date(2024, 1, 8)

    class SnapshotRouter:
        def __init__(self):
            self.symbols = ["000001.SZ", "000002.SZ"]

        def fetch(self, request):
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(day)]) for symbol in self.symbols],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    router = SnapshotRouter()
    update_bars(app_config, router, store, [], day, day, AssetType.STOCK)
    router.symbols = ["000001.SZ"]

    with pytest.raises(DataQualityError, match="保留原有完整快照标记"):
        update_bars(
            app_config,
            router,
            store,
            [],
            day,
            day,
            AssetType.STOCK,
            resume=False,
        )

    assert store.market_snapshot_complete(AssetType.STOCK, day)
    assert store.market_snapshot_symbols(AssetType.STOCK, day) == {
        "000001.SZ",
        "000002.SZ",
    }


def test_snapshot_marking_deep_checks_only_configured_sample(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {"stock": 3}
    app_config.providers.market_snapshot_validation_sample_size = 1
    day = date(2024, 1, 8)

    class SnapshotRouter:
        def fetch(self, request):
            data = pd.concat(
                [
                    _rows(symbol, [pd.Timestamp(day)])
                    for symbol in ("000001.SZ", "000002.SZ", "000003.SZ")
                ],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    samples = []

    def record_sample(asset_type, adjustment, trade_date, symbols):
        samples.append(symbols)
        return True

    monkeypatch.setattr(store, "_deep_market_snapshot_valid", record_sample)
    update_bars(app_config, SnapshotRouter(), store, [], day, day, AssetType.STOCK)
    assert len(samples) == 1
    assert len(samples[0]) == 1


def test_deep_snapshot_audit_detects_content_loss_despite_matching_fingerprint(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 1}
    day = date(2024, 1, 8)

    class SnapshotRouter:
        def fetch(self, request):
            data = _rows("000001.SZ", [pd.Timestamp(day)])
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    update_bars(app_config, SnapshotRouter(), store, [], day, day, AssetType.STOCK)
    path = store.daily_path(AssetType.STOCK, "000001.SZ")
    _rows("000001.SZ", [pd.Timestamp("2024-01-09")]).to_parquet(path, index=False)
    stat = path.stat()
    with store.connect() as con:
        con.execute(
            """
            UPDATE market_snapshot_members SET file_size = ?, file_mtime_ns = ?
            WHERE asset_type = 'stock' AND adjustment = 'none' AND trade_date = ?
            """,
            [stat.st_size, stat.st_mtime_ns, day],
        )

    fresh = DataStore(app_config)
    assert fresh.market_snapshot_complete(AssetType.STOCK, day)
    result = fresh.audit_market_snapshots(AssetType.STOCK)
    assert result == [{"trade_date": "2024-01-08", "symbol_count": 1, "status": "invalid"}]
    assert not fresh.market_snapshot_complete(AssetType.STOCK, day)


def test_deep_snapshot_audit_rejects_invalid_ohlcv(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 1}
    day = date(2024, 1, 8)

    class SnapshotRouter:
        def fetch(self, request):
            data = _rows("000001.SZ", [pd.Timestamp(day)])
            data["adjustment"] = "none"
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    update_bars(app_config, SnapshotRouter(), store, [], day, day, AssetType.STOCK)
    path = store.daily_path(AssetType.STOCK, "000001.SZ")
    pd.read_parquet(path).assign(close=-1.0).to_parquet(path, index=False)

    result = store.audit_market_snapshots(AssetType.STOCK)
    assert result[0]["status"] == "invalid"
    assert not store.market_snapshot_complete(AssetType.STOCK, day)


def test_daily_basic_audit_rejects_invalid_market_value(app_config):
    day = date(2024, 1, 8)
    store = DataStore(app_config)
    frame = pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": [day], "total_mv": [1.0]})
    store.write_daily_basic(frame)
    store.mark_daily_basic_snapshot(
        day,
        row_count=1,
        symbol_count=1,
        expected_symbols=1,
        provider="test",
        status="complete",
        details={"symbol_digest": store.symbol_digest({"000001.SZ"})},
    )
    path = store.daily_basic_path(str(day))
    pd.read_parquet(path).assign(total_mv=-1.0).to_parquet(path, index=False)

    assert not store.daily_basic_complete(day)
    assert store.audit_daily_basic_snapshots() == [
        {"trade_date": "2024-01-08", "status": "invalid"}
    ]
    with store.connect() as con:
        assert (
            con.execute(
                "SELECT status FROM daily_basic_snapshots WHERE trade_date = ?", [day]
            ).fetchone()[0]
            == "incomplete"
        )


def test_complete_snapshot_is_retried_when_a_member_file_is_deleted(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2}

    class SnapshotRouter:
        def __init__(self):
            self.calls = 0

        def fetch(self, request):
            self.calls += 1
            data = pd.concat(
                [
                    _rows("000001.SZ", [pd.Timestamp(request.end)]),
                    _rows("000002.SZ", [pd.Timestamp(request.end)]),
                ],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    router = SnapshotRouter()
    day = date(2024, 1, 8)
    update_bars(app_config, router, store, [], day, day, AssetType.STOCK)
    store.daily_path(AssetType.STOCK, "000001.SZ").unlink()
    assert not store.market_snapshot_complete(AssetType.STOCK, day)

    update_bars(app_config, router, store, [], day, day, AssetType.STOCK)
    assert router.calls == 2
    assert store.daily_path(AssetType.STOCK, "000001.SZ").exists()


def test_unverified_same_day_basic_cannot_lower_historical_snapshot_threshold(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 4500}
    day = date(2005, 1, 4)
    store = DataStore(app_config)
    store.write_daily_basic(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000002.SZ"],
                "trade_date": [day, day],
                "total_mv": [100.0, 200.0],
            }
        )
    )

    class HistoricalRouter:
        def fetch(self, request):
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(day)]) for symbol in ("000001.SZ", "000002.SZ")],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "history", request)

    with pytest.raises(DataQualityError, match="快照不完整"):
        update_bars(app_config, HistoricalRouter(), store, [], day, day, AssetType.STOCK)
    assert not store.market_snapshot_complete(AssetType.STOCK, day)


def test_historical_snapshot_without_basic_uses_provider_security_master(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 4500}
    app_config.providers.market_snapshot_reference_ratio = 0.9
    day = date(2020, 1, 2)

    class HistoricalRouter:
        def fetch(self, request):
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(day)]) for symbol in ("000001.SZ", "000002.SZ")],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(
                data,
                "history",
                request,
                metadata={
                    "expected_symbols": 3,
                    "expected_symbols_source": "security_master",
                },
            )

    store = DataStore(app_config)
    update_bars(app_config, HistoricalRouter(), store, [], day, day, AssetType.STOCK)
    assert store.market_snapshot_complete(AssetType.STOCK, day)
    with store.connect() as con:
        expected, details = con.execute(
            "SELECT expected_symbols, details FROM market_snapshots"
        ).fetchone()
    assert expected == 2
    assert "security_master" in details


def test_cached_market_snapshot_promotes_new_daily_basic_validation(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2}
    day = date(2024, 1, 8)

    class Router:
        def fetch(self, request):
            if request.dataset == Dataset.DAILY_BASIC:
                return DataBatch(
                    pd.DataFrame(
                        {
                            "ts_code": ["000001.SZ", "000002.SZ"],
                            "trade_date": ["20240108", "20240108"],
                            "total_mv": [1.0, 2.0],
                        }
                    ),
                    "basic",
                    request,
                    metadata={
                        "expected_symbols": 2,
                        "expected_symbols_source": "security_master",
                    },
                )
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(day)]) for symbol in ("000001.SZ", "000002.SZ")],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    router = Router()
    update_bars(app_config, router, store, [], day, day, AssetType.STOCK)
    update_daily_basic(router, store, day)
    assert store.daily_basic_complete(day)


def test_daily_basic_same_count_but_wrong_member_is_incomplete(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2}
    day = date(2024, 1, 8)

    class SnapshotRouter:
        def fetch(self, request):
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(day)]) for symbol in ("000001.SZ", "000002.SZ")],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    store.write_daily_basic(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000003.SZ"],
                "trade_date": [day, day],
                "total_mv": [1.0, 3.0],
            }
        )
    )
    store.mark_daily_basic_snapshot(
        day,
        row_count=2,
        symbol_count=2,
        expected_symbols=0,
        provider="basic",
        status="observed",
    )
    update_bars(app_config, SnapshotRouter(), store, [], day, day, AssetType.STOCK)
    assert not store.daily_basic_complete(day)


def test_snapshot_marker_detects_member_file_without_requested_day(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 1}
    day = date(2024, 1, 8)

    class SnapshotRouter:
        def fetch(self, request):
            data = _rows("000001.SZ", [pd.Timestamp(day)])
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    update_bars(app_config, SnapshotRouter(), store, [], day, day, AssetType.STOCK)
    path = store.daily_path(AssetType.STOCK, "000001.SZ")
    replacement = _rows("000001.SZ", [pd.Timestamp("2024-01-09")])
    replacement.to_parquet(path, index=False)
    assert not store.market_snapshot_complete(AssetType.STOCK, day)


def test_partial_daily_basic_is_not_marked_complete(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 3}
    day = date(2024, 1, 8)

    class BasicRouter:
        def fetch(self, request):
            return DataBatch(
                pd.DataFrame(
                    {"ts_code": ["000001.SZ"], "trade_date": ["20240108"], "total_mv": [1.0]}
                ),
                "basic",
                request,
            )

    class SnapshotRouter:
        def fetch(self, request):
            data = pd.concat(
                [
                    _rows(symbol, [pd.Timestamp(day)])
                    for symbol in ("000001.SZ", "000002.SZ", "000003.SZ")
                ],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    store.write_daily_basic(
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "trade_date": [day, day, day],
                "total_mv": [1.0, 2.0, 3.0],
            }
        )
    )
    update_daily_basic(BasicRouter(), store, day)
    assert store.daily_basic_symbol_count(day) == 1
    update_bars(app_config, SnapshotRouter(), store, [], day, day, AssetType.STOCK)
    assert not store.daily_basic_complete(day)
    with store.connect() as con:
        status, expected = con.execute(
            "SELECT status, expected_symbols FROM daily_basic_snapshots"
        ).fetchone()
    assert status == "observed"
    assert expected == 0
    app_config.strategies = {"microcap": {"enabled": True, "asset_type": "stock"}}
    store.write_trade_calendar([day], [day], "test")
    with pytest.raises(DataQualityError, match="daily_basic 缺失或不完整"):
        run_strategy_signal(app_config, store, "microcap", str(day))


def test_correlated_partial_basic_and_market_cannot_bless_each_other(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 3}
    day = date(2024, 1, 8)

    class Router:
        def fetch(self, request):
            if request.dataset == Dataset.DAILY_BASIC:
                return DataBatch(
                    pd.DataFrame(
                        {
                            "ts_code": ["000001.SZ", "000002.SZ"],
                            "trade_date": ["20240108", "20240108"],
                            "total_mv": [1.0, 2.0],
                        }
                    ),
                    "partial-basic",
                    request,
                )
            data = pd.concat(
                [_rows(symbol, [pd.Timestamp(day)]) for symbol in ("000001.SZ", "000002.SZ")],
                ignore_index=True,
            )
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "partial-market", request)

    store = DataStore(app_config)
    router = Router()
    update_daily_basic(router, store, day)
    assert not store.daily_basic_complete(day)
    with pytest.raises(DataQualityError, match="快照不完整"):
        update_bars(app_config, router, store, [], day, day, AssetType.STOCK)
    assert not store.market_snapshot_complete(AssetType.STOCK, day)


@pytest.mark.parametrize("invalid_value", [0, -1, float("inf"), float("-inf"), "bad"])
def test_daily_basic_rejects_invalid_market_value(app_config, invalid_value):
    day = date(2024, 1, 8)

    class Router:
        def fetch(self, request):
            return DataBatch(
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "trade_date": ["20240108"],
                        "total_mv": [invalid_value],
                    }
                ),
                "basic",
                request,
                metadata={"expected_symbols": 1},
            )

    with pytest.raises(DataQualityError, match="无法解析"):
        update_daily_basic(Router(), DataStore(app_config), day)


def test_daily_basic_rejects_blank_symbol(app_config):
    day = date(2024, 1, 8)

    class Router:
        def fetch(self, request):
            return DataBatch(
                pd.DataFrame(
                    {
                        "ts_code": ["   "],
                        "trade_date": ["20240108"],
                        "total_mv": [1.0],
                    }
                ),
                "basic",
                request,
                metadata={"expected_symbols": 1},
            )

    with pytest.raises(DataQualityError, match="无法解析"):
        update_daily_basic(Router(), DataStore(app_config), day)


def test_incomplete_force_refresh_preserves_complete_daily_basic(app_config):
    day = date(2024, 1, 8)

    class Router:
        def __init__(self, expected):
            self.expected = expected

        def fetch(self, request):
            return DataBatch(
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "trade_date": ["20240108", "20240108"],
                        "total_mv": [1.0, 2.0],
                    }
                ),
                "basic",
                request,
                metadata={"expected_symbols": self.expected},
            )

    store = DataStore(app_config)
    update_daily_basic(Router(2), store, day)
    assert store.daily_basic_complete(day)

    with pytest.raises(DataQualityError, match="保留旧版本"):
        update_daily_basic(Router(0), store, day)
    assert store.daily_basic_complete(day)
    assert store.daily_basic_symbols(day) == {"000001.SZ", "000002.SZ"}


def test_market_snapshot_hot_check_uses_fingerprints_not_parquet_scan(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {"stock": 1}
    day = date(2024, 1, 8)

    class Router:
        def fetch(self, request):
            data = _rows("000001.SZ", [pd.Timestamp(day)])
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    update_bars(app_config, Router(), store, [], day, day, AssetType.STOCK)

    def unexpected_deep_scan(*args, **kwargs):
        raise AssertionError("hot path must not reopen parquet contents")

    monkeypatch.setattr(store, "_deep_market_snapshot_valid", unexpected_deep_scan)
    assert store.market_snapshot_complete(AssetType.STOCK, day)
    store.write_daily(_rows("000001.SZ", [pd.Timestamp("2024-01-09")]), AssetType.STOCK)
    assert store.market_snapshot_complete(AssetType.STOCK, day)


def test_market_snapshot_cache_expires_and_detects_external_file_change(app_config):
    app_config.providers.market_snapshot_min_symbols = {"stock": 1}
    app_config.providers.market_snapshot_cache_ttl_seconds = 0
    day = date(2024, 1, 8)

    class Router:
        def fetch(self, request):
            data = _rows("000001.SZ", [pd.Timestamp(day)])
            data["adjustment"] = str(request.adjustment)
            return DataBatch(data, "snapshot", request)

    store = DataStore(app_config)
    update_bars(app_config, Router(), store, [], day, day, AssetType.STOCK)
    path = store.daily_path(AssetType.STOCK, "000001.SZ")
    pd.read_parquet(path).assign(close=999.0).to_parquet(path, index=False)

    assert not store.market_snapshot_complete(AssetType.STOCK, day)


def test_signal_refuses_to_publish_stale_latest_bar(app_config):
    app_config.strategies = {
        "logbias": {
            "enabled": True,
            "asset_type": "etf",
            "symbols": ["510300.SH"],
            "ema_window": 2,
        }
    }
    store = DataStore(app_config)
    bars = _rows("510300.SH", [pd.Timestamp("2024-01-09")])
    store.write_daily(bars, AssetType.ETF)
    with pytest.raises(DataQualityError, match="不会输出过期信号"):
        run_strategy_signal(app_config, store, "logbias", "2024-01-10")


def test_backtest_runs_are_versioned_instead_of_overwritten(app_config):
    app_config.strategies = {
        "logbias": {
            "enabled": True,
            "asset_type": "etf",
            "symbols": ["510300.SH"],
            "ema_window": 2,
            "entry": 1.0,
            "stop": -1.0,
            "overheat": 2.0,
        }
    }
    store = DataStore(app_config)
    bars = _rows("510300.SH", pd.bdate_range("2024-01-02", "2024-01-12"))
    store.write_daily(bars, AssetType.ETF)

    first = run_strategy_backtest(app_config, store, "logbias", "2024-01-02", "2024-01-12")
    second = run_strategy_backtest(app_config, store, "logbias", "2024-01-02", "2024-01-12")

    first_report = Path(first.artifacts["markdown"])
    second_report = Path(second.artifacts["markdown"])
    assert first_report.parent != second_report.parent
    assert first_report.exists()
    assert second_report.exists()


def test_microcap_rejects_incomplete_market_snapshot_even_when_basic_is_complete(app_config):
    day = date(2024, 1, 8)
    app_config.strategies = {
        "microcap": {"enabled": True, "asset_type": "stock", "rebalance": "daily"}
    }
    store = DataStore(app_config)
    store.write_trade_calendar([day], [day], "test")
    store.write_daily(_rows("000001.SZ", [pd.Timestamp(day)]), AssetType.STOCK)
    basic = pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": [day], "total_mv": [1.0]})
    store.write_daily_basic(basic)
    store.mark_daily_basic_snapshot(
        day,
        row_count=1,
        symbol_count=1,
        expected_symbols=1,
        provider="test",
        status="complete",
        details={"symbol_digest": store.symbol_digest({"000001.SZ"})},
    )

    with pytest.raises(DataQualityError, match="全市场快照缺失或不完整"):
        run_strategy_signal(app_config, store, "microcap", str(day))


def test_microcap_uses_calendar_to_detect_wholly_missing_market_day(app_config):
    first, missing_day = date(2024, 1, 8), date(2024, 1, 9)
    app_config.strategies = {
        "microcap": {"enabled": True, "asset_type": "stock", "rebalance": "daily"}
    }
    store = DataStore(app_config)
    store.write_trade_calendar([first, missing_day], [first, missing_day], "test")
    store.write_daily(_rows("000001.SZ", [pd.Timestamp(first)]), AssetType.STOCK)
    assert store.mark_market_snapshot(
        AssetType.STOCK,
        first,
        row_count=1,
        symbol_count=1,
        expected_symbols=1,
        symbols=["000001.SZ"],
    )

    with pytest.raises(DataQualityError, match="2024-01-09"):
        run_strategy_signal(app_config, store, "microcap", str(missing_day))
