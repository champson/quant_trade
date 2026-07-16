from __future__ import annotations

from datetime import date

import pandas as pd

from quant_trade.data.base import EmptyDataError
from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType, DataBatch, Dataset
from quant_trade.services import update_bars


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
