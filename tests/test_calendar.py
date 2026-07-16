from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from quant_trade.data.calendar import trading_days
from quant_trade.data.storage import DataStore
from quant_trade.models import DataBatch


class CalendarRouter:
    def __init__(self, duplicate_open_days: bool = False):
        self.requests = []
        self.duplicate_open_days = duplicate_open_days

    def fetch(self, request):
        self.requests.append(request)
        days = list(pd.date_range(request.start, request.end, freq="D"))
        if self.duplicate_open_days:
            days = [day for day in days for _ in range(2)]
        return DataBatch(
            pd.DataFrame(
                {
                    "cal_date": [day.strftime("%Y%m%d") for day in days],
                    "is_open": [1] * len(days),
                }
            ),
            "fake",
            request,
        )


def test_mutable_calendar_cache_expires_and_can_be_forced(app_config):
    store = DataStore(app_config)
    router = CalendarRouter()
    start, end = date.today(), date.today() + timedelta(days=1)

    assert trading_days(router, start, end, store) == [start, end]
    assert trading_days(router, start, end, store) == [start, end]
    assert len(router.requests) == 1

    with store.connect() as con:
        con.execute(
            "UPDATE calendar_coverage SET updated_at = ?",
            [datetime.now() - timedelta(days=2)],
        )
    trading_days(router, start, end, store)
    assert len(router.requests) == 2

    trading_days(router, start, end, store, force_refresh=True)
    assert len(router.requests) == 3


def test_duplicate_open_days_are_deduplicated_before_storage(app_config):
    store = DataStore(app_config)
    router = CalendarRouter(duplicate_open_days=True)
    start, end = date(2024, 1, 2), date(2024, 1, 3)

    assert trading_days(router, start, end, store) == [start, end]
    with store.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM trade_calendar").fetchone()[0] == 2


def test_historical_calendar_does_not_expire(app_config):
    store = DataStore(app_config)
    router = CalendarRouter()
    start, end = date(2024, 1, 2), date(2024, 1, 3)
    trading_days(router, start, end, store)
    with store.connect() as con:
        con.execute(
            "UPDATE calendar_coverage SET updated_at = ?",
            [datetime.now() - timedelta(days=365)],
        )

    assert trading_days(router, start, end, store) == [start, end]
    assert len(router.requests) == 1
