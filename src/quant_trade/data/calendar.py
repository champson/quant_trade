from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import DataRequest, Dataset


def _calendar_days(frame: pd.DataFrame) -> list[date]:
    if "cal_date" in frame:
        values = frame["cal_date"]
    elif "calendar_date" in frame:
        values = frame["calendar_date"]
    else:
        raise ValueError("交易日历字段无法识别")
    return sorted(pd.to_datetime(values).dt.date.unique().tolist())


def _open_days(frame: pd.DataFrame) -> list[date]:
    if "cal_date" in frame:
        mask = pd.to_numeric(frame.get("is_open", 0), errors="coerce").fillna(0).astype(int) == 1
        values = frame.loc[mask, "cal_date"]
    elif "calendar_date" in frame:
        mask = frame.get("is_trading_day", "0").astype(str) == "1"
        values = frame.loc[mask, "calendar_date"]
    else:
        raise ValueError("交易日历字段无法识别")
    return sorted(set(pd.to_datetime(values).dt.date.tolist()))


def _missing_calendar_ranges(
    start: date, end: date, covered: set[date]
) -> list[tuple[date, date, list[date]]]:
    expected = list(pd.date_range(start, end, freq="D").date)
    missing = [index for index, day in enumerate(expected) if day not in covered]
    if not missing:
        return []
    groups: list[list[int]] = [[missing[0]]]
    for index in missing[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [
        (expected[group[0]], expected[group[-1]], [expected[index] for index in group])
        for group in groups
    ]


def trading_days(
    router: DataRouter,
    start: date,
    end: date,
    store: DataStore | None = None,
    *,
    force_refresh: bool = False,
) -> list[date]:
    if store is None:
        batch = router.fetch(DataRequest(dataset=Dataset.TRADE_CALENDAR, start=start, end=end))
        return _open_days(batch.data.copy())

    providers_config = getattr(getattr(router, "config", None), "providers", None)
    ttl_hours = float(getattr(providers_config, "calendar_mutable_ttl_hours", 24.0))
    covered = (
        set()
        if force_refresh
        else store.covered_calendar_dates(
            start,
            end,
            mutable_from=date.today(),
            mutable_ttl=timedelta(hours=ttl_hours),
        )
    )
    for fetch_start, fetch_end, calendar_days in _missing_calendar_ranges(start, end, covered):
        batch = router.fetch(
            DataRequest(dataset=Dataset.TRADE_CALENDAR, start=fetch_start, end=fetch_end)
        )
        returned_days = [
            day for day in _calendar_days(batch.data.copy()) if fetch_start <= day <= fetch_end
        ]
        if set(returned_days) != set(calendar_days):
            missing = sorted(set(calendar_days) - set(returned_days))
            raise ValueError(
                f"交易日历返回不完整，缺少 {len(missing)} 天，首日 {missing[0] if missing else '-'}"
            )
        open_days = [
            day for day in _open_days(batch.data.copy()) if fetch_start <= day <= fetch_end
        ]
        store.write_trade_calendar(open_days, returned_days, batch.provider)
    return store.read_trading_days(start, end)
