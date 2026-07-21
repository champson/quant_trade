from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.config import RetryConfig
from quant_trade.data.base import (
    DataProvider,
    EmptyDataError,
    PermanentProviderError,
    ProviderError,
)
from quant_trade.data.retry import retry_call
from quant_trade.data.router import DataRouter
from quant_trade.models import DataBatch, DataRequest, Dataset


def test_retry_transient_then_success():
    calls = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("timed out")
        return 42

    sleeps = []
    result = retry_call(
        operation,
        RetryConfig(attempts=4, delays=[0, 0, 0, 0]),
        sleep=sleeps.append,
    )
    assert result == 42
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_permanent_error_is_not_retried():
    calls = []

    def operation():
        calls.append(1)
        raise PermanentProviderError("无权限")

    with pytest.raises(PermanentProviderError):
        retry_call(operation, RetryConfig(attempts=6, delays=[0] * 6), sleep=lambda _: None)
    assert len(calls) == 1


def test_empty_data_is_not_retried():
    calls = []

    def operation():
        calls.append(1)
        raise EmptyDataError("返回空行情")

    with pytest.raises(EmptyDataError):
        retry_call(operation, RetryConfig(attempts=6, delays=[0] * 6), sleep=lambda _: None)
    assert len(calls) == 1


class FakeProvider(DataProvider):
    def __init__(self, name: str, fail: bool):
        self.name, self.fail = name, fail

    def capabilities(self):
        return {Dataset.BARS}

    def fetch(self, request):
        if self.fail:
            raise PermanentProviderError("permission denied")
        data = pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2024-01-02")],
                "bar_time": [pd.NaT],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "volume": [100.0],
                "amount": [1000.0],
                "source": [self.name],
                "adjustment": [str(request.adjustment)],
            }
        )
        return DataBatch(data, self.name, request)


class EmptyProvider(DataProvider):
    def __init__(self, name: str):
        self.name = name

    def capabilities(self):
        return {Dataset.BARS}

    def fetch(self, request):
        raise EmptyDataError("返回空行情")


class WrongAdjustmentProvider(FakeProvider):
    def fetch(self, request):
        batch = super().fetch(request)
        batch.data["adjustment"] = "none"
        return batch


class UnadjustedOnlyProvider(FakeProvider):
    def supports(self, request):
        return str(request.adjustment) == "none" and super().supports(request)


def test_router_falls_back_on_empty_without_tripping_circuit(app_config):
    app_config.providers.priority = ["first", "second"]
    app_config.providers.retry.circuit_failures = 1
    router = DataRouter(
        app_config,
        {
            "first": EmptyProvider("first"),
            "second": FakeProvider("second", False),
        },
    )
    batch = router.fetch(
        DataRequest(
            dataset=Dataset.BARS,
            symbols=("000001.SZ",),
            start=date(2024, 1, 1),
            end=date(2024, 1, 2),
        )
    )
    assert batch.provider == "second"
    # A single failure trips the breaker at threshold 1, so "first" staying
    # available proves the empty result was not counted as a failure.
    assert router.circuit.allow("first")


def test_router_reports_all_empty_as_empty_data(app_config):
    app_config.providers.priority = ["first"]
    router = DataRouter(app_config, {"first": EmptyProvider("first")})
    with pytest.raises(EmptyDataError, match="均返回空结果"):
        router.fetch(
            DataRequest(
                dataset=Dataset.BARS,
                symbols=("000001.SZ",),
                start=date(2024, 1, 1),
                end=date(2024, 1, 2),
            )
        )
    assert router.circuit.allow("first")


def test_router_records_explicit_fallback(app_config):
    app_config.providers.priority = ["first", "second"]
    app_config.providers.retry.delays = [0] * 6
    router = DataRouter(
        app_config,
        {
            "first": FakeProvider("first", True),
            "second": FakeProvider("second", False),
        },
    )
    batch = router.fetch(
        DataRequest(
            dataset=Dataset.BARS,
            symbols=("000001.SZ",),
            start=date(2024, 1, 1),
            end=date(2024, 1, 2),
        )
    )
    assert batch.provider == "second"
    assert any("回退" in warning for warning in batch.warnings)


def test_router_rejects_mislabeled_adjustment(app_config):
    app_config.providers.priority = ["wrong"]
    app_config.providers.retry.attempts = 1
    router = DataRouter(app_config, {"wrong": WrongAdjustmentProvider("wrong", False)})
    with pytest.raises(ProviderError, match="返回复权方式"):
        router.fetch(
            DataRequest(
                dataset=Dataset.BARS,
                symbols=("000001.SZ",),
                start=date(2024, 1, 1),
                end=date(2024, 1, 2),
                adjustment="hfq",
            )
        )


def test_router_falls_back_for_adjusted_request(app_config):
    app_config.providers.priority = ["raw", "adjusted"]
    router = DataRouter(
        app_config,
        {
            "raw": UnadjustedOnlyProvider("raw", False),
            "adjusted": FakeProvider("adjusted", False),
        },
    )
    batch = router.fetch(
        DataRequest(
            dataset=Dataset.BARS,
            symbols=("510300.SH",),
            start=date(2024, 1, 1),
            end=date(2024, 1, 2),
            adjustment="hfq",
        )
    )
    assert batch.provider == "adjusted"
    assert any("raw: 不支持" in warning for warning in batch.warnings)


def test_circuit_open_plus_empty_is_not_confirmed_empty(app_config):
    app_config.providers.priority = ["broken", "empty"]
    app_config.providers.retry.circuit_failures = 1
    router = DataRouter(
        app_config,
        {
            "broken": FakeProvider("broken", False),
            "empty": EmptyProvider("empty"),
        },
    )
    router.circuit.failure("broken")
    with pytest.raises(ProviderError, match="所有数据源均失败"):
        router.fetch(
            DataRequest(
                dataset=Dataset.BARS,
                symbols=("000001.SZ",),
                start=date(2024, 1, 1),
                end=date(2024, 1, 2),
            )
        )


def test_partial_trade_calendar_falls_back_to_next_provider(app_config):
    class CalendarProvider(DataProvider):
        def __init__(self, name, dates, date_column="cal_date"):
            self.name = name
            self.dates = dates
            self.date_column = date_column

        def capabilities(self):
            return {Dataset.TRADE_CALENDAR}

        def fetch(self, request):
            return DataBatch(
                pd.DataFrame(
                    {
                        self.date_column: [
                            pd.Timestamp(day).strftime("%Y%m%d") for day in self.dates
                        ],
                        "is_open": [1] * len(self.dates),
                    }
                ),
                self.name,
                request,
            )

    app_config.providers.priority = ["partial", "complete"]
    app_config.providers.retry.attempts = 1
    router = DataRouter(
        app_config,
        {
            "partial": CalendarProvider("partial", ["2024-01-01", "2024-01-03"]),
            "complete": CalendarProvider(
                "complete",
                ["2024-01-01", "2024-01-02", "2024-01-03"],
                "calendar_date",
            ),
        },
    )
    batch = router.fetch(
        DataRequest(
            dataset=Dataset.TRADE_CALENDAR,
            start=date(2024, 1, 1),
            end=date(2024, 1, 3),
        )
    )
    assert batch.provider == "complete"
    assert any("日历区间不完整" in warning for warning in batch.warnings)


def test_partial_trade_calendar_is_not_retried(app_config):
    class FlakyCalendar(DataProvider):
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def capabilities(self):
            return {Dataset.TRADE_CALENDAR}

        def fetch(self, request):
            self.calls += 1
            days = ["20240101"] if self.calls == 1 else ["20240101", "20240102"]
            return DataBatch(
                pd.DataFrame({"cal_date": days, "is_open": [1] * len(days)}),
                self.name,
                request,
            )

    app_config.providers.priority = ["flaky"]
    app_config.providers.retry.attempts = 2
    app_config.providers.retry.delays = [0, 0]
    provider = FlakyCalendar()
    with pytest.raises(ProviderError, match="交易日历区间不完整"):
        DataRouter(app_config, {provider.name: provider}).fetch(
            DataRequest(
                dataset=Dataset.TRADE_CALENDAR,
                start=date(2024, 1, 1),
                end=date(2024, 1, 2),
            )
        )
    assert provider.calls == 1
