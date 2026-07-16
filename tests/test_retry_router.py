from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.config import RetryConfig
from quant_trade.data.base import DataProvider, EmptyDataError, PermanentProviderError
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
        data = pd.DataFrame({
            "symbol": ["000001.SZ"], "trade_date": [pd.Timestamp("2024-01-02")],
            "bar_time": [pd.NaT], "open": [10.0], "high": [11.0], "low": [9.0],
            "close": [10.5], "volume": [100.0], "amount": [1000.0], "source": [self.name],
        })
        return DataBatch(data, self.name, request)


class EmptyProvider(DataProvider):
    def __init__(self, name: str):
        self.name = name

    def capabilities(self):
        return {Dataset.BARS}

    def fetch(self, request):
        raise EmptyDataError("返回空行情")


def test_router_falls_back_on_empty_without_tripping_circuit(app_config):
    app_config.providers.priority = ["first", "second"]
    app_config.providers.retry.circuit_failures = 1
    router = DataRouter(app_config, {
        "first": EmptyProvider("first"),
        "second": FakeProvider("second", False),
    })
    batch = router.fetch(DataRequest(
        dataset=Dataset.BARS, symbols=("000001.SZ",),
        start=date(2024, 1, 1), end=date(2024, 1, 2),
    ))
    assert batch.provider == "second"
    # A single failure trips the breaker at threshold 1, so "first" staying
    # available proves the empty result was not counted as a failure.
    assert router.circuit.allow("first")


def test_router_records_explicit_fallback(app_config):
    app_config.providers.priority = ["first", "second"]
    app_config.providers.retry.delays = [0] * 6
    router = DataRouter(app_config, {
        "first": FakeProvider("first", True),
        "second": FakeProvider("second", False),
    })
    batch = router.fetch(DataRequest(
        dataset=Dataset.BARS, symbols=("000001.SZ",),
        start=date(2024, 1, 1), end=date(2024, 1, 2),
    ))
    assert batch.provider == "second"
    assert any("回退" in warning for warning in batch.warnings)

