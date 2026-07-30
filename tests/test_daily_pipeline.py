from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.data.base import DataProvider
from quant_trade.data.minute_archive import ImportResult
from quant_trade.data.quality import DataQualityError
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.dashboard.app import _latest_daily_generation
from quant_trade.models import Adjustment, AssetType, DataBatch, Dataset
from quant_trade.pipelines.daily import (
    _anchors,
    _strategy_download_groups,
    _strategy_history_calendar_days,
    run_daily,
)


class OfflineMarketProvider(DataProvider):
    name = "offline"

    def __init__(self):
        self.requests = []

    def capabilities(self):
        return {Dataset.BARS, Dataset.TRADE_CALENDAR}

    def fetch(self, request):
        self.requests.append(request)
        if request.dataset == Dataset.TRADE_CALENDAR:
            dates = pd.date_range(request.start, request.end, freq="D")
            return DataBatch(
                pd.DataFrame(
                    {
                        "cal_date": dates.strftime("%Y%m%d"),
                        "is_open": (dates.weekday < 5).astype(int),
                    }
                ),
                self.name,
                request,
            )
        day = pd.Timestamp(request.end)
        base = 10 + day.day / 10
        rows = []
        for i, symbol in enumerate(("000001.SZ", "600000.SH")):
            price = base - i
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "bar_time": pd.NaT,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "volume": 100,
                    "amount": 1000,
                    "source": self.name,
                    "adjustment": str(request.adjustment),
                }
            )
        return DataBatch(pd.DataFrame(rows), self.name, request)


class PartialConvertibleProvider(OfflineMarketProvider):
    def fetch(self, request):
        batch = super().fetch(request)
        if request.dataset == Dataset.BARS and request.asset_type == AssetType.CONVERTIBLE_BOND:
            return DataBatch(batch.data.iloc[:1].copy(), self.name, request)
        return batch


def test_review_anchors_require_previous_complete_trading_day():
    with pytest.raises(ValueError, match="至少需要一个完整交易日"):
        _anchors([date(2024, 1, 8)], date(2024, 1, 8))


def test_daily_pipeline_runs_offline_and_writes_review(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {
        "stock": 2,
        "etf": 2,
        "convertible_bond": 2,
    }
    app_config.review = {"indices": {}}
    app_config.strategies = {}
    app_config.providers.priority = ["offline"]
    store = DataStore(app_config)
    provider = OfflineMarketProvider()
    router = DataRouter(app_config, {"offline": provider}, store)
    monkeypatch.setattr("quant_trade.pipelines.daily.notify", lambda *_: None)
    result = run_daily(app_config, router, store, date(2024, 1, 8))
    assert result.as_of == date(2024, 1, 8)
    assert set(result.report_paths) >= {"csv", "png", "summary", "daily_review"}
    assert all(pd.io.common.file_exists(path) for path in result.report_paths.values())
    assert (app_config.paths.artifacts_dir / "daily" / "daily_manifest_2024-01-08.json").exists()
    generation = _latest_daily_generation(app_config.paths.artifacts_dir)
    assert generation is not None
    assert generation["as_of"] == pd.Timestamp("2024-01-08")
    full_market_assets = {
        request.asset_type
        for request in provider.requests
        if request.dataset == Dataset.BARS and not request.symbols
    }
    assert full_market_assets == {
        AssetType.STOCK,
        AssetType.ETF,
        AssetType.CONVERTIBLE_BOND,
    }
    with store.connect() as con:
        status = con.execute("SELECT status FROM runs").fetchone()[0]
    assert status == "success"


def test_daily_pipeline_omits_incomplete_convertible_bond_report(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2, "convertible_bond": 2}
    app_config.review = {"indices": {}}
    app_config.strategies = {}
    app_config.providers.priority = ["partial"]
    store = DataStore(app_config)
    router = DataRouter(app_config, {"partial": PartialConvertibleProvider()}, store)
    monkeypatch.setattr("quant_trade.pipelines.daily.notify", lambda *_: None)

    result = run_daily(app_config, router, store, date(2024, 1, 8))

    assert "convertible_bonds" not in result.report_paths
    assert any("可转债报告已跳过，快照不完整" in warning for warning in result.warnings)


def test_daily_pipeline_fails_when_minute_inbox_contains_failed_archive(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {
        "stock": 2,
        "convertible_bond": 2,
    }
    app_config.review = {"indices": {}}
    app_config.strategies = {}
    app_config.providers.priority = ["offline"]
    store = DataStore(app_config)
    router = DataRouter(app_config, {"offline": OfflineMarketProvider()}, store)
    monkeypatch.setattr("quant_trade.pipelines.daily.notify", lambda *_: None)
    monkeypatch.setattr(
        "quant_trade.pipelines.daily.MinuteArchiveImporter.import_inbox",
        lambda *_args, **_kwargs: [ImportResult("bad", "broken.zip", "failed")],
    )

    with pytest.raises(DataQualityError, match="broken.zip"):
        run_daily(app_config, router, store, date(2024, 1, 8))
    with store.connect() as con:
        status = con.execute("SELECT status FROM runs").fetchone()[0]
    assert status == "failed"


def test_strategy_download_groups_honour_asset_and_benchmark_types():
    groups = _strategy_download_groups(
        {
            "custom": {
                "enabled": True,
                "asset_type": "stock",
                "adjustment": "qfq",
                "symbols": ["000001.SZ"],
                "benchmark": "000300.SH",
                "benchmark_asset_type": "index",
            }
        }
    )
    assert groups[(AssetType.STOCK, Adjustment.QFQ)] == {"000001.SZ"}
    assert groups[(AssetType.INDEX, Adjustment.NONE)] == {"000300.SH"}


def test_strategy_history_window_expands_for_large_parameters():
    days = _strategy_history_calendar_days(
        {
            "etf_rotation": {
                "enabled": True,
                "momentum_windows": [500],
                "ma_window": 20,
            }
        }
    )

    assert days >= 1002


def test_daily_failure_does_not_publish_review_or_signal_generation(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {
        "stock": 2,
        "convertible_bond": 2,
    }
    app_config.review = {"indices": {}}
    app_config.strategies = {
        "logbias": {
            "enabled": True,
            "asset_type": "etf",
            "symbols": ["510300.SH"],
            "ema_window": 2,
        }
    }
    app_config.providers.priority = ["strategy-offline"]
    provider = StrategyPipelineProvider()
    store = DataStore(app_config)
    router = DataRouter(app_config, {provider.name: provider}, store)
    monkeypatch.setattr("quant_trade.pipelines.daily.notify", lambda *_: None)
    monkeypatch.setattr(
        "quant_trade.pipelines.daily.build_strategy_signal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DataQualityError("signal failed")),
    )

    with pytest.raises(DataQualityError, match="signal failed"):
        run_daily(app_config, router, store, date(2024, 1, 8))

    assert not list((app_config.paths.artifacts_dir / "reviews").glob("*"))
    assert not list((app_config.paths.artifacts_dir / "signals").glob("**/*"))
    assert not list((app_config.paths.artifacts_dir / "daily").glob("*"))


class StrategyPipelineProvider(DataProvider):
    name = "strategy-offline"

    def __init__(self):
        self.bar_requests = []

    def capabilities(self):
        return {Dataset.BARS, Dataset.TRADE_CALENDAR}

    def fetch(self, request):
        if request.dataset == Dataset.TRADE_CALENDAR:
            dates = pd.date_range(request.start, request.end, freq="D")
            return DataBatch(
                pd.DataFrame(
                    {
                        "cal_date": dates.strftime("%Y%m%d"),
                        "is_open": (dates.weekday < 5).astype(int),
                    }
                ),
                self.name,
                request,
            )
        self.bar_requests.append(request)
        symbols = request.symbols or ("000001.SZ", "600000.SH")
        days = pd.bdate_range(request.start, request.end)
        rows = []
        for symbol_index, symbol in enumerate(symbols):
            for day_index, day in enumerate(days):
                price = 10 + symbol_index + day_index / 100
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": day,
                        "bar_time": pd.NaT,
                        "open": price,
                        "high": price + 0.1,
                        "low": price - 0.1,
                        "close": price + 0.05,
                        "volume": 100,
                        "amount": 1000,
                        "source": self.name,
                        "adjustment": str(request.adjustment),
                    }
                )
        return DataBatch(pd.DataFrame(rows), self.name, request)


def test_daily_pipeline_updates_strategy_and_benchmark_contracts(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2, "convertible_bond": 2}
    app_config.review = {"indices": {}}
    app_config.strategies = {
        "etf_rotation": {
            "enabled": True,
            "asset_type": "etf",
            "adjustment": "hfq",
            "symbols": ["510300.SH", "510500.SH"],
            "momentum_windows": [2],
            "ma_window": 2,
            "min_momentum": -1,
            "rebalance_days": 5,
            "benchmark": "000300.SH",
            "benchmark_asset_type": "index",
        }
    }
    app_config.providers.priority = ["strategy-offline"]
    provider = StrategyPipelineProvider()
    store = DataStore(app_config)
    router = DataRouter(app_config, {provider.name: provider}, store)
    monkeypatch.setattr("quant_trade.pipelines.daily.notify", lambda *_: None)

    result = run_daily(app_config, router, store, date(2024, 1, 8))

    assert "etf_rotation" in result.signals
    contracts = {
        (request.asset_type, request.adjustment, request.symbols[0])
        for request in provider.bar_requests
        if request.symbols
    }
    assert (AssetType.ETF, Adjustment.HFQ, "510300.SH") in contracts
    assert (AssetType.ETF, Adjustment.HFQ, "510500.SH") in contracts
    assert (AssetType.INDEX, Adjustment.NONE, "000300.SH") in contracts
    assert store.daily_path(AssetType.ETF, "510300.SH", "hfq").exists()
    assert store.daily_path(AssetType.INDEX, "000300.SH", "none").exists()
