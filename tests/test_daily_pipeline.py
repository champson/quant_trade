from __future__ import annotations

from datetime import date

import pandas as pd

from quant_trade.data.base import DataProvider
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import Adjustment, AssetType, DataBatch, Dataset
from quant_trade.pipelines.daily import _strategy_download_groups, run_daily


class OfflineMarketProvider(DataProvider):
    name = "offline"

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


def test_daily_pipeline_runs_offline_and_writes_review(app_config, monkeypatch):
    app_config.providers.market_snapshot_min_symbols = {"stock": 2, "convertible_bond": 2}
    app_config.review = {"indices": {}}
    app_config.strategies = {}
    app_config.providers.priority = ["offline"]
    store = DataStore(app_config)
    router = DataRouter(app_config, {"offline": OfflineMarketProvider()}, store)
    monkeypatch.setattr("quant_trade.pipelines.daily.notify", lambda *_: None)
    result = run_daily(app_config, router, store, date(2024, 1, 8))
    assert result.as_of == date(2024, 1, 8)
    assert set(result.report_paths) >= {"csv", "png", "summary"}
    assert all(pd.io.common.file_exists(path) for path in result.report_paths.values())
    with store.connect() as con:
        status = con.execute("SELECT status FROM runs").fetchone()[0]
    assert status == "success"


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
