from __future__ import annotations

from datetime import date

import pandas as pd

from quant_trade.data.base import DataProvider
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import DataBatch, Dataset
from quant_trade.pipelines.daily import run_daily


class OfflineMarketProvider(DataProvider):
    name = "offline"

    def capabilities(self):
        return {Dataset.BARS, Dataset.TRADE_CALENDAR}

    def fetch(self, request):
        if request.dataset == Dataset.TRADE_CALENDAR:
            dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-05", "2024-01-08"])
            return DataBatch(pd.DataFrame({
                "cal_date": dates.strftime("%Y%m%d"), "is_open": 1,
            }), self.name, request)
        day = pd.Timestamp(request.end)
        base = {pd.Timestamp("2023-12-29"): 10, pd.Timestamp("2024-01-05"): 11, pd.Timestamp("2024-01-08"): 12}[day]
        rows = []
        for i, symbol in enumerate(("000001.SZ", "600000.SH")):
            price = base - i
            rows.append({
                "symbol": symbol, "trade_date": day, "bar_time": pd.NaT,
                "open": price, "high": price + 1, "low": price - 1,
                "close": price, "volume": 100, "amount": 1000, "source": self.name,
            })
        return DataBatch(pd.DataFrame(rows), self.name, request)


def test_daily_pipeline_runs_offline_and_writes_review(app_config, monkeypatch):
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
