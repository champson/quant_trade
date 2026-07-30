from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_trade.data.base import PermanentProviderError, TransientProviderError
from quant_trade.data.providers.akshare import AkShareProvider
from quant_trade.data.providers.baostock import BaoStockProvider
from quant_trade.data.providers.tushare import TushareProvider
from quant_trade.data.quality import validate_bars
from quant_trade.models import Adjustment, AssetType, DataRequest, Dataset


class FakeTushareApi:
    def daily(self, *, trade_date):
        assert trade_date == "20200102"
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [trade_date, trade_date],
                "open": [10, 20],
                "high": [11, 21],
                "low": [9, 19],
                "close": [10.5, 20.5],
                "vol": [100, 200],
                "amount": [1000, 4000],
            }
        )

    def stock_basic(self, *, exchange, list_status, fields):
        assert exchange == ""
        assert "delist_date" in fields
        rows = {
            "L": [
                ("000001.SZ", "20100101", None),
                ("000002.SZ", "20150101", None),
                ("200011.SZ", "20100101", None),
            ],
            "D": [
                ("000003.SZ", "20100101", "20191231"),
                ("000005.SZ", "20100101", "20200102"),
            ],
            "P": [("000004.SZ", "20190101", None)],
        }[list_status]
        return pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "list_status": list_status,
                    "list_date": listed,
                    "delist_date": delisted,
                }
                for symbol, listed, delisted in rows
            ]
        )

    def daily_basic(self, *, trade_date):
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "200011.SZ"],
                "trade_date": [trade_date, trade_date, trade_date],
                "total_mv": [100.0, 200.0, 50.0],
            }
        )


def test_tushare_full_market_reports_independent_historical_universe_size():
    provider = TushareProvider(interval_seconds=0)
    provider._pro = FakeTushareApi()
    request = DataRequest(
        dataset=Dataset.BARS,
        start=date(2020, 1, 2),
        end=date(2020, 1, 2),
        asset_type=AssetType.STOCK,
    )

    batch = provider.fetch(request)

    assert batch.metadata["expected_symbols"] == 3
    assert batch.metadata["expected_symbols_source"] == "tushare.stock_basic"
    assert batch.metadata["full_market_response_complete"] is True
    assert batch.metadata["full_market_response_rows"] == 2
    assert batch.metadata["full_market_filtered_rows"] == 2
    assert batch.metadata["full_market_response_limit"] == 6000
    assert set(batch.data["symbol"]) == {"000001.SZ", "000002.SZ"}


def test_tushare_daily_basic_reports_independent_historical_universe_size():
    provider = TushareProvider(interval_seconds=0)
    provider._pro = FakeTushareApi()
    request = DataRequest(
        dataset=Dataset.DAILY_BASIC,
        start=date(2020, 1, 2),
        end=date(2020, 1, 2),
    )

    batch = provider.fetch(request)

    assert batch.metadata["expected_symbols"] == 3
    assert batch.metadata["expected_symbols_source"] == "tushare.stock_basic"
    assert batch.metadata["full_market_response_complete"] is True
    assert batch.metadata["full_market_response_rows"] == 3
    assert batch.metadata["full_market_filtered_rows"] == 2
    assert set(batch.data["ts_code"]) == {"000001.SZ", "000002.SZ"}


@pytest.mark.parametrize("dataset", [Dataset.BARS, Dataset.DAILY_BASIC])
def test_tushare_full_market_empty_response_is_transient(dataset):
    class Api:
        def daily(self, **kwargs):
            return pd.DataFrame()

        def daily_basic(self, **kwargs):
            return pd.DataFrame()

    provider = TushareProvider(interval_seconds=0)
    provider._pro = Api()
    request = DataRequest(
        dataset=dataset,
        start=date(2020, 1, 2),
        end=date(2020, 1, 2),
        asset_type=AssetType.STOCK,
    )

    with pytest.raises(TransientProviderError, match="暂时返回空结果"):
        provider.fetch(request)


def test_symbol_loop_providers_reject_full_market_bar_request():
    request = DataRequest(
        dataset=Dataset.BARS,
        start=date(2020, 1, 2),
        end=date(2020, 1, 2),
        asset_type=AssetType.STOCK,
    )

    assert not BaoStockProvider(interval_seconds=0).supports(request)
    assert not AkShareProvider(interval_seconds=0).supports(request)


def test_tushare_etf_hfq_uses_dated_fund_adjustment_factor():
    class Api:
        def fund_daily(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["510300.SH", "510300.SH"],
                    "trade_date": ["20240102", "20240103"],
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.0, 2.0],
                    "vol": [100, 200],
                    "amount": [1000, 2000],
                }
            )

        def fund_adj(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["510300.SH", "510300.SH"],
                    "trade_date": ["20240102", "20240103"],
                    "adj_factor": [2.0, 3.0],
                }
            )

    provider = TushareProvider(interval_seconds=0)
    provider._pro = Api()
    request = DataRequest(
        dataset=Dataset.BARS,
        symbols=("510300.SH",),
        start=date(2024, 1, 2),
        end=date(2024, 1, 3),
        asset_type=AssetType.ETF,
        adjustment=Adjustment.HFQ,
    )

    batch = provider.fetch(request)

    assert batch.data["close"].tolist() == [2.0, 6.0]
    assert set(batch.data["adjustment"]) == {"hfq"}
    assert batch.metadata["adjustment_evidence"] == "tushare_dated_factor"


def test_tushare_full_market_etf_filters_with_dated_etf_master():
    class Api:
        def etf_basic(self, *, list_status, fields):
            assert fields == "ts_code,list_status,list_date"
            rows = {
                "L": [
                    ("510300.SH", "20120101", None),
                    ("159915.SZ", "20150101", None),
                ],
                "D": [
                    ("510001.SH", "20100101", "20191231"),
                    ("510002.SH", "20100101", "20200102"),
                ],
                "P": [("510003.SH", "20210101", None)],
            }[list_status]
            return pd.DataFrame(
                [
                    {
                        "ts_code": symbol,
                        "list_status": list_status,
                        "list_date": listed,
                    }
                    for symbol, listed, delisted in rows
                ]
            )

        def fund_basic(self, *, market, fields):
            assert market == "E"
            assert "delist_date" in fields
            return pd.DataFrame(
                {
                    "ts_code": [
                        "510300.SH",
                        "159915.SZ",
                        "510001.SH",
                        "510002.SH",
                        "510003.SH",
                    ],
                    "list_date": ["20120101", "20150101", "20100101", "20100101", "20210101"],
                    "delist_date": [None, None, "20191231", "20200102", None],
                }
            )

        def fund_daily(self, *, trade_date):
            assert trade_date == "20200102"
            return pd.DataFrame(
                {
                    "ts_code": ["510300.SH", "159915.SZ", "160001.SZ"],
                    "trade_date": [trade_date] * 3,
                    "open": [4.0, 2.0, 1.0],
                    "high": [4.1, 2.1, 1.1],
                    "low": [3.9, 1.9, 0.9],
                    "close": [4.0, 2.0, 1.0],
                    "vol": [100, 200, 300],
                    "amount": [400, 400, 300],
                }
            )

    provider = TushareProvider(interval_seconds=0)
    provider._pro = Api()
    request = DataRequest(
        dataset=Dataset.BARS,
        start=date(2020, 1, 2),
        end=date(2020, 1, 2),
        asset_type=AssetType.ETF,
    )

    batch = provider.fetch(request)

    assert set(batch.data["symbol"]) == {"510300.SH", "159915.SZ"}
    assert batch.metadata["expected_symbols"] == 2
    assert batch.metadata["expected_symbols_source"] == "tushare.etf_basic"
    assert batch.metadata["full_market_response_complete"] is True
    assert batch.metadata["full_market_response_rows"] == 3
    assert batch.metadata["full_market_filtered_rows"] == 2
    assert batch.metadata["full_market_response_limit"] == 5000


def test_tushare_full_market_etf_rejects_missing_etf_master():
    class Api:
        def etf_basic(self, **kwargs):
            raise RuntimeError("permission denied")

        def fund_daily(self, **kwargs):
            raise AssertionError("must not fetch an unfilterable fund universe")

    provider = TushareProvider(interval_seconds=0)
    provider._pro = Api()
    request = DataRequest(
        dataset=Dataset.BARS,
        start=date(2024, 1, 2),
        end=date(2024, 1, 2),
        asset_type=AssetType.ETF,
    )

    with pytest.raises(PermanentProviderError, match="无法确定 ETF 全市场范围"):
        provider.fetch(request)


def test_tushare_normalizes_suspended_convertible_bond_placeholder():
    class Api:
        def cb_daily(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["110073.SH", "113001.SH"],
                    "trade_date": ["20260721", "20260721"],
                    "open": [0.0, 110.0],
                    "high": [0.0, 112.0],
                    "low": [0.0, 109.0],
                    "close": [106.915, 111.0],
                    "pre_close": [106.915, 110.0],
                    "vol": [0.0, 100.0],
                    "amount": [0.0, 11_100.0],
                }
            )

        def cb_basic(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["110073.SH", "113001.SH", "110001.SH", "110002.SH"],
                    "list_date": ["20200101", "20200101", "20100101", "20100101"],
                    "delist_date": [None, None, "20200101", "20260721"],
                }
            )

    provider = TushareProvider(interval_seconds=0)
    provider._pro = Api()
    request = DataRequest(
        dataset=Dataset.BARS,
        start=date(2026, 7, 21),
        end=date(2026, 7, 21),
        asset_type=AssetType.CONVERTIBLE_BOND,
    )

    batch = provider.fetch(request)

    suspended = batch.data.set_index("symbol").loc["110073.SH"]
    assert suspended[["open", "high", "low", "close"]].tolist() == [106.915] * 4
    assert batch.warnings == ["已将 1 条零成交停牌可转债规范为平盘行情"]
    assert batch.metadata["expected_symbols"] == 2
    assert batch.metadata["expected_symbols_source"] == "tushare.cb_basic"
    assert validate_bars(batch.data) == []


def test_tushare_does_not_normalize_unconfirmed_convertible_bond_zero_ohlc():
    raw = pd.DataFrame(
        {
            "symbol": ["110073.SH"],
            "trade_date": pd.to_datetime(["2026-07-21"]),
            "open": [0.0],
            "high": [0.0],
            "low": [0.0],
            "close": [106.915],
            "volume": [0.0],
            "amount": [0.0],
            "pre_close": [105.0],
        }
    )

    normalized, count = TushareProvider._normalize_convertible_bond_suspensions(raw)

    assert count == 0
    assert normalized.loc[0, "open"] == 0.0
    assert "pre_close" not in normalized
