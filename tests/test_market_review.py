from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_trade.reports.market_review import build_market_review, portfolio_returns
from quant_trade.reports.render import _chinese_font_properties


def test_market_review_counts_all_symbols():
    dates = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"])
    rows = []
    values = {"A": [10, 11, 12], "B": [10, 9, 8], "C": [10, 10, 10]}
    for symbol, prices in values.items():
        for day, price in zip(dates, prices):
            rows.append({"trade_date": day, "symbol": symbol, "close": price})
    report = build_market_review(pd.DataFrame(rows), "2024-01-08")
    assert report.summary["stocks"] == 3
    assert report.summary["up"] == 1
    assert report.summary["down"] == 1
    assert report.breadth["当天"].sum() == 3
    flat = report.breadth.set_index("区间").loc["平盘", "当天"]
    assert flat == 1


def test_market_review_exact_negative_thresholds_are_symmetric():
    dates = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"])
    returns = [-0.07, -0.05, -0.03, 0.03, 0.05, 0.07]
    rows = []
    for index, value in enumerate(returns):
        rows.extend(
            [
                {"trade_date": dates[0], "symbol": str(index), "close": 100.0},
                {"trade_date": dates[1], "symbol": str(index), "close": 100.0},
                {"trade_date": dates[2], "symbol": str(index), "close": 100 * (1 + value)},
            ]
        )
    breadth = build_market_review(pd.DataFrame(rows), "2024-01-08").breadth.set_index("区间")
    assert breadth.loc["下跌5%-7%", "当天"] == 1
    assert breadth.loc["下跌3%-5%", "当天"] == 1
    assert breadth.loc["下跌0%-3%", "当天"] == 1
    assert breadth.loc["上涨0%-3%", "当天"] == 1
    assert breadth.loc["上涨3%-5%", "当天"] == 1
    assert breadth.loc["上涨5%-7%", "当天"] == 1


def test_portfolio_review_rejects_missing_holding_prices():
    dates = pd.to_datetime(["2023-12-29", "2024-01-02"])
    bars = pd.DataFrame(
        [
            {"trade_date": day, "symbol": "000001.SZ", "close": price}
            for day, price in zip(dates, [10, 11], strict=True)
        ]
    )
    portfolio = pd.DataFrame({"代码": ["000001", "000002"], "权重": [0.5, 0.5]})
    with pytest.raises(ValueError, match="000002.SZ"):
        portfolio_returns(bars, portfolio, "2024-01-02")


def test_chinese_font_uses_first_installed_candidate(monkeypatch):
    from matplotlib import font_manager

    monkeypatch.setattr(
        font_manager.fontManager,
        "ttflist",
        [
            SimpleNamespace(name="DejaVu Sans", fname="/fonts/dejavu.ttf"),
            SimpleNamespace(name="Hiragino Sans GB", fname="/fonts/chinese.ttc"),
        ],
    )
    assert _chinese_font_properties().get_file() == "/fonts/chinese.ttc"
