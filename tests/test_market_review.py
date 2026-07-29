from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_trade.reports.market_review import (
    build_daily_review_table,
    build_market_review,
    nav_period_returns,
    portfolio_returns,
    price_bias,
)
from quant_trade.reports.render import _chinese_font_properties, save_market_review


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
    flat = report.breadth.set_index("区间").loc["涨幅>-5%到0%", "当天"]
    assert flat == 1


def test_market_review_uses_requested_exhaustive_return_buckets():
    dates = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"])
    returns = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.11]
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
    assert list(breadth["当天"]) == [1, 1, 1, 1, 1, 1]


def test_daily_review_table_contains_period_counts_percentages_and_bias25():
    dates = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"])
    rows = []
    for symbol, prices in {"A": [10, 11, 12], "B": [10, 9, 8], "C": [10, 10, 10]}.items():
        rows.extend(
            {"trade_date": day, "symbol": symbol, "close": price}
            for day, price in zip(dates, prices, strict=True)
        )
    review = build_market_review(pd.DataFrame(rows), "2024-01-08")
    indices = pd.DataFrame(
        {"今年": [0.1], "本月": [0.1], "本周": [0.02], "当天": [0.01]},
        index=["上证指数"],
    )
    table = build_daily_review_table(
        review,
        index_returns=indices,
        index_bias=pd.Series({"上证指数": 0.0123}),
    ).set_index("名称")

    assert list(table.columns) == ["今年", "本月", "本周", "当天", "BIAS25"]
    assert table.loc["A股总数量", "当天"] == "3"
    assert table.loc["A股上涨比例", "当天"] == "33.33%"
    assert table.loc["涨幅>-5%到0%", "当天"] == "33.33%"
    assert table.loc["上证指数", "今年"] == "10.00%"
    assert table.loc["上证指数", "BIAS25"] == "1.23%"
    assert table.loc["我的实盘合计", "当天"] == ""


def test_price_bias_uses_25_day_simple_moving_average():
    prices = list(range(1, 26))
    bars = pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-01", periods=25),
            "symbol": "000001.SH",
            "close": prices,
        }
    )
    result = price_bias(bars, window=25)
    assert result["000001.SH"] == pytest.approx(25 / 13 - 1)


def test_nav_period_returns_uses_market_review_anchors():
    values = pd.Series(
        [100, 105, 110],
        index=pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"]),
    )
    anchors = {
        "当天": pd.Timestamp("2024-01-05"),
        "本周": pd.Timestamp("2023-12-29"),
        "本月": pd.Timestamp("2023-12-29"),
        "今年": pd.Timestamp("2023-12-29"),
    }
    result = nav_period_returns(values, anchors, "2024-01-08")
    assert result["当天"] == pytest.approx(110 / 105 - 1)
    assert result["今年"] == pytest.approx(0.1)


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


def test_portfolio_review_rejects_missing_anchor_price_for_existing_holding():
    bars = pd.DataFrame(
        [
            {"trade_date": "2023-12-29", "symbol": "000001.SZ", "close": 10},
            {"trade_date": "2024-01-02", "symbol": "000001.SZ", "close": 11},
            {"trade_date": "2024-01-02", "symbol": "000002.SZ", "close": 20},
        ]
    )
    portfolio = pd.DataFrame({"代码": ["000001", "000002"], "权重": [0.5, 0.5]})

    with pytest.raises(ValueError, match="收益锚点缺少行情.*000002.SZ"):
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


def test_review_publish_removes_stale_optional_artifact_and_writes_manifest(tmp_path):
    bars = pd.DataFrame(
        [
            {"trade_date": "2023-12-29", "symbol": "000001.SZ", "close": 10},
            {"trade_date": "2024-01-02", "symbol": "000001.SZ", "close": 11},
        ]
    )
    review = build_market_review(bars, "2024-01-02")
    optional = pd.DataFrame({"当天": [0.1]}, index=["可转债"])

    first = save_market_review(review, tmp_path, convertible_summary=optional)
    assert first["convertible_bonds"].exists()
    assert first["daily_review"].exists()

    second = save_market_review(review, tmp_path)

    assert "convertible_bonds" not in second
    assert not first["convertible_bonds"].exists()
    manifest = tmp_path / "review_manifest_2024-01-02.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "convertible_bonds" not in payload["files"]
    assert set(payload["sha256"]) == set(payload["files"])
