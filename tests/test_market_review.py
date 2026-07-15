from __future__ import annotations

import pandas as pd

from quant_trade.reports.market_review import build_market_review


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
