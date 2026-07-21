from __future__ import annotations

import pandas as pd
import pytest

from quant_trade.backtest.metrics import performance_metrics


def test_sparse_monthly_equity_uses_elapsed_calendar_time_for_cagr():
    equity = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2022-01-31", "2023-01-31", "2024-01-31"]),
    )
    metrics = performance_metrics(equity)
    assert metrics["cagr"] == pytest.approx(0.10, rel=0.01)
    assert metrics["annual_volatility"] < 0.1
