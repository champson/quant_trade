from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_trade.backtest import ExecutionConfig, run_weight_backtest, save_backtest_report


def test_backtest_report_writes_markdown_and_self_contained_html(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=45)
    bars = pd.DataFrame(
        [
            {
                "trade_date": day,
                "symbol": "A",
                "open": 10 + index * 0.1,
                "close": 10.05 + index * 0.1,
            }
            for index, day in enumerate(dates)
        ]
    )
    targets = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    execution = ExecutionConfig(
        initial_cash=100_000,
        commission_rate=0.00025,
        stamp_duty_rate=0.0005,
        slippage_rate=0.0002,
    )
    result = run_weight_backtest(bars, targets, execution)
    benchmark = pd.Series(range(100, 145), index=dates, dtype=float)

    paths = save_backtest_report(
        name="example",
        result=result,
        out_dir=tmp_path / "report",
        execution=execution,
        strategy_config={"window": 20},
        benchmark_equity=benchmark,
        benchmark_name="benchmark",
    )

    assert all(path.exists() for path in paths.as_dict().values())
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert "## 二、核心绩效" in markdown
    assert "未建模" in markdown
    html = paths.html.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "月度表现" in html
    payload = json.loads(paths.metrics.read_text(encoding="utf-8"))
    assert set(payload) == {
        "strategy",
        "benchmark",
        "benchmark_name",
        "benchmark_status",
        "benchmark_period",
    }
    assert payload["benchmark_period"] == {
        "start": "2024-01-02",
        "end": "2024-03-04",
    }


def test_backtest_report_failure_does_not_publish_partial_directory(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A"] * 3,
            "open": [10.0, 10.1, 10.2],
            "close": [10.0, 10.1, 10.2],
        }
    )
    result = run_weight_backtest(
        bars,
        pd.DataFrame({"A": [1.0]}, index=[dates[0]]),
        ExecutionConfig(),
    )
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        "quant_trade.backtest.report._make_charts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("chart failed")),
    )

    with pytest.raises(RuntimeError, match="chart failed"):
        save_backtest_report(
            name="example",
            result=result,
            out_dir=out_dir,
            execution=ExecutionConfig(),
        )

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
