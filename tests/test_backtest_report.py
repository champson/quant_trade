from __future__ import annotations

import json

import pandas as pd

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
    assert set(payload) == {"strategy", "benchmark", "benchmark_name"}
