from datetime import date, timedelta

from typer.testing import CliRunner

from quant_trade.cli import app


runner = CliRunner()


def test_data_update_rejects_unknown_adjustment_before_runtime():
    result = runner.invoke(
        app,
        [
            "data",
            "update",
            "--symbols",
            "000001.SZ",
            "--adjustment",
            "hfq2",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_data_update_rejects_adjusted_full_market_snapshot():
    result = runner.invoke(app, ["data", "update", "--adjustment", "hfq"])
    assert result.exit_code == 2
    assert "全市场快照只支持 adjustment=none" in result.output


def test_data_update_rejects_future_end_before_runtime():
    future = (date.today() + timedelta(days=1)).isoformat()
    result = runner.invoke(
        app,
        ["data", "update", "--symbols", "000001.SZ", "--end", future],
    )
    assert result.exit_code == 2
    assert "不支持请求未来行情日期" in result.output


def test_market_history_and_daily_reject_future_dates_before_runtime():
    future = (date.today() + timedelta(days=1)).isoformat()
    history = runner.invoke(
        app, ["data", "market-history", "--start", "2024-01-01", "--end", future]
    )
    daily = runner.invoke(app, ["daily", "run", "--as-of", future])
    assert history.exit_code == 2
    assert daily.exit_code == 2
    assert "不支持请求未来行情日期" in history.output
    assert "不支持请求未来行情日期" in daily.output
