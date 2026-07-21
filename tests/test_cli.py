from datetime import date, timedelta

import pandas as pd
from typer.testing import CliRunner

from quant_trade.cli import app
from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType


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


def test_data_update_rejects_multi_day_full_market_request():
    result = runner.invoke(
        app,
        ["data", "update", "--start", "2024-01-01", "--end", "2024-01-02"],
    )
    assert result.exit_code == 2
    assert "market-history" in result.output


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


def test_review_close_requires_complete_target_and_anchor_snapshots(app_config, monkeypatch):
    monkeypatch.setattr("quant_trade.cli.load_config", lambda _: app_config)
    store = DataStore(app_config)
    days = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-08"])
    bars = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 3,
            "trade_date": days,
            "open": [10.0, 10.5, 11.0],
            "high": [10.5, 11.0, 11.5],
            "low": [9.5, 10.0, 10.5],
            "close": [10.0, 10.5, 11.0],
            "volume": [100.0] * 3,
            "amount": [1000.0] * 3,
            "source": ["test"] * 3,
        }
    )
    store.write_daily(bars, AssetType.STOCK)

    incomplete = runner.invoke(app, ["review", "close", "--as-of", "2024-01-08"])
    assert incomplete.exit_code == 2
    assert "缺少完整全市场快照" in incomplete.output

    for day in days:
        assert store.mark_market_snapshot(
            AssetType.STOCK,
            day.date(),
            row_count=1,
            symbol_count=1,
            expected_symbols=1,
            symbols=["000001.SZ"],
        )
    complete = runner.invoke(app, ["review", "close", "--as-of", "2024-01-08"])
    assert complete.exit_code == 0


def test_minute_import_inbox_exits_nonzero_when_any_archive_fails(app_config, monkeypatch):
    app_config.ensure_directories()
    (app_config.minute.inbox / "broken.zip").write_text("not a zip", encoding="utf-8")
    monkeypatch.setattr("quant_trade.cli.load_config", lambda _: app_config)

    result = runner.invoke(app, ["data", "minute", "import-inbox"])

    assert result.exit_code == 2
    assert '"status": "failed"' in result.output
