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
