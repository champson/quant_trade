from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_trade.config import AppConfig, BacktestConfig, MinuteConfig, RetryConfig


def test_retry_delays_cannot_be_negative():
    with pytest.raises(ValidationError):
        RetryConfig(delays=[-1])


def test_backtest_rates_and_strategy_storage_contracts_are_validated():
    with pytest.raises(ValidationError):
        BacktestConfig(slippage_rate=1)
    with pytest.raises(ValidationError, match="资产/复权配置无效"):
        AppConfig(strategies={"custom": {"asset_type": "crypto"}})


def test_etf_rotation_rejects_ambiguous_business_day_cycle():
    with pytest.raises(ValidationError, match="rebalance_days"):
        AppConfig(strategies={"etf_rotation": {"rebalance_days": 7}})


def test_enabled_non_marketwide_strategy_requires_symbols():
    with pytest.raises(ValidationError, match="symbols 不能为空"):
        AppConfig(strategies={"logbias": {"enabled": True, "symbols": []}})


def test_microcap_rejects_adjusted_storage_contract():
    with pytest.raises(ValidationError, match="adjustment=none"):
        AppConfig(strategies={"microcap": {"enabled": True, "adjustment": "hfq"}})


def test_minute_timestamp_convention_is_an_enum():
    with pytest.raises(ValidationError, match="timestamp_convention"):
        MinuteConfig(timestamp_convention="guess")


def test_strategy_specific_parameters_are_validated():
    with pytest.raises(ValidationError, match="selection"):
        AppConfig(strategies={"microcap": {"selection": "rnak"}})
    with pytest.raises(ValidationError, match="stop < entry < overheat"):
        AppConfig(
            strategies={
                "logbias": {
                    "symbols": ["510300.SH"],
                    "stop": 0.1,
                    "entry": 0.0,
                    "overheat": 0.2,
                }
            }
        )


def test_strategy_validation_uses_runtime_logbias_and_rps_contracts():
    config = AppConfig(strategies={"logbias": {"entry": 10}})
    assert config.strategies["logbias"]["entry"] == 10

    with pytest.raises(ValidationError, match="rps_lookback_days"):
        AppConfig(strategies={"microcap": {"selection": "rps", "rps_lookback_days": 0}})
    with pytest.raises(ValidationError, match="rps_target"):
        AppConfig(strategies={"microcap": {"selection": "rps", "rps_target": 101}})
    with pytest.raises(ValidationError, match="momentum_windows"):
        AppConfig(
            strategies={
                "etf_rotation": {
                    "symbols": ["510300.SH"],
                    "momentum_windows": [20, 0],
                }
            }
        )
