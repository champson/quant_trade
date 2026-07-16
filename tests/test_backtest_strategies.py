from __future__ import annotations

import pandas as pd
import pytest

from quant_trade.backtest.engine import ExecutionConfig, run_weight_backtest
from quant_trade.strategies.etf_rotation import EtfRotationStrategy, _rebalance_periods
from quant_trade.strategies.microcap import MicrocapStrategy


def _bars(symbols=("A",), periods=5):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for j, symbol in enumerate(symbols):
        for i, day in enumerate(dates):
            price = 10 + j + i
            rows.append(
                {
                    "trade_date": day,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price + 0.5,
                    "volume": 1000,
                    "amount": 10000,
                }
            )
    return pd.DataFrame(rows)


def test_close_signal_executes_on_next_bar_open():
    bars = _bars()
    dates = sorted(bars["trade_date"].unique())
    targets = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    result = run_weight_backtest(
        bars,
        targets,
        ExecutionConfig(initial_cash=1000, commission_rate=0, stamp_duty_rate=0, slippage_rate=0),
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["signal_date"] == dates[0]
    assert trade["date"] == dates[1]
    assert trade["price"] == 11


def test_future_prices_do_not_change_past_trade():
    bars = _bars(periods=4)
    dates = sorted(bars["trade_date"].unique())
    targets = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    first = run_weight_backtest(bars, targets).trades.iloc[0]
    changed = bars.copy()
    changed.loc[changed["trade_date"] == dates[-1], ["open", "close"]] = 999
    second = run_weight_backtest(changed, targets).trades.iloc[0]
    assert first[["date", "price", "quantity"]].equals(second[["date", "price", "quantity"]])


def test_missing_bar_keeps_last_price_in_equity():
    bars = _bars(("A", "B"), periods=4)
    dates = sorted(bars["trade_date"].unique())
    # A is suspended on day 3: no bar at all.
    bars = bars[~((bars["symbol"] == "A") & (bars["trade_date"] == dates[2]))]
    targets = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    result = run_weight_backtest(
        bars,
        targets,
        ExecutionConfig(initial_cash=1000, commission_rate=0, stamp_duty_rate=0, slippage_rate=0),
    )
    # The position is valued at day 2's close during the suspension, not zero.
    assert result.equity.loc[dates[2]] == pytest.approx(result.equity.loc[dates[1]])
    assert result.equity.loc[dates[2]] > 0


def test_order_blocked_by_missing_open_retries_next_day():
    bars = _bars(("A", "B"), periods=5)
    dates = sorted(bars["trade_date"].unique())
    # Buy A on day 2. Its sell target arrives on day 3, when A is suspended.
    bars = bars[~((bars["symbol"] == "A") & (bars["trade_date"] == dates[2]))]
    targets = pd.DataFrame({"A": [1.0, 0.0]}, index=[dates[0], dates[1]])
    result = run_weight_backtest(
        bars,
        targets,
        ExecutionConfig(
            initial_cash=1000,
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )
    sells = result.trades[result.trades["side"] == "SELL"]
    assert len(sells) == 1
    assert sells.iloc[0]["date"] == dates[3]


def test_blocked_order_retry_does_not_rebalance_other_positions():
    bars = _bars(("A", "B"), periods=5)
    dates = sorted(bars["trade_date"].unique())
    bars = bars[~((bars["symbol"] == "A") & (bars["trade_date"] == dates[1]))]
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]])
    result = run_weight_backtest(
        bars,
        targets,
        ExecutionConfig(
            initial_cash=1000,
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )
    retry_day = result.trades[result.trades["date"] == dates[2]]
    assert retry_day["symbol"].tolist() == ["A"]


def test_blocked_sale_retries_dependent_buy_after_cash_is_released():
    bars = _bars(("A", "B"), periods=5)
    dates = sorted(bars["trade_date"].unique())
    bars = bars[~((bars["symbol"] == "A") & (bars["trade_date"] == dates[2]))]
    targets = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[dates[0], dates[1]])
    result = run_weight_backtest(
        bars,
        targets,
        ExecutionConfig(
            initial_cash=1000,
            commission_rate=0,
            stamp_duty_rate=0,
            slippage_rate=0,
        ),
    )
    retry_day = result.trades[result.trades["date"] == dates[3]]
    assert retry_day[["symbol", "side"]].values.tolist() == [
        ["A", "SELL"],
        ["B", "BUY"],
    ]
    assert (result.trades["quantity"] > 0).all()
    assert result.positions.iloc[-1]["B"] == pytest.approx(1.0)


def test_etf_rotation_selects_top_candidate_instead_of_skipping_two():
    bars = _bars(("A", "B", "C"), periods=8)
    strategy = EtfRotationStrategy(
        {
            "momentum_windows": [2],
            "ma_window": 2,
            "hold_num": 1,
            "rebalance_days": 1,
            "min_momentum": -1,
        }
    )
    targets = strategy.generate_targets(bars)
    selected = targets.iloc[-1][targets.iloc[-1] > 0]
    assert len(selected) == 1
    assert selected.index[0] in {"A", "B", "C"}


def test_etf_rebalance_periods_do_not_shift_with_window_start():
    dates = pd.bdate_range("2024-01-02", periods=20)
    full = _rebalance_periods(dates, 5, "2000-01-03")
    sliced = _rebalance_periods(dates[7:], 5, "2000-01-03")
    assert sliced.tolist() == full[7:].tolist()


def test_microcap_selection_uses_previous_day_market_cap():
    dates = pd.bdate_range("2024-01-01", periods=3)
    bars = pd.DataFrame(
        [
            {"trade_date": day, "symbol": symbol, "close": 10, "total_mv": value}
            for day, values in zip(dates, [(1, 2), (100, 1), (100, 1)])
            for symbol, value in zip(("000001.SZ", "000002.SZ"), values)
        ]
    )
    strategy = MicrocapStrategy(
        {
            "pool_size": 1,
            "hold_count": 1,
            "selection": "smallest",
            "rebalance": "daily",
            "exclude_prefixes": [],
        }
    )
    targets = strategy.generate_targets(bars)
    # On day 2 it must still select symbol 1 using day 1's market cap.
    assert targets.loc[dates[1], "000001.SZ"] == 1
    assert targets.loc[dates[2], "000002.SZ"] == 1


def test_microcap_weekly_targets_do_not_change_when_future_day_is_appended():
    dates = pd.bdate_range("2024-01-05", periods=6)
    bars = pd.DataFrame(
        [
            {
                "trade_date": day,
                "symbol": symbol,
                "close": 10,
                "total_mv": value,
            }
            for index, day in enumerate(dates)
            for symbol, value in zip(
                ("000001.SZ", "000002.SZ"),
                ((1, 100) if index < 4 else (100, 1)),
            )
        ]
    )
    strategy = MicrocapStrategy(
        {
            "pool_size": 1,
            "hold_count": 1,
            "selection": "smallest",
            "rebalance": "weekly",
            "exclude_prefixes": [],
        }
    )
    through_thursday = strategy.generate_targets(bars[bars["trade_date"] <= dates[-2]])
    through_friday = strategy.generate_targets(bars)
    pd.testing.assert_frame_equal(
        through_thursday,
        through_friday.loc[through_thursday.index],
    )
