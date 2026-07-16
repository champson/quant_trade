from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_trade.backtest.metrics import performance_metrics


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 1_000_000
    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.0002
    lot_size: int = 1
    risk_free_annual: float = 0.0


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    target_weights: pd.DataFrame
    metrics: dict[str, float]
    artifacts: dict[str, Path] = field(default_factory=dict)


def _panels(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"trade_date", "symbol", "open", "close"}
    if not required <= set(bars.columns):
        raise ValueError(f"回测行情缺少字段: {sorted(required - set(bars.columns))}")
    work = bars.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    opens = work.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    closes = work.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    return opens, closes


def run_weight_backtest(
    bars: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: ExecutionConfig | None = None,
) -> BacktestResult:
    """Execute close-generated targets at the next available bar's open."""
    config = config or ExecutionConfig()
    opens, closes = _panels(bars)
    symbols = sorted(set(opens.columns) | set(target_weights.columns))
    opens = opens.reindex(columns=symbols)
    closes = closes.reindex(columns=symbols)
    targets = target_weights.copy()
    targets.index = pd.to_datetime(targets.index)
    targets = targets.reindex(columns=symbols).fillna(0.0).sort_index()
    if (targets < -1e-12).any().any() or (targets.sum(axis=1) > 1.000001).any():
        raise ValueError("目标权重只支持不超过100%的多头组合")

    cash = float(config.initial_cash)
    shares = pd.Series(0.0, index=symbols)
    # Positions keep their last observed price across missing bars (e.g. suspension)
    # instead of being valued at zero.
    marks = closes.fillna(opens).ffill()
    previous_marks = marks.shift(1)
    equities: list[float] = []
    positions: list[pd.Series] = []
    trades: list[dict] = []
    dates = opens.index
    last_desired: pd.Series | None = None
    pending_symbols: set[str] = set()

    for current in dates:
        open_px = opens.loc[current]
        # A signal observed at the previous bar close can only trade now.
        previous_dates = targets.index[targets.index < current]
        desired = targets.loc[previous_dates[-1]] if len(previous_dates) else None
        signal_changed = desired is not None and (
            last_desired is None or not desired.equals(last_desired)
        )
        should_trade = desired is not None and (signal_changed or bool(pending_symbols))
        if should_trade:
            mark_open = open_px.fillna(previous_marks.loc[current])
            portfolio_open = cash + float((shares * mark_open.fillna(0)).sum())
            current_value = shares * mark_open
            desired_value = desired * portfolio_open
            deltas = desired_value - current_value
            # Sell first so sale proceeds are available for buys.
            candidates = (
                deltas.index if signal_changed else deltas.index[deltas.index.isin(pending_symbols)]
            )
            retry_deltas = deltas.loc[candidates]
            order = list(retry_deltas[retry_deltas < -1e-9].index) + list(
                retry_deltas[retry_deltas > 1e-9].index
            )
            next_pending: set[str] = set()
            for symbol in order:
                price = open_px.get(symbol)
                if pd.isna(price) or price <= 0:
                    next_pending.add(symbol)
                    continue
                delta_value = float(deltas[symbol])
                side = "BUY" if delta_value > 0 else "SELL"
                exec_price = float(price) * (
                    1 + config.slippage_rate if side == "BUY" else 1 - config.slippage_rate
                )
                quantity = abs(delta_value) / exec_price
                if config.lot_size > 1:
                    quantity = np.floor(quantity / config.lot_size) * config.lot_size
                if side == "SELL":
                    quantity = min(quantity, shares[symbol])
                if quantity <= 0:
                    continue
                requested_quantity = float(quantity)
                notional = quantity * exec_price
                commission = notional * config.commission_rate
                tax = notional * config.stamp_duty_rate if side == "SELL" else 0.0
                if side == "BUY":
                    affordable = cash / (exec_price * (1 + config.commission_rate))
                    quantity = min(quantity, affordable)
                    # A blocked sale can leave a dependent buy with no cash.
                    # Preserve every material unfilled remainder for the next
                    # open; never emit a zero-quantity trade.
                    remainder = requested_quantity - quantity
                    if remainder * exec_price >= exec_price * max(config.lot_size, 1):
                        next_pending.add(symbol)
                    if quantity <= 1e-12:
                        continue
                    notional = quantity * exec_price
                    commission = notional * config.commission_rate
                    cash -= notional + commission
                    shares[symbol] += quantity
                else:
                    cash += notional - commission - tax
                    shares[symbol] -= quantity
                trades.append(
                    {
                        "date": current,
                        "signal_date": previous_dates[-1],
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": exec_price,
                        "notional": notional,
                        "commission": commission,
                        "tax": tax,
                    }
                )
            pending_symbols = next_pending
            last_desired = desired.copy()
        mark_close = marks.loc[current].fillna(0)
        equity = cash + float((shares * mark_close).sum())
        equities.append(equity)
        value = shares * mark_close
        weights = value / equity if equity else value * 0
        weights.name = current
        positions.append(weights)

    equity_series = pd.Series(equities, index=dates, name="equity")
    positions_df = pd.DataFrame(positions, index=dates).fillna(0.0)
    trades_df = pd.DataFrame(trades)
    return BacktestResult(
        equity=equity_series,
        positions=positions_df,
        trades=trades_df,
        target_weights=targets,
        metrics=performance_metrics(equity_series, risk_free=config.risk_free_annual),
    )
