from __future__ import annotations

from dataclasses import dataclass

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
    equities: list[float] = []
    positions: list[pd.Series] = []
    trades: list[dict] = []
    dates = opens.index
    last_desired: pd.Series | None = None

    for i, current in enumerate(dates):
        open_px = opens.loc[current]
        close_px = closes.loc[current]
        # A signal observed at the previous bar close can only trade now.
        previous_dates = targets.index[targets.index < current]
        desired = targets.loc[previous_dates[-1]] if len(previous_dates) else None
        changed = desired is not None and (
            last_desired is None or not desired.equals(last_desired)
        )
        if changed:
            mark_open = open_px.where(open_px.notna(), close_px)
            portfolio_open = cash + float((shares * mark_open.fillna(0)).sum())
            current_value = shares * mark_open
            desired_value = desired * portfolio_open
            deltas = desired_value - current_value
            # Sell first so sale proceeds are available for buys.
            order = list(deltas[deltas < -1e-9].index) + list(deltas[deltas > 1e-9].index)
            for symbol in order:
                price = open_px.get(symbol)
                if pd.isna(price) or price <= 0:
                    continue
                delta_value = float(deltas[symbol])
                side = "BUY" if delta_value > 0 else "SELL"
                exec_price = float(price) * (1 + config.slippage_rate if side == "BUY" else 1 - config.slippage_rate)
                quantity = abs(delta_value) / exec_price
                if config.lot_size > 1:
                    quantity = np.floor(quantity / config.lot_size) * config.lot_size
                if side == "SELL":
                    quantity = min(quantity, shares[symbol])
                if quantity <= 0:
                    continue
                notional = quantity * exec_price
                commission = notional * config.commission_rate
                tax = notional * config.stamp_duty_rate if side == "SELL" else 0.0
                if side == "BUY":
                    affordable = cash / (exec_price * (1 + config.commission_rate))
                    quantity = min(quantity, affordable)
                    notional = quantity * exec_price
                    commission = notional * config.commission_rate
                    cash -= notional + commission
                    shares[symbol] += quantity
                else:
                    cash += notional - commission - tax
                    shares[symbol] -= quantity
                trades.append({
                    "date": current, "signal_date": previous_dates[-1], "symbol": symbol,
                    "side": side, "quantity": quantity, "price": exec_price,
                    "notional": notional, "commission": commission, "tax": tax,
                })
            last_desired = desired.copy()
        equity = cash + float((shares * close_px.fillna(open_px).fillna(0)).sum())
        equities.append(equity)
        value = shares * close_px.fillna(open_px).fillna(0)
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
