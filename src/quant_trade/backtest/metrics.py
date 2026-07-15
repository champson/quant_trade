from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(equity: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    returns = equity.pct_change().dropna()
    years = max(len(returns) / periods_per_year, 1 / periods_per_year)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    rf_period = (1 + risk_free) ** (1 / periods_per_year) - 1
    sharpe = ((returns.mean() - rf_period) / returns.std(ddof=1) * np.sqrt(periods_per_year)) if returns.std(ddof=1) > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "total_return": float(total_return), "cagr": float(cagr),
        "annual_volatility": float(volatility), "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown), "calmar": float(calmar),
    }

