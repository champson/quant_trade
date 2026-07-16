from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trade.strategies.base import SignalResult, Strategy, StrategyMetadata


def _rebalance_periods(index: pd.DatetimeIndex, rebalance_days: int, anchor: str) -> np.ndarray:
    anchor_day = np.datetime64(pd.Timestamp(anchor).date(), "D")
    dates = index.to_numpy(dtype="datetime64[D]")
    return np.busday_count(anchor_day, dates) // rebalance_days


class EtfRotationStrategy(Strategy):
    metadata = StrategyMetadata("etf_rotation", "ETF 动量轮动", "多周期动量与均线趋势过滤")

    def _matrices(self, bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        prices = bars.pivot(index="trade_date", columns="symbol", values="close").sort_index()
        windows = tuple(self.config.get("momentum_windows", [20, 60]))
        score = sum(prices.pct_change(int(w), fill_method=None) for w in windows) / len(windows)
        ma = prices.rolling(int(self.config.get("ma_window", 28))).mean()
        eligible = (score > float(self.config.get("min_momentum", 0.0))) & (prices > ma)
        return prices, score, eligible

    def generate_targets(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices, scores, eligible = self._matrices(bars)
        hold_num = max(1, int(self.config.get("hold_num", 1)))
        rebalance_days = max(1, int(self.config.get("rebalance_days", 5)))
        periods = _rebalance_periods(
            prices.index,
            rebalance_days,
            str(self.config.get("rebalance_anchor", "2000-01-03")),
        )
        targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        current = pd.Series(0.0, index=prices.columns)
        previous_period: int | None = None
        for date, period in zip(prices.index, periods, strict=True):
            if previous_period is None or period != previous_period:
                candidates = scores.loc[date].where(eligible.loc[date]).dropna().nlargest(hold_num)
                current = pd.Series(0.0, index=prices.columns)
                if not candidates.empty:
                    current.loc[candidates.index] = 1.0 / len(candidates)
            targets.loc[date] = current
            previous_period = int(period)
        return targets

    def latest_signal(self, bars: pd.DataFrame) -> SignalResult:
        prices, scores, eligible = self._matrices(bars)
        result = super().latest_signal(bars)
        date = result.as_of
        diagnostics = pd.DataFrame(
            {
                "close": prices.loc[date],
                "momentum_score": scores.loc[date],
                "eligible": eligible.loc[date],
                "target_weight": result.target_weights,
            }
        ).sort_values("momentum_score", ascending=False)
        result.diagnostics = diagnostics
        return result
