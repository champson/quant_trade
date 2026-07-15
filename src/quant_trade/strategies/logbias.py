from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trade.strategies.base import SignalResult, Strategy, StrategyMetadata


class LogBiasStrategy(Strategy):
    metadata = StrategyMetadata("logbias", "对数乖离策略", "价格对数相对 EMA 的偏离择时")

    def _indicator(self, prices: pd.DataFrame) -> pd.DataFrame:
        window = int(self.config.get("ema_window", 20))
        return (np.log(prices) - np.log(prices.ewm(span=window, adjust=False).mean())) * 100

    def generate_targets(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices = bars.pivot(index="trade_date", columns="symbol", values="close").sort_index()
        bias = self._indicator(prices)
        entry = float(self.config.get("entry", 5.0))
        overheat = float(self.config.get("overheat", 15.0))
        stop = float(self.config.get("stop", -5.0))
        targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        held = pd.Series(False, index=prices.columns)
        for date in prices.index:
            value = bias.loc[date]
            held = ((held & (value > stop)) | ((value >= entry) & (value < overheat))).fillna(False)
            selected = held[held].index
            if len(selected):
                targets.loc[date, selected] = 1.0 / len(selected)
        return targets

    def latest_signal(self, bars: pd.DataFrame) -> SignalResult:
        prices = bars.pivot(index="trade_date", columns="symbol", values="close").sort_index()
        bias = self._indicator(prices)
        result = super().latest_signal(bars)
        result.diagnostics = pd.DataFrame({
            "close": prices.loc[result.as_of], "logbias": bias.loc[result.as_of],
            "target_weight": result.target_weights,
        }).sort_values("logbias", ascending=False)
        return result

