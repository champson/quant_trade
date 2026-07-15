from __future__ import annotations

import pandas as pd

from quant_trade.strategies.base import Strategy, StrategyMetadata


class MicrocapStrategy(Strategy):
    metadata = StrategyMetadata("microcap", "微盘股等权", "以前一交易日市值选择最小股票并等权持有")

    def generate_targets(self, bars: pd.DataFrame) -> pd.DataFrame:
        if "total_mv" not in bars:
            raise ValueError("微盘股策略需要 point-in-time total_mv 字段")
        work = bars.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"])
        mv = work.pivot(index="trade_date", columns="symbol", values="total_mv").sort_index().shift(1)
        prices = work.pivot(index="trade_date", columns="symbol", values="close").reindex(mv.index)
        pool_size = int(self.config.get("pool_size", self.config.get("count", 400)))
        hold_count = int(self.config.get("hold_count", pool_size))
        selection = str(self.config.get("selection", "pool"))
        rank_start = max(1, int(self.config.get("rank_start", 1)))
        prefixes = tuple(str(x) for x in self.config.get("exclude_prefixes", ["688", "8", "4"]))
        rebalance = str(self.config.get("rebalance", "weekly"))
        targets = pd.DataFrame(0.0, index=mv.index, columns=mv.columns)
        current = pd.Series(0.0, index=mv.columns)
        for date in mv.index:
            if rebalance == "weekly":
                week = date.to_period("W")
                future_same_week = mv.index[(mv.index > date) & (mv.index.to_period("W") == week)]
                should_rebalance = len(future_same_week) == 0
            elif rebalance == "monthly":
                month = date.to_period("M")
                future_same_month = mv.index[(mv.index > date) & (mv.index.to_period("M") == month)]
                should_rebalance = len(future_same_month) == 0
            else:
                should_rebalance = True
            if should_rebalance:
                values = mv.loc[date].where(prices.loc[date].notna()).dropna()
                values = values[~values.index.to_series().str.split(".").str[0].str.startswith(prefixes)]
                pool = values.nsmallest(pool_size)
                if selection == "rank":
                    picked = pool.iloc[rank_start - 1 : rank_start - 1 + hold_count]
                elif selection == "rps":
                    lookback = int(self.config.get("rps_lookback_days", 120))
                    location = prices.index.get_loc(date)
                    if location < lookback:
                        picked = pd.Series(dtype=float)
                    else:
                        momentum = prices.loc[date, pool.index] / prices.iloc[location - lookback][pool.index] - 1
                        rps = momentum.rank(pct=True) * 100
                        target = float(self.config.get("rps_target", 70))
                        picked = pool.reindex((rps - target).abs().nsmallest(hold_count).index)
                elif selection == "smallest":
                    picked = pool.head(hold_count)
                else:
                    picked = pool
                current = pd.Series(0.0, index=mv.columns)
                if len(picked):
                    current.loc[picked.index] = 1 / len(picked)
            targets.loc[date] = current
        return targets
