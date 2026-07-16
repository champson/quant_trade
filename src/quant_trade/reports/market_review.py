from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


BUCKETS = [
    ("上涨>7%", 0.07, float("inf")),
    ("上涨5%-7%", 0.05, 0.07),
    ("上涨3%-5%", 0.03, 0.05),
    ("上涨0%-3%", 0.0, 0.03),
    ("平盘", 0.0, 0.0),
    ("下跌0%-3%", -0.03, 0.0),
    ("下跌3%-5%", -0.05, -0.03),
    ("下跌5%-7%", -0.07, -0.05),
    ("下跌>7%", -float("inf"), -0.07),
]


@dataclass
class MarketReview:
    as_of: pd.Timestamp
    breadth: pd.DataFrame
    summary: dict[str, float | int | str]
    returns: pd.DataFrame


def _nearest_on_or_before(dates: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    eligible = dates[dates <= target]
    if not len(eligible):
        raise ValueError(f"没有 {target.date()} 以前的数据")
    return eligible[-1]


def build_market_review(
    bars: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> MarketReview:
    if bars.empty:
        raise ValueError("没有市场行情可复盘")
    work = bars.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    close = work.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    target = pd.Timestamp(as_of) if as_of else close.index[-1]
    latest = _nearest_on_or_before(close.index, target)
    previous = _nearest_on_or_before(close.index, latest - pd.Timedelta(days=1))
    anchors = {
        "当天": previous,
        "本周": latest - pd.Timedelta(days=latest.weekday() + 1),
        "本月": latest.replace(day=1) - pd.Timedelta(days=1),
        "今年": latest.replace(month=1, day=1) - pd.Timedelta(days=1),
    }
    returns: dict[str, pd.Series] = {}
    for name, anchor in anchors.items():
        base = _nearest_on_or_before(close.index, anchor)
        returns[name] = close.loc[latest] / close.loc[base] - 1
    ret = pd.DataFrame(returns)
    rows = []
    for label, lower, upper in BUCKETS:
        row = {"区间": label}
        for period in ret.columns:
            values = ret[period].dropna()
            if lower == upper == 0.0:
                mask = values == 0.0
            elif lower == -float("inf"):
                mask = values <= upper
            elif upper == float("inf"):
                mask = values > lower
            elif upper == 0.0:
                mask = (values > lower) & (values < upper)
            else:
                mask = (values > lower) & (values <= upper)
            row[period] = int(mask.sum())
        rows.append(row)
    day = ret["当天"].dropna()
    summary = {
        "as_of": str(latest.date()),
        "stocks": int(len(day)),
        "up": int((day > 0).sum()),
        "down": int((day < 0).sum()),
        "flat": int((day == 0).sum()),
        "median_return": float(day.median()),
        "mean_return": float(day.mean()),
    }
    return MarketReview(latest, pd.DataFrame(rows), summary, ret)


def period_returns(bars: pd.DataFrame, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Return symbol x period returns using the same anchors as the market review."""
    return build_market_review(bars, as_of).returns


def portfolio_returns(
    bars: pd.DataFrame, portfolio: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> pd.Series:
    ret = period_returns(bars, as_of)
    holdings = portfolio.copy()
    code_col = "代码" if "代码" in holdings else "symbol"
    weight_col = "权重" if "权重" in holdings else "weight"

    def normalize(code: str) -> str:
        value = str(code).strip().split(".")[0].zfill(6)
        exchange = (
            "SH"
            if value.startswith(("5", "6", "9"))
            else "BJ"
            if value.startswith(("4", "8"))
            else "SZ"
        )
        return f"{value}.{exchange}"

    holdings["symbol"] = holdings[code_col].map(normalize)
    holdings["weight"] = pd.to_numeric(holdings[weight_col], errors="coerce").fillna(0)
    holdings["weight"] /= holdings["weight"].sum()
    available = holdings.set_index("symbol")["weight"].reindex(ret.index).dropna()
    if available.empty:
        return pd.Series(dtype=float)
    available /= available.sum()
    return ret.loc[available.index].mul(available, axis=0).sum().rename("portfolio")


def asset_return_summary(
    bars: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    ret = period_returns(bars, as_of)
    return pd.DataFrame({"等权平均": ret.mean(), "中位数": ret.median(), "数量": ret.count()}).T


def logbias_table(bars: pd.DataFrame, window: int = 20, days: int = 10) -> pd.DataFrame:
    import numpy as np

    work = bars.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    prices = work.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    bias = (np.log(prices) - np.log(prices.ewm(span=window, adjust=False).mean())) * 100
    return bias.tail(days).T
