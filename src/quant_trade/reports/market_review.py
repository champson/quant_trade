from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BUCKETS = [
    ("涨幅>10%", 0.10, float("inf")),
    ("涨幅>5%到10%", 0.05, 0.10),
    ("涨幅>0%到5%", 0.0, 0.05),
    ("涨幅>-5%到0%", -0.05, 0.0),
    ("涨幅>-10%到-5%", -0.10, -0.05),
    ("涨幅小于-10%", -float("inf"), -0.10),
]

PERIODS = ["今年", "本月", "本周", "当天"]


@dataclass
class MarketReview:
    as_of: pd.Timestamp
    breadth: pd.DataFrame
    summary: dict[str, float | int | str]
    returns: pd.DataFrame
    anchor_dates: dict[str, pd.Timestamp]


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
    resolved_anchors: dict[str, pd.Timestamp] = {}
    for name, anchor in anchors.items():
        base = _nearest_on_or_before(close.index, anchor)
        resolved_anchors[name] = base
        returns[name] = close.loc[latest] / close.loc[base] - 1
    ret = pd.DataFrame(returns)
    rows = []
    for label, lower, upper in BUCKETS:
        row = {"区间": label}
        for period in ret.columns:
            # Price division introduces binary noise around exact 3/5/7%
            # boundaries; normalize it before applying the documented bins.
            values = ret[period].dropna().round(12)
            if lower == -float("inf"):
                mask = values <= upper
            elif upper == float("inf"):
                mask = values > lower
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
    return MarketReview(latest, pd.DataFrame(rows), summary, ret, resolved_anchors)


def period_returns(bars: pd.DataFrame, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Return symbol x period returns using the same anchors as the market review."""
    return build_market_review(bars, as_of).returns


def normalize_stock_symbol(code: str) -> str:
    value = str(code).strip().split(".")[0].zfill(6)
    exchange = (
        "SH"
        if value.startswith(("5", "6", "9"))
        else "BJ"
        if value.startswith(("4", "8"))
        else "SZ"
    )
    return f"{value}.{exchange}"


def portfolio_returns(
    bars: pd.DataFrame, portfolio: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> pd.Series:
    ret = period_returns(bars, as_of)
    holdings = portfolio.copy()
    code_col = "代码" if "代码" in holdings else "symbol"
    weight_col = "权重" if "权重" in holdings else "weight"

    holdings["symbol"] = holdings[code_col].map(normalize_stock_symbol)
    holdings["weight"] = pd.to_numeric(holdings[weight_col], errors="coerce").fillna(0)
    missing = sorted(set(holdings["symbol"]) - set(ret.index))
    if missing:
        raise ValueError("组合持仓缺少复盘行情: " + ", ".join(missing))
    if holdings["weight"].sum() <= 0:
        raise ValueError("组合权重合计必须大于0")
    holdings["weight"] /= holdings["weight"].sum()
    available = holdings.set_index("symbol")["weight"].reindex(ret.index).dropna()
    if available.empty:
        return pd.Series(dtype=float)
    available /= available.sum()
    selected_returns = ret.loc[available.index]
    missing_periods = selected_returns.columns[selected_returns.isna().any()].tolist()
    if missing_periods:
        missing_symbols = sorted(
            set(selected_returns.index[selected_returns[missing_periods].isna().any(axis=1)])
        )
        raise ValueError(
            "组合持仓在部分收益锚点缺少行情: "
            + ", ".join(missing_symbols)
            + "；涉及周期: "
            + ", ".join(missing_periods)
        )
    return selected_returns.mul(available, axis=0).sum().rename("portfolio")


def asset_return_summary(
    bars: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    ret = period_returns(bars, as_of)
    return pd.DataFrame({"等权平均": ret.mean(), "中位数": ret.median(), "数量": ret.count()}).T


def price_bias(
    bars: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    window: int = 25,
) -> pd.Series:
    """Return close/simple-moving-average bias for every symbol on the target day."""
    if window <= 0:
        raise ValueError("BIAS 窗口必须大于0")
    if bars.empty:
        return pd.Series(dtype=float, name=f"BIAS{window}")
    work = bars.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    prices = work.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    target = pd.Timestamp(as_of) if as_of is not None else prices.index[-1]
    latest = _nearest_on_or_before(prices.index, target)
    average = prices.rolling(window=window, min_periods=window).mean()
    return (prices.loc[latest] / average.loc[latest] - 1).rename(f"BIAS{window}")


def nav_period_returns(
    values: pd.Series,
    anchor_dates: dict[str, pd.Timestamp],
    as_of: str | pd.Timestamp,
) -> pd.Series:
    """Calculate review-period returns from a dated NAV/equity series."""
    series = values.copy()
    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty or (series <= 0).any():
        raise ValueError("净值序列必须包含正数")
    latest_day = _nearest_on_or_before(series.index, pd.Timestamp(as_of))
    result = {}
    for period, anchor in anchor_dates.items():
        base_day = _nearest_on_or_before(series.index, anchor)
        result[period] = float(series.loc[latest_day] / series.loc[base_day] - 1)
    return pd.Series(result, name=values.name)


def build_daily_review_table(
    review: MarketReview,
    *,
    index_returns: pd.DataFrame | None = None,
    index_bias: pd.Series | None = None,
    convertible_summary: pd.DataFrame | None = None,
    underlying_summary: pd.DataFrame | None = None,
    microcap_returns: pd.Series | None = None,
    portfolio_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the display-ready daily review matrix requested by the dashboard."""
    columns = ["名称", *PERIODS, "BIAS25"]
    rows: list[dict[str, str]] = []

    def percentage(value) -> str:
        return "" if pd.isna(value) else f"{float(value):.2%}"

    def add(name: str, values=None, *, counts: bool = False, bias=np.nan) -> None:
        series = pd.Series(dtype=float) if values is None else pd.Series(values)
        row = {"名称": name, "BIAS25": percentage(bias)}
        for period in PERIODS:
            value = series.get(period, np.nan)
            row[period] = "" if pd.isna(value) else str(int(value)) if counts else percentage(value)
        rows.append(row)

    returns = review.returns.reindex(columns=PERIODS)
    total = returns.count()
    add("A股总数量", total, counts=True)
    add("A股上涨数量", (returns > 0).sum(), counts=True)
    add("A股下跌数量", (returns < 0).sum(), counts=True)
    add("A股零涨幅数量", (returns == 0).sum(), counts=True)
    rounded = returns.round(12)
    for label, lower, upper in BUCKETS:
        if lower == -float("inf"):
            selected = rounded <= upper
        elif upper == float("inf"):
            selected = rounded > lower
        else:
            selected = (rounded > lower) & (rounded <= upper)
        add(label, selected.sum().div(total).where(total.gt(0)))
    add("A股上涨比例", (returns > 0).sum().div(total).where(total.gt(0)))
    add("A股下跌比例", (returns < 0).sum().div(total).where(total.gt(0)))
    add("A股算术平均涨幅", returns.mean())
    add("A股涨幅中位数", returns.median())

    def add_summary(prefix: str, summary: pd.DataFrame | None) -> None:
        mean = (
            summary.loc["等权平均"] if summary is not None and "等权平均" in summary.index else None
        )
        median = (
            summary.loc["中位数"] if summary is not None and "中位数" in summary.index else None
        )
        add(f"{prefix}算术平均涨幅", mean)
        add(f"{prefix}涨幅中位数", median)

    add_summary("可转债", convertible_summary)
    add_summary("正股", underlying_summary)

    if index_returns is not None:
        for name, values in index_returns.iterrows():
            bias = index_bias.get(name, np.nan) if index_bias is not None else np.nan
            add(str(name), values, bias=bias)

    add("微盘股（自建等权）", microcap_returns)
    cb_mean = (
        convertible_summary.loc["等权平均"]
        if convertible_summary is not None and "等权平均" in convertible_summary.index
        else None
    )
    add("可转债等权", cb_mean)
    add("我的实盘合计", portfolio_returns)
    return pd.DataFrame(rows, columns=columns)


def logbias_table(bars: pd.DataFrame, window: int = 20, days: int = 10) -> pd.DataFrame:
    work = bars.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    prices = work.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    bias = (np.log(prices) - np.log(prices.ewm(span=window, adjust=False).mean())) * 100
    return bias.tail(days).T
