from __future__ import annotations

import pandas as pd


def normalize_daily(
    df: pd.DataFrame,
    *,
    symbol: str,
    provider: str,
    columns: dict[str, str] | None = None,
    adjustment: str = "none",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.rename(columns=columns or {}).copy()
    if "symbol" not in out:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].fillna(symbol).astype(str)
    if "trade_date" not in out:
        raise ValueError("数据源未返回日期字段")
    out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
    for name in ("open", "high", "low", "close", "volume", "amount"):
        if name not in out:
            out[name] = pd.NA
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out["bar_time"] = pd.NaT
    out["source"] = provider
    out["adjustment"] = str(adjustment)
    return out[
        [
            "symbol",
            "trade_date",
            "bar_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "source",
            "adjustment",
        ]
    ].dropna(subset=["trade_date", "open", "high", "low", "close"])


def ymd(value) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y%m%d")
