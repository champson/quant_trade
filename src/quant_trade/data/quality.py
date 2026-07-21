from __future__ import annotations

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    pass


def validate_bars(df: pd.DataFrame, *, minute: bool = False) -> list[str]:
    required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
    if minute:
        required.add("bar_time")
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataQualityError(f"行情缺少字段: {', '.join(missing)}")
    if df.empty:
        raise DataQualityError("行情为空")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    key_columns = ["symbol", "trade_date"] + (["bar_time"] if minute else [])
    invalid_required = (
        df[key_columns].isna().any(axis=1)
        | numeric.isna().any(axis=1)
        | df["symbol"].astype(str).str.strip().eq("")
    )
    if invalid_required.any():
        raise DataQualityError(f"存在 {int(invalid_required.sum())} 条关键字段为空的行情")
    key = ["symbol", "bar_time"] if minute else ["symbol", "trade_date"]
    duplicated = int(df.duplicated(key).sum())
    if duplicated:
        raise DataQualityError(f"存在 {duplicated} 条重复行情")
    invalid_ohlc = (
        (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        | (numeric[["open", "high", "low", "close"]].min(axis=1) <= 0)
        | ~np.isfinite(numeric[["open", "high", "low", "close"]]).all(axis=1)
    )
    if invalid_ohlc.any():
        raise DataQualityError(f"存在 {int(invalid_ohlc.sum())} 条非法 OHLC")
    invalid_trade = (numeric["volume"] < 0) | ~np.isfinite(numeric["volume"])
    if "amount" in df:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        supplied = df["amount"].notna()
        invalid_trade |= supplied & (amount.isna() | (amount < 0) | ~np.isfinite(amount))
    if invalid_trade.any():
        raise DataQualityError(f"存在 {int(invalid_trade.sum())} 条负成交量或成交额")
    warnings: list[str] = []
    if "amount" in df and df["amount"].isna().mean() > 0.2:
        warnings.append("超过20%的成交额为空")
    return warnings
