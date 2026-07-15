from __future__ import annotations

import pandas as pd


class DataQualityError(ValueError):
    pass


def validate_bars(df: pd.DataFrame, *, minute: bool = False) -> list[str]:
    required = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataQualityError(f"行情缺少字段: {', '.join(missing)}")
    if df.empty:
        raise DataQualityError("行情为空")
    key = ["symbol", "bar_time"] if minute else ["symbol", "trade_date"]
    duplicated = int(df.duplicated(key).sum())
    if duplicated:
        raise DataQualityError(f"存在 {duplicated} 条重复行情")
    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[["open", "high", "low", "close"]].min(axis=1) <= 0)
    )
    if invalid_ohlc.any():
        raise DataQualityError(f"存在 {int(invalid_ohlc.sum())} 条非法 OHLC")
    warnings: list[str] = []
    if df["amount"].isna().mean() > 0.2 if "amount" in df else False:
        warnings.append("超过20%的成交额为空")
    return warnings

