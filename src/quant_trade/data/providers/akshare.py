from __future__ import annotations

import time

import pandas as pd

from quant_trade.data.base import DataProvider, EmptyDataError, PermanentProviderError
from quant_trade.data.providers.common import normalize_daily, ymd
from quant_trade.models import AssetType, DataBatch, DataRequest, Dataset, Frequency


class AkShareProvider(DataProvider):
    name = "akshare"

    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = interval_seconds

    def capabilities(self) -> set[Dataset]:
        return {Dataset.BARS}

    def supports(self, request: DataRequest) -> bool:
        return super().supports(request) and request.frequency == Frequency.DAY and request.asset_type in {
            AssetType.STOCK, AssetType.ETF, AssetType.INDEX
        }

    def fetch(self, request: DataRequest) -> DataBatch:
        if not self.supports(request):
            raise PermanentProviderError(f"AkShare 不支持请求: {request}")
        import akshare as ak

        frames = []
        adjust = "" if request.adjustment == "none" else request.adjustment
        for symbol in request.symbols:
            code = symbol.split(".")[0]
            if request.asset_type == AssetType.ETF:
                raw = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=ymd(request.start), end_date=ymd(request.end), adjust=adjust)
            elif request.asset_type == AssetType.INDEX:
                raw = ak.index_zh_a_hist(symbol=code, period="daily", start_date=ymd(request.start), end_date=ymd(request.end))
            else:
                raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=ymd(request.start), end_date=ymd(request.end), adjust=adjust)
            frames.append(normalize_daily(
                raw, symbol=symbol, provider=self.name,
                columns={"日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"},
            ))
            time.sleep(self.interval_seconds)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            raise EmptyDataError("AkShare 返回空行情")
        return DataBatch(data.sort_values(["trade_date", "symbol"]), self.name, request)
