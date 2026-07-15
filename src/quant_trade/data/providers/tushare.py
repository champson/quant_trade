from __future__ import annotations

import time

import pandas as pd

from quant_trade.config import Secrets
from quant_trade.data.base import DataProvider, EmptyDataError, PermanentProviderError
from quant_trade.data.providers.common import normalize_daily, ymd
from quant_trade.models import AssetType, DataBatch, DataRequest, Dataset, Frequency


class TushareProvider(DataProvider):
    name = "tushare"

    def __init__(self, interval_seconds: float = 0.5, secrets: Secrets | None = None):
        self.interval_seconds = interval_seconds
        self.secrets = secrets or Secrets()
        self._pro = None

    def capabilities(self) -> set[Dataset]:
        return {Dataset.BARS, Dataset.DAILY_BASIC, Dataset.ADJ_FACTOR, Dataset.TRADE_CALENDAR}

    def supports(self, request: DataRequest) -> bool:
        return super().supports(request) and request.frequency == Frequency.DAY

    def _api(self):
        if self._pro is not None:
            return self._pro
        if not self.secrets.tushare_token:
            raise PermanentProviderError("未设置 TUSHARE_TOKEN")
        import tushare as ts

        ts.set_token(self.secrets.tushare_token)
        pro = ts.pro_api(self.secrets.tushare_token)
        if self.secrets.tushare_http_url:
            pro._DataApi__token = self.secrets.tushare_token
            pro._DataApi__http_url = self.secrets.tushare_http_url
        self._pro = pro
        return pro

    def fetch(self, request: DataRequest) -> DataBatch:
        if not self.supports(request):
            raise PermanentProviderError(f"Tushare 不支持请求: {request}")
        pro = self._api()
        if request.dataset == Dataset.TRADE_CALENDAR:
            df = pro.trade_cal(start_date=ymd(request.start), end_date=ymd(request.end))
            time.sleep(self.interval_seconds)
            return DataBatch(df, self.name, request)
        if request.dataset == Dataset.DAILY_BASIC:
            frames = []
            if request.symbols:
                for symbol in request.symbols:
                    frames.append(pro.daily_basic(ts_code=symbol, start_date=ymd(request.start), end_date=ymd(request.end)))
                    time.sleep(self.interval_seconds)
            else:
                frames.append(pro.daily_basic(trade_date=ymd(request.end)))
            df = pd.concat([x for x in frames if x is not None], ignore_index=True)
            return DataBatch(df, self.name, request)
        if request.dataset == Dataset.ADJ_FACTOR:
            frames = []
            for symbol in request.symbols:
                frames.append(pro.adj_factor(ts_code=symbol, start_date=ymd(request.start), end_date=ymd(request.end)))
                time.sleep(self.interval_seconds)
            df = pd.concat([x for x in frames if x is not None], ignore_index=True)
            return DataBatch(df, self.name, request)

        frames = []
        symbols = request.symbols
        if not symbols and request.asset_type in {AssetType.STOCK, AssetType.CONVERTIBLE_BOND}:
            raw = (
                pro.daily(trade_date=ymd(request.end))
                if request.asset_type == AssetType.STOCK
                else pro.cb_daily(trade_date=ymd(request.end))
            )
            time.sleep(self.interval_seconds)
            data = normalize_daily(
                raw, symbol="", provider=self.name,
                columns={"ts_code": "symbol", "vol": "volume"},
            )
            if data.empty:
                raise EmptyDataError("Tushare 返回空行情")
            return DataBatch(data.sort_values(["trade_date", "symbol"]), self.name, request)
        for symbol in symbols:
            kwargs = dict(ts_code=symbol, start_date=ymd(request.start), end_date=ymd(request.end))
            if request.asset_type == AssetType.ETF:
                raw = pro.fund_daily(**kwargs)
            elif request.asset_type == AssetType.INDEX:
                raw = pro.index_daily(**kwargs)
            elif request.asset_type == AssetType.CONVERTIBLE_BOND:
                raw = pro.cb_daily(**kwargs)
            else:
                raw = pro.daily(**kwargs)
            time.sleep(self.interval_seconds)
            frame = normalize_daily(
                raw,
                symbol=symbol,
                provider=self.name,
                columns={"ts_code": "symbol", "vol": "volume"},
            )
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            raise EmptyDataError("Tushare 返回空行情")
        return DataBatch(data.sort_values(["trade_date", "symbol"]), self.name, request)
