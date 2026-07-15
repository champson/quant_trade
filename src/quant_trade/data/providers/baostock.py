from __future__ import annotations

import time

import pandas as pd

from quant_trade.data.base import DataProvider, EmptyDataError, PermanentProviderError, ProviderError
from quant_trade.data.providers.common import normalize_daily
from quant_trade.models import AssetType, DataBatch, DataRequest, Dataset, Frequency


class BaoStockProvider(DataProvider):
    name = "baostock"

    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self._bs = None
        self._logged_in = False

    def capabilities(self) -> set[Dataset]:
        return {Dataset.BARS, Dataset.TRADE_CALENDAR}

    def supports(self, request: DataRequest) -> bool:
        return (
            super().supports(request)
            and request.frequency == Frequency.DAY
            and request.asset_type in {AssetType.STOCK, AssetType.INDEX}
        )

    @staticmethod
    def _code(symbol: str) -> str:
        code, _, exchange = symbol.partition(".")
        if exchange:
            return f"{exchange.lower()}.{code}"
        return f"{'sh' if code.startswith(('5', '6', '9')) else 'sz'}.{code}"

    def _api(self):
        if self._bs is None:
            import baostock as bs
            self._bs = bs
        if not self._logged_in:
            result = self._bs.login()
            if result.error_code != "0":
                raise ProviderError(f"BaoStock 登录失败: {result.error_msg}")
            self._logged_in = True
        return self._bs

    def fetch(self, request: DataRequest) -> DataBatch:
        if not self.supports(request):
            raise PermanentProviderError(f"BaoStock 不支持请求: {request}")
        bs = self._api()
        if request.dataset == Dataset.TRADE_CALENDAR:
            rs = bs.query_trade_dates(
                start_date=pd.Timestamp(request.start).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(request.end).strftime("%Y-%m-%d"),
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            return DataBatch(pd.DataFrame(rows, columns=rs.fields), self.name, request)
        frames = []
        fields = "date,code,open,high,low,close,volume,amount,adjustflag"
        adjustflag = {"none": "3", "qfq": "2", "hfq": "1"}.get(request.adjustment, "3")
        for symbol in request.symbols:
            rs = bs.query_history_k_data_plus(
                self._code(symbol), fields,
                start_date=pd.Timestamp(request.start).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(request.end).strftime("%Y-%m-%d"),
                frequency="d", adjustflag=adjustflag,
            )
            if rs.error_code != "0":
                raise ProviderError(rs.error_msg)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            raw = pd.DataFrame(rows, columns=rs.fields)
            frames.append(normalize_daily(
                raw, symbol=symbol, provider=self.name,
                columns={"date": "trade_date", "code": "provider_code"},
            ))
            time.sleep(self.interval_seconds)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            raise EmptyDataError("BaoStock 返回空行情")
        return DataBatch(data.sort_values(["trade_date", "symbol"]), self.name, request)

    def close(self) -> None:
        if self._bs is not None and self._logged_in:
            self._bs.logout()
            self._logged_in = False

