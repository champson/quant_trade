from __future__ import annotations

import time

import pandas as pd

from quant_trade.config import Secrets
from quant_trade.data.base import DataProvider, EmptyDataError, PermanentProviderError
from quant_trade.data.providers.common import normalize_daily, ymd
from quant_trade.models import Adjustment, AssetType, DataBatch, DataRequest, Dataset, Frequency


class TushareProvider(DataProvider):
    name = "tushare"

    def __init__(self, interval_seconds: float = 0.5, secrets: Secrets | None = None):
        self.interval_seconds = interval_seconds
        self.secrets = secrets or Secrets()
        self._pro = None
        self._stock_master: pd.DataFrame | None = None
        self._convertible_bond_master: pd.DataFrame | None = None

    def capabilities(self) -> set[Dataset]:
        return {Dataset.BARS, Dataset.DAILY_BASIC, Dataset.ADJ_FACTOR, Dataset.TRADE_CALENDAR}

    def supports(self, request: DataRequest) -> bool:
        if request.dataset == Dataset.BARS and request.adjustment != Adjustment.NONE:
            # HFQ is stable across incremental downloads: raw price * dated
            # factor. QFQ is anchored to each request's end date and would mix
            # incompatible scales in an append-only cache, so keep rejecting it.
            if request.adjustment != Adjustment.HFQ or request.asset_type not in {
                AssetType.STOCK,
                AssetType.ETF,
            }:
                return False
        return super().supports(request) and request.frequency == Frequency.DAY

    def _factor(self, pro, request: DataRequest, symbol: str) -> pd.DataFrame:
        kwargs = {
            "ts_code": symbol,
            "start_date": ymd(request.start),
            "end_date": ymd(request.end),
        }
        factor = (
            pro.fund_adj(**kwargs)
            if request.asset_type == AssetType.ETF
            else pro.adj_factor(**kwargs)
        )
        time.sleep(self.interval_seconds)
        if (
            factor is None
            or factor.empty
            or not {"trade_date", "adj_factor"} <= set(factor.columns)
        ):
            raise EmptyDataError(f"Tushare 未返回 {symbol} 的后复权因子")
        out = factor[["trade_date", "adj_factor"]].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
        out["adj_factor"] = pd.to_numeric(out["adj_factor"], errors="coerce")
        out["ts_code"] = symbol
        return out.dropna().drop_duplicates("trade_date", keep="last")[
            ["ts_code", "trade_date", "adj_factor"]
        ]

    def _apply_adjustment(
        self, pro, request: DataRequest, symbol: str, frame: pd.DataFrame
    ) -> pd.DataFrame:
        if request.adjustment == Adjustment.NONE or frame.empty:
            return frame
        factor = self._factor(pro, request, symbol)
        adjusted = frame.merge(
            factor[["trade_date", "adj_factor"]],
            on="trade_date",
            how="left",
            validate="one_to_one",
        )
        if adjusted["adj_factor"].isna().any() or (adjusted["adj_factor"] <= 0).any():
            raise EmptyDataError(f"Tushare {symbol} 后复权因子未覆盖全部行情日期")
        for column in ("open", "high", "low", "close"):
            adjusted[column] = adjusted[column] * adjusted["adj_factor"]
        return adjusted.drop(columns="adj_factor")

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

    @staticmethod
    def _normalize_convertible_bond_suspensions(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Convert Tushare's zero-OHLC suspended-bond placeholder into a flat bar."""
        if frame.empty:
            return frame, 0
        zero_price = frame[["open", "high", "low"]].eq(0).all(axis=1)
        zero_trade = frame["volume"].fillna(0).eq(0) & frame["amount"].fillna(0).eq(0)
        suspended = zero_price & zero_trade & frame["close"].gt(0)
        count = int(suspended.sum())
        if count:
            frame = frame.copy()
            for column in ("open", "high", "low"):
                frame.loc[suspended, column] = frame.loc[suspended, "close"]
        return frame, count

    def _expected_stock_symbols(self, pro, trade_date) -> tuple[int, str | None]:
        """Return the independently derived active A-share universe size."""
        try:
            if self._stock_master is None:
                frames = []
                for status in ("L", "D", "P"):
                    frame = pro.stock_basic(
                        exchange="",
                        list_status=status,
                        fields="ts_code,list_status,list_date,delist_date",
                    )
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                    time.sleep(self.interval_seconds)
                self._stock_master = (
                    pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="last")
                    if frames
                    else pd.DataFrame()
                )
            master = self._stock_master.copy()
            if master.empty:
                return 0, "Tushare stock_basic 返回空结果，未生成证券主表参考"
            day = pd.Timestamp(trade_date).normalize()
            listed = pd.to_datetime(master["list_date"], errors="coerce")
            delisted = pd.to_datetime(master["delist_date"], errors="coerce")
            active = listed.le(day) & (delisted.isna() | delisted.ge(day))
            return int(master.loc[active, "ts_code"].nunique()), None
        except Exception as exc:
            return 0, f"证券主表参考获取失败: {exc}"

    def _expected_convertible_bond_symbols(self, pro, trade_date) -> tuple[int, str | None]:
        """Return the independently derived listed convertible-bond universe size."""
        try:
            if self._convertible_bond_master is None:
                self._convertible_bond_master = pro.cb_basic(fields="ts_code,list_date,delist_date")
                time.sleep(self.interval_seconds)
            master = self._convertible_bond_master.copy()
            required = {"ts_code", "list_date", "delist_date"}
            if master.empty or not required <= set(master.columns):
                return 0, "Tushare cb_basic 返回空结果或缺少日期字段"
            day = pd.Timestamp(trade_date).normalize()
            listed = pd.to_datetime(master["list_date"], errors="coerce")
            delisted = pd.to_datetime(master["delist_date"], errors="coerce")
            active = listed.le(day) & (delisted.isna() | delisted.ge(day))
            return int(master.loc[active, "ts_code"].nunique()), None
        except Exception as exc:
            return 0, f"可转债主表参考获取失败: {exc}"

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
                    frames.append(
                        pro.daily_basic(
                            ts_code=symbol, start_date=ymd(request.start), end_date=ymd(request.end)
                        )
                    )
                    time.sleep(self.interval_seconds)
            else:
                frames.append(pro.daily_basic(trade_date=ymd(request.end)))
            valid = [x for x in frames if x is not None and not x.empty]
            df = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
            metadata: dict[str, int | str] = {}
            warnings: list[str] = []
            if not request.symbols:
                expected, warning = self._expected_stock_symbols(pro, request.end)
                metadata["expected_symbols"] = expected
                metadata["expected_symbols_source"] = "tushare.stock_basic"
                if warning:
                    warnings.append(warning)
            return DataBatch(
                df,
                self.name,
                request,
                warnings=warnings,
                metadata=metadata,
            )
        if request.dataset == Dataset.ADJ_FACTOR:
            frames = []
            for symbol in request.symbols:
                frames.append(self._factor(pro, request, symbol))
            valid = [x for x in frames if x is not None and not x.empty]
            df = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
            return DataBatch(df, self.name, request)

        frames = []
        normalized_suspensions = 0
        symbols = request.symbols
        if not symbols and request.asset_type in {AssetType.STOCK, AssetType.CONVERTIBLE_BOND}:
            if pd.Timestamp(request.start).date() != pd.Timestamp(request.end).date():
                raise PermanentProviderError("Tushare 全市场接口只支持单个交易日")
            raw = (
                pro.daily(trade_date=ymd(request.end))
                if request.asset_type == AssetType.STOCK
                else pro.cb_daily(trade_date=ymd(request.end))
            )
            time.sleep(self.interval_seconds)
            data = normalize_daily(
                raw,
                symbol="",
                provider=self.name,
                columns={"ts_code": "symbol", "vol": "volume"},
                adjustment=request.adjustment,
            )
            if request.asset_type == AssetType.CONVERTIBLE_BOND:
                data, normalized_suspensions = self._normalize_convertible_bond_suspensions(data)
            if data.empty:
                raise EmptyDataError("Tushare 返回空行情")
            metadata = {"adjustment_evidence": "unadjusted_endpoint"}
            warnings = []
            if normalized_suspensions:
                warnings.append(f"已将 {normalized_suspensions} 条零成交停牌可转债规范为平盘行情")
            if request.asset_type == AssetType.STOCK:
                expected, warning = self._expected_stock_symbols(pro, request.end)
                metadata["expected_symbols"] = expected
                metadata["expected_symbols_source"] = "tushare.stock_basic"
                if warning:
                    warnings.append(warning)
            elif request.asset_type == AssetType.CONVERTIBLE_BOND:
                expected, warning = self._expected_convertible_bond_symbols(pro, request.end)
                metadata["expected_symbols"] = expected
                metadata["expected_symbols_source"] = "tushare.cb_basic"
                if warning:
                    warnings.append(warning)
            return DataBatch(
                data.sort_values(["trade_date", "symbol"]),
                self.name,
                request,
                warnings=warnings,
                metadata=metadata,
            )
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
                adjustment=request.adjustment,
            )
            if request.asset_type == AssetType.CONVERTIBLE_BOND:
                frame, normalized = self._normalize_convertible_bond_suspensions(frame)
                normalized_suspensions += normalized
            frame = self._apply_adjustment(pro, request, symbol, frame)
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            raise EmptyDataError("Tushare 返回空行情")
        return DataBatch(
            data.sort_values(["trade_date", "symbol"]),
            self.name,
            request,
            warnings=(
                [f"已将 {normalized_suspensions} 条零成交停牌可转债规范为平盘行情"]
                if normalized_suspensions
                else []
            ),
            metadata={
                "adjustment_evidence": (
                    "tushare_dated_factor"
                    if request.adjustment == Adjustment.HFQ
                    else "unadjusted_endpoint"
                )
            },
        )
