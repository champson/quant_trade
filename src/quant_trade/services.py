from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from quant_trade.backtest import ExecutionConfig, run_weight_backtest, save_backtest_report
from quant_trade.config import AppConfig
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType, DataRequest, Dataset, Frequency
from quant_trade.strategies import get_strategy


# A cache only covers the request when its first row is within the longest
# A-share holiday span of the requested start and the rows are dense enough to
# be a real daily series rather than scattered snapshot dates.
_HEAD_TOLERANCE_DAYS = 12
_MIN_TRADING_DENSITY = 0.6


def _cached_range_state(cached: pd.DataFrame, start: date, end: date) -> tuple[bool, date]:
    """Return (fully_covered, fetch_start) for a cached daily series."""
    days = pd.to_datetime(cached["trade_date"]).dt.date
    first, last = days.min(), days.max()
    span = pd.bdate_range(first, last)
    dense = len(span) == 0 or days.nunique() >= _MIN_TRADING_DENSITY * len(span)
    if not dense or (first - start).days > _HEAD_TOLERANCE_DAYS:
        return False, start
    if last >= end:
        return True, start
    return False, max(start, last + timedelta(days=1))


def update_bars(
    config: AppConfig,
    router: DataRouter,
    store: DataStore,
    symbols: list[str],
    start: date,
    end: date,
    asset_type: AssetType,
    provider: str = "auto",
    adjustment: str = "none",
    resume: bool = True,
) -> pd.DataFrame:
    groups = [[symbol] for symbol in symbols] if symbols else [[]]
    frames: list[pd.DataFrame] = []
    for group in groups:
        fetch_start = start
        if resume and not group and start == end and store.market_snapshot_complete(asset_type.value, end):
            continue
        if resume and group:
            cached = store.read_daily(group, str(start), str(end))
            if not cached.empty:
                covered, fetch_start = _cached_range_state(cached, start, end)
                if covered:
                    continue
        batch = router.fetch(DataRequest(
            dataset=Dataset.BARS, symbols=tuple(group), start=fetch_start, end=end,
            frequency=Frequency.DAY, asset_type=asset_type, provider=provider,
            adjustment=adjustment,
        ))
        store.write_daily(batch.data, asset_type.value)
        if not group and start == end:
            store.mark_market_snapshot(asset_type.value, end)
        frames.append(batch.data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_daily_basic(
    router: DataRouter, store: DataStore, trade_date: date, provider: str = "auto",
) -> pd.DataFrame:
    batch = router.fetch(DataRequest(
        dataset=Dataset.DAILY_BASIC, start=trade_date, end=trade_date, provider=provider,
    ))
    store.write_daily_basic(batch.data)
    return batch.data


def strategy_bars(store: DataStore, symbols: list[str], start: str | None, end: str | None) -> pd.DataFrame:
    if symbols:
        data = store.read_daily(symbols, start, end)
    else:
        paths = list((store.root / "daily" / AssetType.STOCK.value).glob("*.parquet"))
        data = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()
        if not data.empty:
            data["trade_date"] = pd.to_datetime(data["trade_date"])
            if start:
                data = data[data["trade_date"] >= pd.Timestamp(start)]
            if end:
                data = data[data["trade_date"] <= pd.Timestamp(end)]
    if data.empty:
        raise ValueError("本地没有策略所需行情，请先执行 qt data update")
    missing = sorted(set(symbols) - set(data["symbol"]))
    if missing:
        raise ValueError("本地缺少行情: " + ", ".join(missing))
    return data


def run_strategy_signal(config: AppConfig, store: DataStore, name: str, as_of: str | None = None):
    cfg = config.strategies.get(name, {})
    symbols = list(cfg.get("symbols", []))
    bars = strategy_bars(store, symbols, None, as_of)
    if name == "microcap":
        basic = store.read_daily_basic(None, as_of)
        bars = bars.merge(basic[["symbol", "trade_date", "total_mv"]], on=["symbol", "trade_date"], how="inner")
    strategy = get_strategy(name, cfg)
    result = strategy.latest_signal(bars)
    out_dir = config.paths.artifacts_dir / "signals" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.as_of.strftime("%Y%m%d")
    result.diagnostics.to_csv(out_dir / f"signal_{stamp}.csv", encoding="utf-8-sig")
    pd.Series({"as_of": str(result.as_of), "summary": result.summary}).to_json(
        out_dir / f"signal_{stamp}.json", force_ascii=False, indent=2
    )
    return result


def run_strategy_backtest(
    config: AppConfig, store: DataStore, name: str, start: str, end: str | None = None
):
    cfg = config.strategies.get(name, {})
    symbols = list(cfg.get("symbols", []))
    bars = strategy_bars(store, symbols, start, end)
    if name == "microcap":
        basic = store.read_daily_basic(start, end)
        bars = bars.merge(basic[["symbol", "trade_date", "total_mv"]], on=["symbol", "trade_date"], how="inner")
    strategy = get_strategy(name, cfg)
    targets = strategy.generate_targets(bars)
    bc = config.backtest
    execution = ExecutionConfig(
        initial_cash=bc.initial_cash, commission_rate=bc.commission_rate,
        stamp_duty_rate=bc.stamp_duty_rate, slippage_rate=bc.slippage_rate,
        risk_free_annual=bc.risk_free_annual,
    )
    result = run_weight_backtest(bars, targets, execution)
    out_dir = config.paths.artifacts_dir / "backtests" / name
    benchmark_name = cfg.get("benchmark")
    benchmark_equity = None
    if benchmark_name:
        benchmark_bars = store.read_daily([benchmark_name], start, end)
        if not benchmark_bars.empty:
            closes = benchmark_bars.sort_values("trade_date").set_index("trade_date")["close"]
            benchmark_equity = closes / closes.iloc[0] * execution.initial_cash
    report_paths = save_backtest_report(
        name=name,
        result=result,
        out_dir=out_dir,
        execution=execution,
        strategy_config=cfg,
        benchmark_equity=benchmark_equity,
        benchmark_name=benchmark_name,
    )
    result.artifacts = report_paths.as_dict()
    return result
