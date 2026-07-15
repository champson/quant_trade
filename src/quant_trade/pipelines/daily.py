from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant_trade.config import AppConfig
from quant_trade.data.minute_archive import MinuteArchiveImporter
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import AssetType, DataRequest, Dataset
from quant_trade.notifications import notify
from quant_trade.reports.market_review import (
    asset_return_summary,
    build_market_review,
    logbias_table,
    period_returns,
    portfolio_returns,
)
from quant_trade.reports.render import save_market_review
from quant_trade.runs import RunTracker
from quant_trade.services import run_strategy_signal, update_bars, update_daily_basic


@dataclass
class DailyResult:
    as_of: date
    report_paths: dict[str, str] = field(default_factory=dict)
    signals: dict[str, str] = field(default_factory=dict)
    minute_imports: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def trading_days(router: DataRouter, start: date, end: date) -> list[date]:
    batch = router.fetch(DataRequest(dataset=Dataset.TRADE_CALENDAR, start=start, end=end))
    df = batch.data.copy()
    if "cal_date" in df:
        mask = pd.to_numeric(df.get("is_open", 0), errors="coerce").fillna(0).astype(int) == 1
        values = df.loc[mask, "cal_date"]
    elif "calendar_date" in df:
        mask = df.get("is_trading_day", "0").astype(str) == "1"
        values = df.loc[mask, "calendar_date"]
    else:
        raise ValueError("交易日历字段无法识别")
    return sorted(pd.to_datetime(values).dt.date.tolist())


def _anchors(days: list[date], as_of: date) -> list[date]:
    open_days = [d for d in days if d <= as_of]
    if not open_days or open_days[-1] != as_of:
        raise ValueError(f"{as_of} 不是交易日或交易日历尚未更新")
    previous = open_days[-2]
    targets = [
        previous,
        as_of - timedelta(days=as_of.weekday() + 1),
        as_of.replace(day=1) - timedelta(days=1),
        as_of.replace(month=1, day=1) - timedelta(days=1),
    ]
    result = [as_of]
    for target in targets:
        eligible = [d for d in days if d <= target]
        if eligible:
            result.append(eligible[-1])
    return sorted(set(result))


def run_daily(config: AppConfig, router: DataRouter, store: DataStore, as_of: date) -> DailyResult:
    tracker = RunTracker(config, store, "daily", str(as_of))
    result = DailyResult(as_of)
    try:
        calendar = trading_days(router, date(as_of.year - 1, 12, 1), as_of)
        snapshot_dates = _anchors(calendar, as_of)
        for snapshot in snapshot_dates:
            update_bars(config, router, store, [], snapshot, snapshot, AssetType.STOCK)
            if config.strategies.get("microcap", {}).get("enabled"):
                update_daily_basic(router, store, snapshot)
            try:
                update_bars(config, router, store, [], snapshot, snapshot, AssetType.CONVERTIBLE_BOND)
            except Exception as exc:
                result.warnings.append(f"可转债快照失败 {snapshot}: {exc}")

        index_codes = list((config.review.get("indices") or {}).values())
        bias_file = config.review.get("bias_symbols_file")
        bias_codes: list[str] = []
        if bias_file and Path(bias_file).exists():
            bias_pool = pd.read_csv(bias_file, dtype=str).fillna("")
            code_col = "代码" if "代码" in bias_pool else bias_pool.columns[-1]
            bias_codes = bias_pool[code_col].str.strip().loc[lambda x: x.ne("")].tolist()
        all_indices = list(dict.fromkeys(index_codes + bias_codes))
        if all_indices:
            update_bars(
                config, router, store, all_indices,
                min(snapshot_dates) - timedelta(days=80), as_of, AssetType.INDEX,
            )

        strategy_start = as_of - timedelta(days=420)
        symbols_by_type: dict[AssetType, set[str]] = {AssetType.ETF: set()}
        for name, cfg in config.strategies.items():
            if cfg.get("enabled") and name != "microcap":
                symbols_by_type[AssetType.ETF].update(cfg.get("symbols", []))
        for asset_type, symbols in symbols_by_type.items():
            if symbols:
                update_bars(config, router, store, sorted(symbols), strategy_start, as_of, asset_type, adjustment="hfq")

        imported = MinuteArchiveImporter(config, store).import_inbox()
        result.minute_imports = [r.__dict__ for r in imported]

        market = store.read_daily([], None, str(as_of))
        # read_daily([]) intentionally returns empty; enumerate stored stock snapshots.
        stock_paths = list((store.root / "daily" / AssetType.STOCK.value).glob("*.parquet"))
        if stock_paths:
            market = pd.concat([pd.read_parquet(path) for path in stock_paths], ignore_index=True)
        review = build_market_review(market, as_of)
        index_bars = store.read_daily(index_codes, None, str(as_of)) if index_codes else pd.DataFrame()
        index_ret = period_returns(index_bars, as_of) if not index_bars.empty else None
        if index_ret is not None:
            name_by_code = {v: k for k, v in (config.review.get("indices") or {}).items()}
            index_ret = index_ret.rename(index=name_by_code)
        portfolio = None
        portfolio_file = config.review.get("portfolio_file")
        if portfolio_file and Path(portfolio_file).exists():
            portfolio = portfolio_returns(market, pd.read_csv(portfolio_file, dtype=str), as_of)
        cb_paths = list((store.root / "daily" / AssetType.CONVERTIBLE_BOND.value).glob("*.parquet"))
        cb_summary = None
        if cb_paths:
            cb_bars = pd.concat([pd.read_parquet(path) for path in cb_paths], ignore_index=True)
            cb_summary = asset_return_summary(cb_bars, as_of)
        bias = None
        if bias_codes:
            bias_bars = store.read_daily(bias_codes, None, str(as_of))
            if not bias_bars.empty:
                bias = logbias_table(bias_bars, int(config.review.get("bias_ema_window", 20)))
        paths = save_market_review(
            review, config.paths.artifacts_dir / "reviews",
            index_returns=index_ret, portfolio=portfolio,
            convertible_summary=cb_summary, bias=bias,
        )
        result.report_paths = {k: str(v) for k, v in paths.items()}

        for name, cfg in config.strategies.items():
            if cfg.get("enabled"):
                signal = run_strategy_signal(config, store, name, str(as_of))
                result.signals[name] = signal.summary
        tracker.finish("success", result.__dict__)
        notify("Quant Trade 复盘完成", f"{as_of}：{len(result.signals)} 个策略已更新")
        return result
    except Exception as exc:
        tracker.finish("failed", {"error": str(exc), "partial": result.__dict__})
        notify("Quant Trade 复盘失败", str(exc))
        raise
