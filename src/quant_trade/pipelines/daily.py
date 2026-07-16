from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant_trade.config import AppConfig
from quant_trade.data.calendar import trading_days
from quant_trade.data.minute_archive import MinuteArchiveImporter
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import Adjustment, AssetType
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


def _strategy_download_groups(
    strategies: dict,
) -> dict[tuple[AssetType, Adjustment], set[str]]:
    """Group strategy and benchmark symbols by their actual storage contract."""
    groups: dict[tuple[AssetType, Adjustment], set[str]] = {}
    for name, strategy in strategies.items():
        if not strategy.get("enabled"):
            continue
        asset = AssetType(strategy.get("asset_type", "stock" if name == "microcap" else "etf"))
        adjustment = Adjustment(strategy.get("adjustment", "none"))
        symbols = set(strategy.get("symbols", []))
        if symbols:
            groups.setdefault((asset, adjustment), set()).update(symbols)

        benchmark = strategy.get("benchmark")
        if benchmark:
            benchmark_asset = AssetType(strategy.get("benchmark_asset_type", asset.value))
            benchmark_adjustment = Adjustment(
                strategy.get(
                    "benchmark_adjustment",
                    "none" if benchmark_asset == AssetType.INDEX else adjustment.value,
                )
            )
            groups.setdefault((benchmark_asset, benchmark_adjustment), set()).add(str(benchmark))
    return groups


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
        calendar = trading_days(router, date(as_of.year - 1, 12, 1), as_of, store)
        snapshot_dates = _anchors(calendar, as_of)
        for snapshot in snapshot_dates:
            update_bars(config, router, store, [], snapshot, snapshot, AssetType.STOCK)
            if config.strategies.get("microcap", {}).get("enabled"):
                update_daily_basic(router, store, snapshot)
            try:
                update_bars(
                    config, router, store, [], snapshot, snapshot, AssetType.CONVERTIBLE_BOND
                )
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
                config,
                router,
                store,
                all_indices,
                min(snapshot_dates) - timedelta(days=80),
                as_of,
                AssetType.INDEX,
            )

        strategy_start = as_of - timedelta(days=420)
        for (asset_type, adjustment), symbols in _strategy_download_groups(
            config.strategies
        ).items():
            update_bars(
                config,
                router,
                store,
                sorted(symbols),
                strategy_start,
                as_of,
                asset_type,
                adjustment=adjustment.value,
            )

        imported = MinuteArchiveImporter(config, store).import_inbox()
        result.minute_imports = [r.__dict__ for r in imported]

        market = store.read_daily(
            [], None, str(as_of), asset_type=AssetType.STOCK, adjustment="none"
        )
        review = build_market_review(market, as_of)
        index_bars = (
            store.read_daily(
                index_codes,
                None,
                str(as_of),
                asset_type=AssetType.INDEX,
                adjustment="none",
            )
            if index_codes
            else pd.DataFrame()
        )
        index_ret = period_returns(index_bars, as_of) if not index_bars.empty else None
        if index_ret is not None:
            name_by_code = {v: k for k, v in (config.review.get("indices") or {}).items()}
            index_ret = index_ret.rename(index=name_by_code)
        portfolio = None
        portfolio_file = config.review.get("portfolio_file")
        if portfolio_file and Path(portfolio_file).exists():
            portfolio = portfolio_returns(market, pd.read_csv(portfolio_file, dtype=str), as_of)
        cb_bars = store.read_daily(
            [],
            None,
            str(as_of),
            asset_type=AssetType.CONVERTIBLE_BOND,
            adjustment="none",
        )
        cb_summary = None
        if not cb_bars.empty:
            cb_summary = asset_return_summary(cb_bars, as_of)
        bias = None
        if bias_codes:
            bias_bars = store.read_daily(
                bias_codes,
                None,
                str(as_of),
                asset_type=AssetType.INDEX,
                adjustment="none",
            )
            if not bias_bars.empty:
                bias = logbias_table(bias_bars, int(config.review.get("bias_ema_window", 20)))
        paths = save_market_review(
            review,
            config.paths.artifacts_dir / "reviews",
            index_returns=index_ret,
            portfolio=portfolio,
            convertible_summary=cb_summary,
            bias=bias,
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
