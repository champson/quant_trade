from __future__ import annotations

from datetime import date

import pandas as pd

from quant_trade.backtest import ExecutionConfig, run_weight_backtest, save_backtest_report
from quant_trade.config import AppConfig
from quant_trade.data.calendar import trading_days
from quant_trade.data.base import EmptyDataError
from quant_trade.data.quality import DataQualityError
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import Adjustment, AssetType, DataRequest, Dataset, Frequency
from quant_trade.strategies import get_strategy


def _missing_ranges(
    expected: list[date], covered: set[date]
) -> list[tuple[date, date, list[date]]]:
    missing_positions = [index for index, day in enumerate(expected) if day not in covered]
    if not missing_positions:
        return []
    groups: list[list[int]] = [[missing_positions[0]]]
    for position in missing_positions[1:]:
        if position == groups[-1][-1] + 1:
            groups[-1].append(position)
        else:
            groups.append([position])
    return [
        (expected[group[0]], expected[group[-1]], [expected[index] for index in group])
        for group in groups
    ]


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
    mode = Adjustment(adjustment)
    end = min(end, date.today())
    if start > end:
        return pd.DataFrame()
    groups = [[symbol] for symbol in symbols] if symbols else [[]]
    frames: list[pd.DataFrame] = []
    expected = trading_days(router, start, end, store) if symbols else []
    for group in groups:
        if (
            resume
            and not group
            and start == end
            and store.market_snapshot_complete(asset_type, end, mode)
        ):
            continue
        if not group:
            ranges = [(start, end, [start])]
        else:
            cached = store.read_daily(
                group, str(start), str(end), asset_type=asset_type, adjustment=mode
            )
            actual = (
                set(pd.to_datetime(cached["trade_date"]).dt.date) if not cached.empty else set()
            )
            covered = actual | store.confirmed_empty_daily_dates(
                asset_type, mode, group[0], start, end
            )
            ranges = _missing_ranges(expected, covered if resume else set())
        for fetch_start, fetch_end, covered_days in ranges:
            # Today's bar may not be published yet when fetching intraday;
            # keep it uncovered so the next run fetches it again.
            durable_days = [day for day in covered_days if day < date.today()]
            try:
                batch = router.fetch(
                    DataRequest(
                        dataset=Dataset.BARS,
                        symbols=tuple(group),
                        start=fetch_start,
                        end=fetch_end,
                        frequency=Frequency.DAY,
                        asset_type=asset_type,
                        provider=provider,
                        adjustment=mode,
                    )
                )
            except EmptyDataError:
                if not group:
                    raise
                store.mark_daily_empty_dates(asset_type, mode, group[0], durable_days)
                continue
            if not batch.data.empty:
                store.write_daily(batch.data, asset_type)
            if not group and start == end:
                snapshot = batch.data.copy()
                snapshot["trade_date"] = pd.to_datetime(snapshot["trade_date"]).dt.date
                snapshot = snapshot[snapshot["trade_date"] == end]
                row_count = len(snapshot)
                symbol_count = int(snapshot["symbol"].nunique())
                configured_min = int(
                    config.providers.market_snapshot_min_symbols.get(asset_type.value, 0)
                )
                basic_count = (
                    store.daily_basic_symbol_count(end) if asset_type == AssetType.STOCK else 0
                )
                prior_count = store.latest_complete_snapshot_symbol_count(asset_type, mode, end)
                reference_floor = int(
                    prior_count * config.providers.market_snapshot_reference_ratio
                )
                basic_floor = int(basic_count * config.providers.market_snapshot_reference_ratio)
                provider_expected = int(batch.metadata.get("expected_symbols", 0))
                expected_symbols = max(
                    configured_min, basic_floor, reference_floor, provider_expected
                )
                complete = expected_symbols > 0 and symbol_count >= expected_symbols
                status = "complete" if complete else "incomplete"
                store.mark_market_snapshot(
                    asset_type,
                    end,
                    mode,
                    row_count=row_count,
                    symbol_count=symbol_count,
                    expected_symbols=expected_symbols,
                    provider=batch.provider,
                    status=status,
                    details={
                        "configured_min": configured_min,
                        "daily_basic_symbols": basic_count,
                        "prior_complete_symbols": prior_count,
                        "provider_expected_symbols": provider_expected,
                    },
                )
                if not complete:
                    raise DataQualityError(
                        f"全市场快照不完整：{symbol_count} 个证券，至少需要 {expected_symbols}；"
                        "已保存数据但不会标记完成"
                    )
            frames.append(batch.data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_daily_basic(
    router: DataRouter,
    store: DataStore,
    trade_date: date,
    provider: str = "auto",
) -> pd.DataFrame:
    batch = router.fetch(
        DataRequest(
            dataset=Dataset.DAILY_BASIC,
            start=trade_date,
            end=trade_date,
            provider=provider,
        )
    )
    store.write_daily_basic(batch.data)
    return batch.data


def strategy_bars(
    store: DataStore,
    symbols: list[str],
    start: str | None,
    end: str | None,
    asset_type: AssetType,
    adjustment: str = "none",
) -> pd.DataFrame:
    data = store.read_daily(symbols, start, end, asset_type=asset_type, adjustment=adjustment)
    if data.empty:
        raise ValueError("本地没有策略所需行情，请先执行 qt data update")
    missing = sorted(set(symbols) - set(data["symbol"]))
    if missing:
        raise ValueError("本地缺少行情: " + ", ".join(missing))
    return data


def run_strategy_signal(config: AppConfig, store: DataStore, name: str, as_of: str | None = None):
    cfg = config.strategies.get(name, {})
    symbols = list(cfg.get("symbols", []))
    asset_type = AssetType(cfg.get("asset_type", "stock" if name == "microcap" else "etf"))
    bars = strategy_bars(
        store, symbols, None, as_of, asset_type, str(cfg.get("adjustment", "none"))
    )
    if name == "microcap":
        basic = store.read_daily_basic(None, as_of)
        bars = bars.merge(
            basic[["symbol", "trade_date", "total_mv"]], on=["symbol", "trade_date"], how="inner"
        )
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
    adjustment = str(cfg.get("adjustment", "none"))
    asset_type = AssetType(cfg.get("asset_type", "stock" if name == "microcap" else "etf"))
    bars = strategy_bars(store, symbols, start, end, asset_type, adjustment)
    if name == "microcap":
        basic = store.read_daily_basic(start, end)
        bars = bars.merge(
            basic[["symbol", "trade_date", "total_mv"]], on=["symbol", "trade_date"], how="inner"
        )
    strategy = get_strategy(name, cfg)
    targets = strategy.generate_targets(bars)
    bc = config.backtest
    execution = ExecutionConfig(
        initial_cash=bc.initial_cash,
        commission_rate=bc.commission_rate,
        stamp_duty_rate=bc.stamp_duty_rate,
        slippage_rate=bc.slippage_rate,
        risk_free_annual=bc.risk_free_annual,
    )
    result = run_weight_backtest(bars, targets, execution)
    out_dir = config.paths.artifacts_dir / "backtests" / name
    benchmark_name = cfg.get("benchmark")
    benchmark_equity = None
    if benchmark_name:
        benchmark_asset_type = AssetType(cfg.get("benchmark_asset_type", asset_type.value))
        benchmark_adjustment = Adjustment(
            cfg.get(
                "benchmark_adjustment",
                "none" if benchmark_asset_type == AssetType.INDEX else adjustment,
            )
        )
        benchmark_bars = store.read_daily(
            [benchmark_name],
            start,
            end,
            asset_type=benchmark_asset_type,
            adjustment=benchmark_adjustment,
        )
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
