from __future__ import annotations

from datetime import date, datetime

import numpy as np
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


def _record_market_snapshot(
    config: AppConfig,
    store: DataStore,
    batch,
    asset_type: AssetType,
    trade_date: date,
    mode: Adjustment,
    *,
    preserve_existing_complete: bool = False,
) -> None:
    snapshot = batch.data.copy()
    snapshot["trade_date"] = pd.to_datetime(snapshot["trade_date"]).dt.date
    snapshot = snapshot[snapshot["trade_date"] == trade_date]
    row_count = len(snapshot)
    symbol_count = int(snapshot["symbol"].nunique())
    configured_min = int(config.providers.market_snapshot_min_symbols.get(asset_type.value, 0))
    basic_verified = (
        store.daily_basic_complete(trade_date) if asset_type == AssetType.STOCK else False
    )
    basic_count = store.daily_basic_symbol_count(trade_date) if asset_type == AssetType.STOCK else 0
    prior_count = store.latest_complete_snapshot_symbol_count(asset_type, mode, trade_date)
    ratio = config.providers.market_snapshot_reference_ratio
    reference_floor = max(1, int(prior_count * ratio)) if prior_count else 0
    basic_floor = max(1, int(basic_count * ratio)) if basic_count else 0
    provider_expected = int(batch.metadata.get("expected_symbols", 0))
    provider_floor = max(1, int(provider_expected * ratio)) if provider_expected else 0
    if basic_verified:
        expected_symbols = max(basic_floor, provider_floor)
    elif provider_expected:
        expected_symbols = provider_floor
    elif asset_type == AssetType.CONVERTIBLE_BOND:
        # The listed bond universe can contract quickly after clustered
        # redemptions; an old larger snapshot is not a safe lower bound.
        expected_symbols = configured_min
    elif prior_count:
        expected_symbols = reference_floor
    else:
        expected_symbols = configured_min
    complete = expected_symbols > 0 and symbol_count >= expected_symbols
    complete = store.mark_market_snapshot(
        asset_type,
        trade_date,
        mode,
        row_count=row_count,
        symbol_count=symbol_count,
        expected_symbols=expected_symbols,
        provider=batch.provider,
        status="complete" if complete else "incomplete",
        details={
            "configured_min": configured_min,
            "daily_basic_symbols": basic_count,
            "daily_basic_verified": basic_verified,
            "prior_complete_symbols": prior_count,
            "provider_expected_symbols": provider_expected,
            "provider_expected_floor": provider_floor,
            "provider_expected_source": batch.metadata.get("expected_symbols_source"),
        },
        symbols=sorted(snapshot["symbol"].dropna().astype(str).unique()),
        validation_sample_size=config.providers.market_snapshot_validation_sample_size,
        preserve_existing_complete=preserve_existing_complete,
    )
    if not complete:
        raise DataQualityError(
            f"全市场快照不完整：{symbol_count} 个证券，至少需要 {expected_symbols}；"
            + (
                "已保存数据，但保留原有完整快照标记"
                if preserve_existing_complete
                else "已保存数据但不会标记完成"
            )
        )


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
    if not symbols and start != end:
        raise ValueError("全市场行情更新只支持单个交易日")
    groups = [[symbol] for symbol in symbols] if symbols else [[]]
    frames: list[pd.DataFrame] = []
    expected = trading_days(router, start, end, store) if symbols else []
    for group in groups:
        existing_snapshot_complete = (
            not group and start == end and store.market_snapshot_complete(asset_type, end, mode)
        )
        if resume and existing_snapshot_complete:
            continue
        if not group:
            ranges = [(start, end, [start])]
        else:
            if mode == Adjustment.QFQ:
                cached = store.read_daily(group, None, None, asset_type=asset_type, adjustment=mode)
                if cached.empty:
                    refresh_start, refresh_end = start, end
                else:
                    cached_days = pd.to_datetime(cached["trade_date"]).dt.date
                    refresh_start = min(start, cached_days.min())
                    refresh_end = max(end, cached_days.max())
                expected_for_group = trading_days(router, refresh_start, refresh_end, store)
            else:
                cached = store.read_daily(
                    group, str(start), str(end), asset_type=asset_type, adjustment=mode
                )
                expected_for_group = expected
            actual = (
                set(pd.to_datetime(cached["trade_date"]).dt.date) if not cached.empty else set()
            )
            covered = actual | store.confirmed_empty_daily_dates(
                asset_type, mode, group[0], start, end
            )
            # QFQ is anchored to a moving reference date. Incrementally appending
            # a new tail can therefore mix incompatible price scales after a
            # corporate action, even when every row carries the same qfq label.
            # Re-fetch the entire requested window so every row shares one scale.
            reusable_coverage = resume and mode != Adjustment.QFQ
            ranges = _missing_ranges(expected_for_group, covered if reusable_coverage else set())
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
                if mode == Adjustment.QFQ:
                    raise DataQualityError(f"{group[0]} QFQ 全历史刷新返回空结果；保留原缓存")
                store.mark_daily_empty_dates(asset_type, mode, group[0], durable_days)
                continue
            if not batch.data.empty:
                if mode == Adjustment.QFQ:
                    returned = set(pd.to_datetime(batch.data["trade_date"]).dt.date)
                    cached_dates = (
                        set(pd.to_datetime(cached["trade_date"]).dt.date)
                        if not cached.empty
                        else set()
                    )
                    missing_cached = sorted(cached_dates - returned)
                    if missing_cached:
                        raise DataQualityError(
                            f"{group[0]} QFQ 刷新缺少原缓存中的 {len(missing_cached)} 个交易日，"
                            f"例如 {missing_cached[:5]}；保留原缓存"
                        )
                    store.replace_daily(batch.data, asset_type)
                else:
                    store.write_daily(batch.data, asset_type)
            if not group and start == end:
                _record_market_snapshot(
                    config,
                    store,
                    batch,
                    asset_type,
                    end,
                    mode,
                    preserve_existing_complete=existing_snapshot_complete,
                )
            frames.append(batch.data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_daily_basic(
    router: DataRouter,
    store: DataStore,
    trade_date: date,
    provider: str = "auto",
    reference_ratio: float = 0.9,
) -> pd.DataFrame:
    previous_complete = store.daily_basic_complete(trade_date)
    previous_symbols = store.daily_basic_symbols(trade_date) if previous_complete else set()
    batch = router.fetch(
        DataRequest(
            dataset=Dataset.DAILY_BASIC,
            start=trade_date,
            end=trade_date,
            provider=provider,
        )
    )
    if batch.data.empty:
        raise EmptyDataError("daily_basic 返回空结果")
    work = batch.data.rename(columns={"ts_code": "symbol"}).copy()
    required = {"symbol", "trade_date", "total_mv"}
    missing = required - set(work.columns)
    if missing:
        raise DataQualityError("daily_basic 缺少字段: " + ", ".join(sorted(missing)))
    work["symbol"] = work["symbol"].astype("string").str.strip()
    work["trade_date"] = pd.to_datetime(work["trade_date"].astype(str), errors="coerce")
    work["total_mv"] = pd.to_numeric(work["total_mv"], errors="coerce")
    invalid = (
        work[["symbol", "trade_date", "total_mv"]].isna().any(axis=1)
        | work["symbol"].eq("")
        | ~np.isfinite(work["total_mv"])
        | work["total_mv"].le(0)
    )
    if invalid.any():
        raise DataQualityError(f"daily_basic 有 {int(invalid.sum())} 条无法解析的记录")
    days = set(work["trade_date"].dt.date)
    if days != {trade_date}:
        raise DataQualityError(f"daily_basic 日期不匹配: {sorted(days)}，请求 {trade_date}")
    if work.duplicated(["symbol", "trade_date"]).any():
        raise DataQualityError("daily_basic 包含重复证券")
    symbol_count = int(work["symbol"].nunique())
    provider_expected = int(batch.metadata.get("expected_symbols", 0))
    provider_floor = max(1, int(provider_expected * reference_ratio)) if provider_expected else 0
    independently_complete = provider_floor > 0 and symbol_count >= provider_floor
    if previous_complete:
        missing_previous = sorted(previous_symbols - set(work["symbol"].astype(str)))
        if missing_previous:
            raise DataQualityError(
                "daily_basic 刷新结果缺少原完整快照中的证券，共 "
                f"{len(missing_previous)} 个，例如 {missing_previous[:5]}；保留旧版本"
            )
        if not independently_complete:
            raise DataQualityError("daily_basic 刷新结果缺少独立证券主表完整性证明；保留旧版本")
    store.write_daily_basic(work, replace_dates=True)
    store.mark_daily_basic_snapshot(
        trade_date,
        row_count=len(work),
        symbol_count=symbol_count,
        expected_symbols=provider_floor,
        provider=batch.provider,
        status="complete" if independently_complete else "observed",
        details={
            "provider_expected_symbols": provider_expected,
            "provider_expected_floor": provider_floor,
            "provider_expected_source": batch.metadata.get("expected_symbols_source"),
            "reference_ratio": reference_ratio,
            "symbol_digest": store.symbol_digest(set(work["symbol"].astype(str))),
        },
    )
    return work


def update_market_history(
    config: AppConfig,
    router: DataRouter,
    store: DataStore,
    days: list[date],
    *,
    include_basic: bool = True,
    force: bool = False,
    progress=None,
) -> int:
    """Backfill full-market snapshots with bounded, multi-day storage merges."""
    pending = []
    total_rows = 0
    incomplete_market = set(store.incomplete_market_snapshot_dates(AssetType.STOCK, days))
    complete_market = set(days) - incomplete_market

    def flush() -> None:
        nonlocal total_rows
        if not pending:
            return
        store.write_daily(pd.concat([batch.data for _, batch, _ in pending]), AssetType.STOCK)
        first_error: Exception | None = None
        for trade_date, batch, existing_complete in pending:
            try:
                _record_market_snapshot(
                    config,
                    store,
                    batch,
                    AssetType.STOCK,
                    trade_date,
                    Adjustment.NONE,
                    preserve_existing_complete=existing_complete,
                )
            except DataQualityError as exc:
                first_error = first_error or exc
            total_rows += len(batch.data)
        pending.clear()
        if first_error is not None:
            raise first_error

    for index, trade_date in enumerate(days, 1):
        if include_basic and (force or not store.daily_basic_complete(trade_date)):
            update_daily_basic(
                router,
                store,
                trade_date,
                reference_ratio=config.providers.market_snapshot_reference_ratio,
            )
        existing_complete = trade_date in complete_market
        if not force and existing_complete:
            if progress:
                progress(index, len(days), trade_date, 0)
            continue
        batch = router.fetch(
            DataRequest(
                dataset=Dataset.BARS,
                start=trade_date,
                end=trade_date,
                frequency=Frequency.DAY,
                asset_type=AssetType.STOCK,
                adjustment=Adjustment.NONE,
            )
        )
        pending.append((trade_date, batch, existing_complete))
        if progress:
            progress(index, len(days), trade_date, len(batch.data))
        if len(pending) >= config.providers.market_history_batch_days:
            flush()
    flush()
    if include_basic:
        incomplete = store.incomplete_daily_basic_dates(days)
        if incomplete:
            raise DataQualityError(
                f"daily_basic 缺失或不完整，共 {len(incomplete)} 天，例如 {incomplete[:5]}"
            )
    return total_rows


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


def _attach_microcap_basic(
    store: DataStore,
    bars: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    bar_days = sorted(pd.to_datetime(bars["trade_date"]).dt.date.unique())
    first_day = pd.Timestamp(start).date() if start else bar_days[0]
    last_day = pd.Timestamp(end).date() if end else bar_days[-1]
    if not store.calendar_range_complete(first_day, last_day):
        raise DataQualityError(
            f"微盘股策略所需交易日历不完整：{first_day} 至 {last_day}；"
            "请先执行 qt data market-history"
        )
    expected_days = store.read_trading_days(first_day, last_day)
    incomplete_market = store.incomplete_market_snapshot_dates(AssetType.STOCK, expected_days)
    if incomplete_market:
        examples = ", ".join(str(day) for day in incomplete_market[:5])
        raise DataQualityError(
            f"微盘股策略所需全市场快照缺失或不完整，共 {len(incomplete_market)} 天：{examples}"
        )
    incomplete_basic = store.incomplete_daily_basic_dates(expected_days)
    if incomplete_basic:
        examples = ", ".join(str(day) for day in incomplete_basic[:5])
        raise DataQualityError(
            f"微盘股策略所需 daily_basic 缺失或不完整，共 {len(incomplete_basic)} 天：{examples}"
        )
    basic = store.read_daily_basic(start, end)
    merged = bars.merge(
        basic[["symbol", "trade_date", "total_mv"]],
        on=["symbol", "trade_date"],
        how="left",
    )
    missing = merged["total_mv"].isna()
    if missing.any():
        examples = (
            merged.loc[missing, ["trade_date", "symbol"]].head(10).astype(str).to_dict("records")
        )
        raise DataQualityError(
            f"微盘股行情有 {int(missing.sum())} 条记录缺少 total_mv，例如 {examples}"
        )
    return merged


def run_strategy_signal(config: AppConfig, store: DataStore, name: str, as_of: str | None = None):
    cfg = config.strategies.get(name, {})
    symbols = list(cfg.get("symbols", []))
    if name != "microcap" and not symbols:
        raise ValueError(f"策略 {name} 未配置 symbols")
    asset_type = AssetType(cfg.get("asset_type", "stock" if name == "microcap" else "etf"))
    bars = strategy_bars(
        store, symbols, None, as_of, asset_type, str(cfg.get("adjustment", "none"))
    )
    if name == "microcap":
        bars = _attach_microcap_basic(store, bars, None, as_of)
    strategy = get_strategy(name, cfg)
    result = strategy.latest_signal(bars)
    if as_of is not None and result.as_of.normalize() != pd.Timestamp(as_of).normalize():
        raise DataQualityError(
            f"策略 {name} 的最新信号日期为 {result.as_of.date()}，请求日期为 "
            f"{pd.Timestamp(as_of).date()}；不会输出过期信号"
        )
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
    if name != "microcap" and not symbols:
        raise ValueError(f"策略 {name} 未配置 symbols")
    adjustment = str(cfg.get("adjustment", "none"))
    asset_type = AssetType(cfg.get("asset_type", "stock" if name == "microcap" else "etf"))
    bars = strategy_bars(store, symbols, start, end, asset_type, adjustment)
    if name == "microcap":
        bars = _attach_microcap_basic(store, bars, start, end)
    strategy = get_strategy(name, cfg)
    targets = strategy.generate_targets(bars)
    bc = config.backtest
    execution = ExecutionConfig(
        initial_cash=bc.initial_cash,
        commission_rate=bc.commission_rate,
        stamp_duty_rate=(bc.stamp_duty_rate if asset_type == AssetType.STOCK else 0.0),
        slippage_rate=bc.slippage_rate,
        lot_size=bc.lot_size,
        risk_free_annual=bc.risk_free_annual,
    )
    result = run_weight_backtest(bars, targets, execution)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    out_dir = config.paths.artifacts_dir / "backtests" / name / run_id
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
