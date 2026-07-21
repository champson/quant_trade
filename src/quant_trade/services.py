from __future__ import annotations

import shutil
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_trade.backtest import ExecutionConfig, run_weight_backtest, save_backtest_report
from quant_trade.config import AppConfig
from quant_trade.data.calendar import trading_days
from quant_trade.data.base import EmptyDataError, ProviderError
from quant_trade.data.quality import DataQualityError
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import Adjustment, AssetType, DataRequest, Dataset, Frequency
from quant_trade.strategies import get_strategy
from quant_trade.strategies.base import SignalResult


def reference_symbol_floor(expected: int, ratio: float, tolerance_symbols: int = 0) -> int:
    """Turn a reference universe size into a robust minimum response size."""
    if expected <= 0:
        return 0
    return max(1, int(expected * ratio) - tolerance_symbols)


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
    snapshot_symbols = set(snapshot["symbol"].dropna().astype(str))
    row_count = len(snapshot)
    symbol_count = len(snapshot_symbols)
    configured_min = int(config.providers.market_snapshot_min_symbols.get(asset_type.value, 0))
    basic_verified = (
        store.daily_basic_complete(trade_date) if asset_type == AssetType.STOCK else False
    )
    basic_count = store.daily_basic_symbol_count(trade_date) if asset_type == AssetType.STOCK else 0
    basic_symbols = store.daily_basic_symbols(trade_date) if basic_verified else set()
    missing_basic_symbols = sorted(basic_symbols - snapshot_symbols)
    prior_count = store.latest_complete_snapshot_symbol_count(asset_type, mode, trade_date)
    ratio = config.providers.market_snapshot_reference_ratio
    tolerance = config.providers.market_snapshot_reference_tolerance_symbols
    reference_floor = reference_symbol_floor(prior_count, ratio, tolerance)
    basic_floor = reference_symbol_floor(basic_count, ratio, tolerance)
    provider_expected = int(batch.metadata.get("expected_symbols", 0))
    provider_floor = reference_symbol_floor(provider_expected, ratio, tolerance)
    response_complete = batch.metadata.get("full_market_response_complete")
    response_limit = int(batch.metadata.get("full_market_response_limit", 0))
    if response_complete is True:
        # Tushare's single-date endpoint returns the whole result below its
        # documented row limit. Suspended stocks intentionally have no daily
        # bar, so the active security master is not an exact row-count target.
        expected_symbols = basic_count if basic_verified else symbol_count
    elif response_complete is False:
        # Hitting the source row limit means the response may be truncated.
        expected_symbols = max(provider_floor, response_limit + 1, symbol_count + 1)
    elif basic_verified:
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
    complete = (
        expected_symbols > 0 and symbol_count >= expected_symbols and not missing_basic_symbols
    )
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
            "reference_tolerance_symbols": tolerance,
            "full_market_response_complete": response_complete,
            "full_market_response_rows": batch.metadata.get("full_market_response_rows"),
            "full_market_response_limit": response_limit,
            "missing_daily_basic_symbols": len(missing_basic_symbols),
            "missing_daily_basic_examples": missing_basic_symbols[:10],
        },
        symbols=sorted(snapshot_symbols),
        validation_sample_size=config.providers.market_snapshot_validation_sample_size,
        preserve_existing_complete=preserve_existing_complete,
    )
    if not complete:
        if missing_basic_symbols:
            raise DataQualityError(
                "全市场快照未覆盖完整 daily_basic：缺少 "
                f"{len(missing_basic_symbols)} 个证券，例如 {missing_basic_symbols[:5]}；"
                + (
                    "已保存数据，但保留原有完整快照标记"
                    if preserve_existing_complete
                    else "已保存数据但不会标记完成"
                )
            )
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
    reference_tolerance_symbols: int = 0,
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
    provider_floor = reference_symbol_floor(
        provider_expected, reference_ratio, reference_tolerance_symbols
    )
    response_complete = batch.metadata.get("full_market_response_complete")
    response_limit = int(batch.metadata.get("full_market_response_limit", 0))
    if response_complete is not None:
        independently_complete = response_complete is True
        expected_symbols = (
            symbol_count
            if independently_complete
            else max(provider_floor, response_limit + 1, symbol_count + 1)
        )
    else:
        independently_complete = provider_floor > 0 and symbol_count >= provider_floor
        expected_symbols = provider_floor
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
        expected_symbols=expected_symbols,
        provider=batch.provider,
        status="complete" if independently_complete else "observed",
        details={
            "provider_expected_symbols": provider_expected,
            "provider_expected_floor": provider_floor,
            "provider_expected_source": batch.metadata.get("expected_symbols_source"),
            "reference_ratio": reference_ratio,
            "reference_tolerance_symbols": reference_tolerance_symbols,
            "full_market_response_complete": response_complete,
            "full_market_response_rows": batch.metadata.get("full_market_response_rows"),
            "full_market_response_limit": response_limit,
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
    on_error=None,
) -> int:
    """Backfill full-market snapshots with bounded, multi-day storage merges."""
    pending = []
    total_rows = 0
    incomplete_market = set(store.incomplete_market_snapshot_dates(AssetType.STOCK, days))
    complete_market = set(days) - incomplete_market

    def report_error(dataset: str, trade_date: date, exc: Exception) -> None:
        if on_error:
            on_error(dataset, trade_date, exc)

    def flush() -> None:
        nonlocal total_rows
        if not pending:
            return
        store.write_daily(pd.concat([batch.data for _, batch, _ in pending]), AssetType.STOCK)
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
                report_error("行情", trade_date, exc)
            total_rows += len(batch.data)
        pending.clear()

    try:
        for index, trade_date in enumerate(days, 1):
            if include_basic and (force or not store.daily_basic_complete(trade_date)):
                try:
                    update_daily_basic(
                        router,
                        store,
                        trade_date,
                        reference_ratio=config.providers.market_snapshot_reference_ratio,
                        reference_tolerance_symbols=(
                            config.providers.market_snapshot_reference_tolerance_symbols
                        ),
                    )
                except (ProviderError, DataQualityError) as exc:
                    report_error("daily_basic", trade_date, exc)
            existing_complete = trade_date in complete_market
            if not force and existing_complete:
                if progress:
                    progress(index, len(days), trade_date, 0)
                continue
            try:
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
            except (ProviderError, DataQualityError) as exc:
                report_error("行情", trade_date, exc)
                if progress:
                    progress(index, len(days), trade_date, 0)
                continue
            pending.append((trade_date, batch, existing_complete))
            if progress:
                progress(index, len(days), trade_date, len(batch.data))
            if len(pending) >= config.providers.market_history_batch_days:
                flush()
    finally:
        # A later provider failure must not discard already downloaded days
        # that have not yet reached the configured batch size.
        flush()

    incomplete_market = store.incomplete_market_snapshot_dates(AssetType.STOCK, days)
    incomplete_basic = store.incomplete_daily_basic_dates(days) if include_basic else []
    if incomplete_market or incomplete_basic:
        parts = []
        if incomplete_market:
            parts.append(
                f"行情缺失或不完整 {len(incomplete_market)} 天，例如 {incomplete_market[:5]}"
            )
        if incomplete_basic:
            parts.append(
                f"daily_basic 缺失或不完整 {len(incomplete_basic)} 天，例如 {incomplete_basic[:5]}"
            )
        raise DataQualityError("历史回填未全部完成；" + "；".join(parts) + "。可直接重跑以续传")
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


def _microcap_signal_bars(
    store: DataStore,
    cfg: dict,
    as_of: str | None,
) -> pd.DataFrame:
    target = (
        pd.Timestamp(as_of).date()
        if as_of is not None
        else store.latest_complete_market_snapshot_date(AssetType.STOCK)
    )
    if target is None:
        raise DataQualityError("没有完整的股票快照，无法生成微盘股信号")
    dates = store.completed_market_snapshot_dates(AssetType.STOCK, end=target)
    if not dates or dates[-1] != target:
        raise DataQualityError(f"微盘股策略所需全市场快照缺失或不完整: {target}")
    selection = str(cfg.get("selection", "pool"))
    rebalance = str(cfg.get("rebalance", "weekly"))
    required = {"daily": 2, "weekly": 10, "monthly": 35}[rebalance]
    if selection == "rps":
        required = max(required, int(cfg.get("rps_lookback_days", 120)) + 2)
    selected_dates = dates[-required:]
    bars = store.read_daily_dates(
        [], selected_dates, asset_type=AssetType.STOCK, adjustment=Adjustment.NONE
    )
    if bars.empty:
        raise DataQualityError("完整快照没有可用股票行情")
    bars = _attach_microcap_basic(store, bars, str(selected_dates[0]), str(selected_dates[-1]))
    if len(selected_dates) < required:
        raise DataQualityError(
            f"微盘股信号历史不足：需要至少 {required} 个完整交易日，"
            f"当前只有 {len(selected_dates)} 个"
        )
    return bars


def _strategy_warmup_bars(name: str, cfg: dict) -> int:
    if name == "etf_rotation":
        return max(
            [int(value) + 1 for value in cfg.get("momentum_windows", [20, 60])]
            + [int(cfg.get("ma_window", 28))]
        )
    if name == "logbias":
        return int(cfg.get("ema_window", 20)) * 5
    if name == "microcap" and cfg.get("selection", "pool") == "rps":
        return int(cfg.get("rps_lookback_days", 120)) + 2
    return 35 if name == "microcap" and cfg.get("rebalance", "weekly") == "monthly" else 10


def strategy_warmup_calendar_days(name: str, cfg: dict) -> int:
    bars = _strategy_warmup_bars(name, cfg)
    return max(30, bars * 2)


def _validate_backtest_warmup(
    bars: pd.DataFrame,
    *,
    name: str,
    cfg: dict,
    symbols: list[str],
    requested_start: pd.Timestamp,
) -> None:
    required = _strategy_warmup_bars(name, cfg)
    before = bars[pd.to_datetime(bars["trade_date"]) < requested_start]
    if name == "microcap":
        actual = int(pd.to_datetime(before["trade_date"]).nunique())
        if actual < required:
            raise DataQualityError(
                f"策略 {name} 回测预热历史不足：起始日前需要至少 {required} 个完整交易日，"
                f"当前只有 {actual} 个；请先补齐更早历史数据"
            )
        return
    counts = before.groupby("symbol")["trade_date"].nunique()
    missing = {
        symbol: int(counts.get(symbol, 0))
        for symbol in symbols
        if int(counts.get(symbol, 0)) < required
    }
    if missing:
        detail = ", ".join(f"{symbol}={count}" for symbol, count in missing.items())
        raise DataQualityError(
            f"策略 {name} 回测预热历史不足：每个证券起始日前至少需要 {required} 根行情，"
            f"当前 {detail}；请先补齐更早历史数据"
        )


def _validate_signal_history(
    bars: pd.DataFrame,
    *,
    name: str,
    cfg: dict,
    symbols: list[str],
    target: pd.Timestamp,
) -> None:
    required = _strategy_warmup_bars(name, cfg)
    history = bars[pd.to_datetime(bars["trade_date"]).dt.normalize() <= target]
    counts = history.groupby("symbol")["trade_date"].nunique()
    missing = {
        symbol: int(counts.get(symbol, 0))
        for symbol in symbols
        if int(counts.get(symbol, 0)) < required
    }
    if missing:
        detail = ", ".join(f"{symbol}={count}" for symbol, count in missing.items())
        raise DataQualityError(
            f"策略 {name} 信号历史不足：每个证券至少需要 {required} 根行情，当前 {detail}；"
            "不会发布未充分预热的信号"
        )


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


def build_strategy_signal(
    config: AppConfig, store: DataStore, name: str, as_of: str | None = None
) -> SignalResult:
    cfg = config.strategies.get(name, {})
    symbols = list(cfg.get("symbols", []))
    if name != "microcap" and not symbols:
        raise ValueError(f"策略 {name} 未配置 symbols")
    asset_type = AssetType(cfg.get("asset_type", "stock" if name == "microcap" else "etf"))
    if name == "microcap":
        bars = _microcap_signal_bars(store, cfg, as_of)
    else:
        bars = strategy_bars(
            store, symbols, None, as_of, asset_type, str(cfg.get("adjustment", "none"))
        )
        target = (
            pd.Timestamp(as_of).normalize()
            if as_of is not None
            else pd.to_datetime(bars["trade_date"]).max().normalize()
        )
        actual = set(
            bars.loc[pd.to_datetime(bars["trade_date"]).dt.normalize().eq(target), "symbol"].astype(
                str
            )
        )
        stale = sorted(set(symbols) - actual)
        if stale:
            raise DataQualityError(
                f"策略 {name} 在 {target.date()} 缺少当日行情: "
                + ", ".join(stale)
                + "；不会输出过期信号，也不会发布部分证券缺失的信号"
            )
        _validate_signal_history(
            bars,
            name=name,
            cfg=cfg,
            symbols=symbols,
            target=target,
        )
    strategy = get_strategy(name, cfg)
    result = strategy.latest_signal(bars)
    if as_of is not None and result.as_of.normalize() != pd.Timestamp(as_of).normalize():
        raise DataQualityError(
            f"策略 {name} 的最新信号日期为 {result.as_of.date()}，请求日期为 "
            f"{pd.Timestamp(as_of).date()}；不会输出过期信号"
        )
    return result


def save_strategy_signal(result: SignalResult, out_dir: Path) -> dict[str, Path]:
    """Publish one signal generation only after both files are complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.as_of.strftime("%Y%m%d")
    staging = out_dir / ".staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    names = {
        "csv": f"signal_{stamp}.csv",
        "json": f"signal_{stamp}.json",
    }
    try:
        result.diagnostics.to_csv(staging / names["csv"], encoding="utf-8-sig")
        pd.Series({"as_of": str(result.as_of), "summary": result.summary}).to_json(
            staging / names["json"], force_ascii=False, indent=2
        )
        for filename in names.values():
            (staging / filename).replace(out_dir / filename)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {key: out_dir / filename for key, filename in names.items()}


def run_strategy_signal(config: AppConfig, store: DataStore, name: str, as_of: str | None = None):
    result = build_strategy_signal(config, store, name, as_of)
    out_dir = config.paths.artifacts_dir / "signals" / name
    save_strategy_signal(result, out_dir)
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
    requested_start = pd.Timestamp(start).normalize()
    warmup_start = requested_start - pd.Timedelta(days=strategy_warmup_calendar_days(name, cfg))
    bars = strategy_bars(store, symbols, str(warmup_start.date()), end, asset_type, adjustment)
    if name == "microcap":
        bars = _attach_microcap_basic(store, bars, str(warmup_start.date()), end)
    _validate_backtest_warmup(
        bars,
        name=name,
        cfg=cfg,
        symbols=symbols,
        requested_start=requested_start,
    )
    strategy = get_strategy(name, cfg)
    targets = strategy.generate_targets(bars)
    evaluation_bars = bars[pd.to_datetime(bars["trade_date"]) >= requested_start].copy()
    prior_targets = targets.index[targets.index < requested_start]
    target_start = prior_targets[-1] if len(prior_targets) else requested_start
    evaluation_targets = targets[targets.index >= target_start].copy()
    if evaluation_bars.empty or evaluation_targets.empty:
        raise ValueError(f"回测区间 {start} 以后没有可用行情或目标权重")
    bc = config.backtest
    execution = ExecutionConfig(
        initial_cash=bc.initial_cash,
        commission_rate=bc.commission_rate,
        stamp_duty_rate=(bc.stamp_duty_rate if asset_type == AssetType.STOCK else 0.0),
        slippage_rate=bc.slippage_rate,
        lot_size=bc.lot_size,
        risk_free_annual=bc.risk_free_annual,
    )
    result = run_weight_backtest(evaluation_bars, evaluation_targets, execution)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    out_dir = config.paths.artifacts_dir / "backtests" / name / run_id
    benchmark_name = cfg.get("benchmark")
    benchmark_equity = None
    benchmark_status = "未配置"
    if benchmark_name:
        benchmark_status = "本地无数据"
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
            closes.index = pd.to_datetime(closes.index).normalize()
            closes = closes[~closes.index.duplicated(keep="last")]
            required_dates = pd.DatetimeIndex(result.equity.index).normalize()
            missing_dates = required_dates.difference(closes.index)
            if len(missing_dates):
                examples = ", ".join(str(day.date()) for day in missing_dates[:5])
                benchmark_status = (
                    f"区间不完整：缺少 {len(missing_dates)} 个策略交易日，例如 {examples}；"
                    "未计算基准指标"
                )
            else:
                closes = closes.reindex(required_dates)
                benchmark_equity = closes / closes.iloc[0] * execution.initial_cash
                benchmark_status = (
                    f"完整覆盖 {required_dates.min().date()} 至 {required_dates.max().date()}"
                )
    report_paths = save_backtest_report(
        name=name,
        result=result,
        out_dir=out_dir,
        execution=execution,
        strategy_config=cfg,
        benchmark_equity=benchmark_equity,
        benchmark_name=benchmark_name,
        benchmark_status=benchmark_status,
    )
    result.artifacts = report_paths.as_dict()
    return result
