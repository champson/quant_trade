from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_trade.config import AppConfig
from quant_trade.data.base import EmptyDataError, ProviderError
from quant_trade.data.calendar import trading_days
from quant_trade.data.minute_archive import MinuteArchiveImporter
from quant_trade.data.quality import DataQualityError
from quant_trade.data.router import DataRouter
from quant_trade.data.storage import DataStore
from quant_trade.models import Adjustment, AssetType
from quant_trade.notifications import notify
from quant_trade.reports.market_review import (
    MarketReview,
    asset_return_summary,
    build_market_review,
    logbias_table,
    nav_period_returns,
    normalize_stock_symbol,
    period_returns,
    portfolio_returns,
    price_bias,
)
from quant_trade.reports.render import save_market_review
from quant_trade.runs import RunTracker
from quant_trade.services import (
    build_strategy_signal,
    reference_symbol_floor,
    save_strategy_signal,
    strategy_warmup_calendar_days,
    update_bars,
    update_daily_basic,
)


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


def _strategy_history_calendar_days(strategies: dict) -> int:
    enabled = [
        strategy_warmup_calendar_days(name, strategy)
        for name, strategy in strategies.items()
        if strategy.get("enabled")
    ]
    return max([420, *enabled])


def _anchors(days: list[date], as_of: date) -> list[date]:
    open_days = [d for d in days if d <= as_of]
    if not open_days or open_days[-1] != as_of:
        raise ValueError(f"{as_of} 不是交易日或交易日历尚未更新")
    if len(open_days) < 2:
        raise ValueError(f"{as_of} 之前至少需要一个完整交易日作为日收益锚点")
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


def _read_review_nav(path: Path, review: MarketReview, name: str) -> pd.Series:
    frame = pd.read_csv(path)
    date_column = next(
        (column for column in ("trade_date", "date", "日期") if column in frame),
        None,
    )
    value_column = next(
        (column for column in ("nav", "equity", "净值", "总资产") if column in frame),
        None,
    )
    if date_column is None or value_column is None:
        raise ValueError(f"{path} 必须包含 trade_date/date/日期 与 nav/equity/净值/总资产列")
    values = pd.Series(
        frame[value_column].to_numpy(),
        index=pd.to_datetime(frame[date_column]),
        name=name,
    )
    return nav_period_returns(values, review.anchor_dates, review.as_of)


def _read_review_symbols(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if frame.empty:
        return []
    column = next(
        (name for name in ("正股代码", "代码", "symbol") if name in frame),
        frame.columns[-1],
    )
    return sorted(
        {normalize_stock_symbol(value) for value in frame[column].astype(str).str.strip() if value}
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_daily_generation(
    config: AppConfig,
    stage: Path,
    as_of: date,
    staged_reviews: dict[str, Path],
    staged_signals: dict[str, dict[str, Path]],
) -> tuple[dict[str, str], Path]:
    """Publish all review and signal artifacts, then expose one manifest last."""
    artifact_root = config.paths.artifacts_dir
    destinations: dict[Path, Path] = {}
    review_destinations: dict[str, Path] = {}
    for name, source in staged_reviews.items():
        destination = artifact_root / "reviews" / source.name
        destinations[source] = destination
        review_destinations[name] = destination
    review_manifest = stage / "reviews" / f"review_manifest_{as_of}.json"
    if review_manifest.exists():
        destinations[review_manifest] = artifact_root / "reviews" / review_manifest.name

    signal_destinations: dict[str, dict[str, Path]] = {}
    for strategy, files in staged_signals.items():
        signal_destinations[strategy] = {}
        for kind, source in files.items():
            destination = artifact_root / "signals" / strategy / source.name
            destinations[source] = destination
            signal_destinations[strategy][kind] = destination

    checksums = {
        str(destination.relative_to(artifact_root)): _sha256(source)
        for source, destination in destinations.items()
    }
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in destinations.items():
        source.replace(destination)

    stamp = as_of.isoformat()
    current_review_names = {path.name for path in review_destinations.values()}
    for optional in ("indices", "portfolio", "convertible_bonds", "logbias"):
        candidate = artifact_root / "reviews" / f"{optional}_{stamp}.csv"
        if candidate.name not in current_review_names:
            candidate.unlink(missing_ok=True)

    manifest_payload = {
        "as_of": stamp,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "review_files": {
            name: str(path.relative_to(artifact_root)) for name, path in review_destinations.items()
        },
        "signal_files": {
            strategy: {kind: str(path.relative_to(artifact_root)) for kind, path in files.items()}
            for strategy, files in signal_destinations.items()
        },
        "sha256": checksums,
    }
    manifest_dir = artifact_root / "daily"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"daily_manifest_{stamp}.json"
    temporary = manifest_dir / f".{manifest.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return {name: str(path) for name, path in review_destinations.items()}, manifest


def run_daily(config: AppConfig, router: DataRouter, store: DataStore, as_of: date) -> DailyResult:
    tracker = RunTracker(config, store, "daily", str(as_of))
    result = DailyResult(as_of)
    try:
        calendar = trading_days(router, date(as_of.year - 1, 12, 1), as_of, store)
        snapshot_dates = _anchors(calendar, as_of)
        for snapshot in snapshot_dates:
            microcap_enabled = bool(config.strategies.get("microcap", {}).get("enabled"))
            if not store.daily_basic_complete(snapshot):
                try:
                    update_daily_basic(
                        router,
                        store,
                        snapshot,
                        reference_ratio=config.providers.market_snapshot_reference_ratio,
                        reference_tolerance_symbols=(
                            config.providers.market_snapshot_reference_tolerance_symbols
                        ),
                    )
                except (EmptyDataError, ProviderError) as exc:
                    if microcap_enabled:
                        raise
                    result.warnings.append(f"daily_basic 获取失败 {snapshot}: {exc}")
            update_bars(config, router, store, [], snapshot, snapshot, AssetType.STOCK)
            if microcap_enabled and not store.daily_basic_complete(snapshot):
                basic_count = store.daily_basic_symbol_count(snapshot)
                market_count = store.market_snapshot_symbol_count(AssetType.STOCK, snapshot)
                expected = reference_symbol_floor(
                    market_count,
                    config.providers.market_snapshot_reference_ratio,
                    config.providers.market_snapshot_reference_tolerance_symbols,
                )
                raise DataQualityError(
                    f"{snapshot} daily_basic 不完整：{basic_count} 个证券，至少需要 {expected}"
                )
            try:
                update_bars(
                    config, router, store, [], snapshot, snapshot, AssetType.CONVERTIBLE_BOND
                )
            except (EmptyDataError, ProviderError, DataQualityError) as exc:
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

        strategy_start = as_of - timedelta(days=_strategy_history_calendar_days(config.strategies))
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

        imported = MinuteArchiveImporter(config, store).import_inbox(
            frequency=config.minute.inbox_frequency,
            asset_type=config.minute.inbox_asset_type,
        )
        result.minute_imports = [r.__dict__ for r in imported]
        failed_minute_imports = [item for item in imported if item.status == "failed"]
        if failed_minute_imports:
            message = "分钟 ZIP 导入失败: " + ", ".join(
                item.file_name for item in failed_minute_imports
            )
            result.warnings.append(message)
            if config.minute.fail_daily_on_import_error:
                raise DataQualityError(message)

        incomplete_snapshot_dates = [
            day
            for day in snapshot_dates
            if not store.market_snapshot_complete(AssetType.STOCK, day)
        ]
        if incomplete_snapshot_dates:
            raise DataQualityError(
                "复盘目标日或收益锚点快照不完整: "
                + ", ".join(str(day) for day in incomplete_snapshot_dates)
            )
        market = store.read_daily_dates(
            [], snapshot_dates, asset_type=AssetType.STOCK, adjustment="none"
        )
        review = build_market_review(market, as_of)
        required_review_dates = {review.as_of.date()} | {
            value.date() for value in review.anchor_dates.values()
        }
        incomplete_review_dates = sorted(
            day
            for day in required_review_dates
            if not store.market_snapshot_complete(AssetType.STOCK, day)
        )
        if incomplete_review_dates:
            raise DataQualityError(
                "复盘目标日或收益锚点快照不完整: "
                + ", ".join(str(day) for day in incomplete_review_dates)
            )
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
        index_bias = price_bias(index_bars, as_of, 25) if not index_bars.empty else None
        if index_ret is not None:
            name_by_code = {v: k for k, v in (config.review.get("indices") or {}).items()}
            index_names = list((config.review.get("indices") or {}).keys())
            index_ret = index_ret.rename(index=name_by_code).reindex(index_names)
            if index_bias is not None:
                index_bias = index_bias.rename(index=name_by_code).reindex(index_names)
        portfolio = None
        portfolio_nav_file = config.review.get("portfolio_nav_file")
        portfolio_file = config.review.get("portfolio_file")
        if portfolio_nav_file and Path(portfolio_nav_file).exists():
            portfolio = _read_review_nav(Path(portfolio_nav_file), review, "我的实盘合计")
        elif portfolio_nav_file:
            result.warnings.append(f"实盘净值文件不存在: {portfolio_nav_file}")
        elif portfolio_file and Path(portfolio_file).exists():
            portfolio = portfolio_returns(market, pd.read_csv(portfolio_file, dtype=str), as_of)
        elif portfolio_file:
            result.warnings.append(f"实盘持仓文件不存在: {portfolio_file}")
        cb_summary = None
        incomplete_cb_dates = store.incomplete_market_snapshot_dates(
            AssetType.CONVERTIBLE_BOND, snapshot_dates
        )
        if incomplete_cb_dates:
            result.warnings.append(
                "可转债报告已跳过，快照不完整: "
                + ", ".join(str(day) for day in incomplete_cb_dates)
            )
        else:
            cb_bars = store.read_daily_dates(
                [],
                snapshot_dates,
                asset_type=AssetType.CONVERTIBLE_BOND,
                adjustment="none",
            )
            if not cb_bars.empty:
                cb_summary = asset_return_summary(cb_bars, as_of)
        underlying_summary = None
        underlying_file = config.review.get("convertible_underlyings_file")
        if underlying_file and Path(underlying_file).exists():
            underlying_symbols = _read_review_symbols(Path(underlying_file))
            underlying_bars = market[market["symbol"].astype(str).isin(underlying_symbols)]
            if not underlying_bars.empty:
                underlying_summary = asset_return_summary(underlying_bars, as_of)
        elif underlying_file:
            result.warnings.append(f"可转债正股映射文件不存在: {underlying_file}")
        microcap_returns = None
        microcap_nav_file = config.review.get("microcap_nav_file")
        if microcap_nav_file and Path(microcap_nav_file).exists():
            microcap_returns = _read_review_nav(
                Path(microcap_nav_file), review, "微盘股（自建等权）"
            )
        elif microcap_nav_file:
            result.warnings.append(f"微盘股净值文件不存在: {microcap_nav_file}")
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
        stage = config.paths.artifacts_dir / ".staging" / "daily" / f"{as_of}-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            staged_reviews = save_market_review(
                review,
                stage / "reviews",
                index_returns=index_ret,
                index_bias=index_bias,
                portfolio=portfolio,
                convertible_summary=cb_summary,
                underlying_summary=underlying_summary,
                microcap_returns=microcap_returns,
                bias=bias,
            )
            staged_signals: dict[str, dict[str, Path]] = {}
            for name, cfg in config.strategies.items():
                if cfg.get("enabled"):
                    signal = build_strategy_signal(config, store, name, str(as_of))
                    staged_signals[name] = save_strategy_signal(signal, stage / "signals" / name)
                    result.signals[name] = signal.summary
            result.report_paths, _ = _publish_daily_generation(
                config,
                stage,
                as_of,
                staged_reviews,
                staged_signals,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        tracker.finish("success", result.__dict__)
        notify("Quant Trade 复盘完成", f"{as_of}：{len(result.signals)} 个策略已更新")
        return result
    except Exception as exc:
        tracker.finish("failed", {"error": str(exc), "partial": result.__dict__})
        notify("Quant Trade 复盘失败", str(exc))
        raise
