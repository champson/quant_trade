from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from quant_trade.config import load_config
from quant_trade.data.minute_archive import MinuteArchiveImporter
from quant_trade.data.minute_directory import MinuteDirectoryImporter
from quant_trade.data.router import build_router
from quant_trade.data.storage import DataStore
from quant_trade.models import Adjustment, AssetType
from quant_trade.data.calendar import trading_days
from quant_trade.pipelines.daily import run_daily
from quant_trade.reports.market_review import build_market_review
from quant_trade.reports.render import save_market_review
from quant_trade.services import (
    run_strategy_backtest,
    run_strategy_signal,
    update_bars,
    update_daily_basic,
)
from quant_trade.strategies import strategy_names


app = typer.Typer(help="A股复盘、数据管理和策略研究平台", no_args_is_help=True)
data_app = typer.Typer(help="数据下载和导入", no_args_is_help=True)
minute_app = typer.Typer(help="分钟ZIP与目录数据", no_args_is_help=True)
strategy_app = typer.Typer(help="策略信号和回测", no_args_is_help=True)
review_app = typer.Typer(help="市场复盘", no_args_is_help=True)
daily_app = typer.Typer(help="每日流水线", no_args_is_help=True)
research_app = typer.Typer(help="研究任务", no_args_is_help=True)
app.add_typer(data_app, name="data")
data_app.add_typer(minute_app, name="minute")
app.add_typer(strategy_app, name="strategy")
app.add_typer(review_app, name="review")
app.add_typer(daily_app, name="daily")
app.add_typer(research_app, name="research")


def _runtime(config_path: str | None):
    cfg = load_config(config_path)
    store = DataStore(cfg)
    return cfg, store, build_router(cfg, store)


def _date(value: str | None, default: date) -> date:
    return pd.Timestamp(value).date() if value else default


def _reject_future(value: date, param_hint: str) -> None:
    if value > date.today():
        raise typer.BadParameter("不支持请求未来行情日期", param_hint=param_hint)


@data_app.command("update")
def data_update(
    symbols: Annotated[str, typer.Option(help="逗号分隔代码；留空表示 Tushare 当日全市场")] = "",
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    asset_type: Annotated[AssetType, typer.Option()] = AssetType.STOCK,
    provider: Annotated[str, typer.Option()] = "auto",
    adjustment: Annotated[Adjustment, typer.Option()] = Adjustment.NONE,
    force: Annotated[bool, typer.Option("--force", help="忽略完整缓存并重新下载")] = False,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    end_date = _date(end, date.today())
    _reject_future(end_date, "--end")
    start_date = _date(start, end_date if not symbols else end_date - timedelta(days=420))
    codes = [x.strip() for x in symbols.split(",") if x.strip()]
    if not codes and adjustment != Adjustment.NONE:
        raise typer.BadParameter("全市场快照只支持 adjustment=none", param_hint="--adjustment")
    if not codes and asset_type not in {
        AssetType.STOCK,
        AssetType.CONVERTIBLE_BOND,
    }:
        raise typer.BadParameter(
            "全市场快照只支持 stock 或 convertible_bond", param_hint="--asset-type"
        )
    cfg, store, router = _runtime(config)
    try:
        frame = update_bars(
            cfg,
            router,
            store,
            codes,
            start_date,
            end_date,
            asset_type,
            provider,
            adjustment,
            resume=not force,
        )
        source = frame["source"].iloc[0] if not frame.empty else "本地缓存"
        typer.echo(f"完成：新增/更新 {len(frame):,} 行，来源 {source}")
    finally:
        router.close()


@data_app.command("market-history")
def market_history(
    start: Annotated[str, typer.Option()],
    end: Annotated[str | None, typer.Option()] = None,
    include_basic: Annotated[bool, typer.Option("--include-basic/--no-basic")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """按交易日下载全市场行情，为市场宽度和微盘股研究准备数据。"""
    start_date, end_date = pd.Timestamp(start).date(), _date(end, date.today())
    _reject_future(end_date, "--end")
    cfg, store, router = _runtime(config)
    try:
        days = trading_days(router, start_date, end_date, store)
        for index, trade_date in enumerate(days, 1):
            frame = update_bars(
                cfg,
                router,
                store,
                [],
                trade_date,
                trade_date,
                AssetType.STOCK,
                resume=not force,
            )
            if include_basic:
                basic_path = store.daily_basic_path(str(trade_date))
                if force or not basic_path.exists():
                    update_daily_basic(router, store, trade_date)
            typer.echo(f"[{index}/{len(days)}] {trade_date} {len(frame):,} 行")
    finally:
        router.close()


@minute_app.command("inspect")
def minute_inspect(
    path: Path, config: Annotated[str | None, typer.Option("--config")] = None
) -> None:
    cfg = load_config(config)
    profile = MinuteArchiveImporter(cfg, DataStore(cfg)).inspect(path)
    typer.echo(json.dumps(asdict(profile), ensure_ascii=False, indent=2))


@minute_app.command("import")
def minute_import(
    path: Path,
    frequency: Annotated[str | None, typer.Option()] = None,
    asset_type: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    result = MinuteArchiveImporter(cfg, DataStore(cfg)).import_archive(
        path,
        frequency=frequency or cfg.minute.inbox_frequency,
        asset_type=asset_type or cfg.minute.inbox_asset_type,
    )
    typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))


@minute_app.command("import-inbox")
def minute_import_inbox(
    frequency: Annotated[str | None, typer.Option()] = None,
    asset_type: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    results = MinuteArchiveImporter(cfg, DataStore(cfg)).import_inbox(
        frequency=frequency or cfg.minute.inbox_frequency,
        asset_type=asset_type or cfg.minute.inbox_asset_type,
    )
    typer.echo(json.dumps([asdict(x) for x in results], ensure_ascii=False, indent=2))


@minute_app.command("inspect-directory")
def minute_inspect_directory(
    path: Path,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    profile = MinuteDirectoryImporter(cfg, DataStore(cfg)).inspect_directory(path)
    typer.echo(json.dumps(asdict(profile), ensure_ascii=False, indent=2))
    if not profile.valid:
        raise typer.Exit(2)


@minute_app.command("import-directory")
def minute_import_directory(
    path: Path,
    frequency: Annotated[str, typer.Option()] = "5min",
    force: Annotated[bool, typer.Option("--force", help="忽略文件哈希并重新导入")] = False,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    importer = MinuteDirectoryImporter(cfg, DataStore(cfg))

    def progress(index: int, total: int, item) -> None:
        if index == 1 or index == total or index % 50 == 0 or item.status == "failed":
            typer.echo(
                f"[{index}/{total}] {item.symbol} {item.status} "
                f"写入={item.rows_written:,} 过滤={item.rows_filtered:,}"
            )

    result = importer.import_directory(
        path, frequency=frequency, resume=not force, progress=progress
    )
    typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    if result.files_failed:
        raise typer.Exit(2)


@minute_app.command("verify")
def minute_verify(
    frequency: Annotated[str, typer.Option()] = "5min",
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    store = DataStore(cfg)
    with store.connect() as con:
        summary = con.execute(
            """
            SELECT asset_type, COUNT(DISTINCT symbol) AS symbols,
                   COUNT(*) AS partitions, SUM(rows) AS rows,
                   MIN(min_time) AS min_time, MAX(max_time) AS max_time
            FROM minute_partitions WHERE frequency = ?
            GROUP BY asset_type ORDER BY asset_type
            """,
            [frequency],
        ).df()
        paths = con.execute(
            "SELECT path FROM minute_partitions WHERE frequency = ?", [frequency]
        ).fetchall()
    missing = [path for (path,) in paths if not Path(path).exists()]
    typer.echo(summary.to_string(index=False) if not summary.empty else "没有已导入分区")
    typer.echo(f"\n缺失Parquet文件：{len(missing)}")
    if missing:
        typer.echo("\n".join(missing[:20]))
        raise typer.Exit(2)


@review_app.command("close")
def review_close(
    as_of: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    store = DataStore(cfg)
    bars = store.read_daily([], None, as_of, asset_type=AssetType.STOCK, adjustment="none")
    if bars.empty:
        raise typer.BadParameter("没有股票行情，请先执行 qt data update")
    report = build_market_review(bars, as_of)
    outputs = save_market_review(report, cfg.paths.artifacts_dir / "reviews")
    typer.echo(json.dumps({k: str(v) for k, v in outputs.items()}, ensure_ascii=False, indent=2))


@strategy_app.command("list")
def strategy_list() -> None:
    typer.echo("\n".join(strategy_names()))


@strategy_app.command("signal")
def strategy_signal(
    name: str,
    as_of: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    result = run_strategy_signal(cfg, DataStore(cfg), name, as_of)
    typer.echo(f"数据日期：{result.as_of.date()}\n目标：{result.summary}")
    typer.echo(result.diagnostics.to_string())


@strategy_app.command("backtest")
def strategy_backtest(
    name: str,
    start: Annotated[str, typer.Option()] = "2020-01-01",
    end: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    result = run_strategy_backtest(cfg, DataStore(cfg), name, start, end)
    typer.echo(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    typer.echo("\n报告产物：")
    typer.echo(
        json.dumps(
            {key: str(path) for key, path in result.artifacts.items()}, ensure_ascii=False, indent=2
        )
    )


@research_app.command("correlation")
def research_correlation(
    symbols: Annotated[str, typer.Option(help="逗号分隔代码")],
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    asset_type: Annotated[AssetType, typer.Option()] = AssetType.STOCK,
    adjustment: Annotated[Adjustment, typer.Option()] = Adjustment.NONE,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    cfg = load_config(config)
    store = DataStore(cfg)
    codes = [x.strip() for x in symbols.split(",") if x.strip()]
    bars = store.read_daily(codes, start, end, asset_type=asset_type, adjustment=adjustment)
    prices = bars.pivot(index="trade_date", columns="symbol", values="close")
    corr = prices.pct_change(fill_method=None).corr()
    out = cfg.paths.artifacts_dir / "research" / "correlation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out, encoding="utf-8-sig")
    typer.echo(corr.to_string())
    typer.echo(f"\n已保存 {out}")


@daily_app.command("run")
def daily_run(
    as_of: Annotated[str | None, typer.Option()] = None,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    as_of_date = _date(as_of, date.today())
    _reject_future(as_of_date, "--as-of")
    cfg, store, router = _runtime(config)
    try:
        result = run_daily(cfg, router, store, as_of_date)
        typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))
    finally:
        router.close()


@app.command("dashboard")
def dashboard(
    port: Annotated[int, typer.Option()] = 8501,
    config: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    if config:
        import os

        os.environ["QT_CONFIG"] = config
    path = Path(__file__).parent / "dashboard" / "app.py"
    raise typer.Exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(path), "--server.port", str(port)]
        ).returncode
    )


if __name__ == "__main__":
    app()
