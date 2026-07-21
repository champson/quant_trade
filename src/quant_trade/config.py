from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    runs_dir: Path = Path("runs")
    database: Path = Path("data/quant_trade.duckdb")


class RetryConfig(BaseModel):
    attempts: int = Field(default=6, ge=1)
    delays: list[Annotated[float, Field(ge=0)]] = Field(
        default_factory=lambda: [2, 5, 10, 20, 40, 60], min_length=1
    )
    circuit_failures: int = Field(default=5, ge=1)
    circuit_cooldown_seconds: int = Field(default=300, ge=0)


class ProvidersConfig(BaseModel):
    priority: list[str] = Field(default_factory=lambda: ["tushare", "baostock", "akshare"])
    allow_fallback: bool = True
    calendar_mutable_ttl_hours: float = Field(default=24.0, ge=0)
    market_snapshot_min_symbols: dict[str, int] = Field(
        default_factory=lambda: {"stock": 4500, "convertible_bond": 250}
    )
    market_snapshot_reference_ratio: float = Field(default=0.9, gt=0, le=1)
    market_snapshot_validation_sample_size: int = Field(default=32, ge=0)
    market_snapshot_cache_ttl_seconds: float = Field(default=60.0, ge=0)
    market_history_batch_days: int = Field(default=20, ge=1)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    tushare: dict[str, Any] = Field(default_factory=dict)
    baostock: dict[str, Any] = Field(default_factory=dict)
    akshare: dict[str, Any] = Field(default_factory=dict)

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("providers.priority 不能为空")
        if len(value) != len(set(value)):
            raise ValueError("providers.priority 不能包含重复数据源")
        return value


class MinuteConfig(BaseModel):
    inbox: Path = Path("data/inbox/minute")
    archive: Path = Path("data/archive/minute")
    quarantine: Path = Path("data/quarantine/minute")
    encoding_candidates: list[str] = Field(
        default_factory=lambda: ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    )
    filename_symbol_regex: str = r"(?P<symbol>\d{6})(?:[._-]?(?P<exchange>SH|SZ|BJ))?"
    timestamp_convention: Literal["source", "bar_start", "bar_end"] = "source"
    inbox_frequency: Literal["1min", "5min", "15min", "30min", "60min"] = "1min"
    inbox_asset_type: Literal["auto", "stock", "etf", "index"] = "auto"
    compression: str = "zstd"
    compression_level: int = Field(default=6, ge=1, le=22)
    chunk_rows: int = Field(default=250_000, ge=1)
    row_group_rows: int = Field(default=250_000, ge=1)
    fail_daily_on_import_error: bool = True

    @field_validator("filename_symbol_regex")
    @classmethod
    def validate_symbol_regex(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"filename_symbol_regex 不是有效正则表达式: {exc}") from exc
        return value


class BacktestConfig(BaseModel):
    initial_cash: float = Field(default=1_000_000, gt=0)
    commission_rate: float = Field(default=0.00025, ge=0, lt=1)
    stamp_duty_rate: float = Field(default=0.0005, ge=0, lt=1)
    slippage_rate: float = Field(default=0.0002, ge=0, lt=1)
    risk_free_annual: float = Field(default=0.015, gt=-1)
    lot_size: int = Field(default=100, ge=1)


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    minute: MinuteConfig = Field(default_factory=MinuteConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_strategy_storage_contracts(self) -> "AppConfig":
        for name, strategy in self.strategies.items():
            symbols = strategy.get("symbols")
            if symbols is not None and (
                not isinstance(symbols, list)
                or any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols)
            ):
                raise ValueError(f"策略 {name}.symbols 必须是非空代码字符串列表")
            if name != "microcap" and strategy.get("enabled") and not symbols:
                raise ValueError(f"已启用策略 {name}.symbols 不能为空")
            benchmark = strategy.get("benchmark")
            if benchmark is not None and (not isinstance(benchmark, str) or not benchmark.strip()):
                raise ValueError(f"策略 {name}.benchmark 必须是非空代码字符串")
            try:
                if "asset_type" in strategy:
                    from quant_trade.models import AssetType

                    AssetType(strategy["asset_type"])
                if "adjustment" in strategy:
                    from quant_trade.models import Adjustment

                    Adjustment(strategy["adjustment"])
                if "benchmark_asset_type" in strategy:
                    from quant_trade.models import AssetType

                    AssetType(strategy["benchmark_asset_type"])
                if "benchmark_adjustment" in strategy:
                    from quant_trade.models import Adjustment

                    Adjustment(strategy["benchmark_adjustment"])
            except ValueError as exc:
                raise ValueError(f"策略 {name} 的资产/复权配置无效: {exc}") from exc
            if name == "etf_rotation":
                rebalance_days = int(strategy.get("rebalance_days", 5))
                if rebalance_days not in {1, 5, 20, 21, 22, 23}:
                    raise ValueError("策略 etf_rotation.rebalance_days 只支持 1、5 或 20-23")
                windows = strategy.get("momentum_windows", [20, 60, 120])
                if (
                    not isinstance(windows, list)
                    or not windows
                    or any(not isinstance(value, int) or value <= 0 for value in windows)
                ):
                    raise ValueError("策略 etf_rotation.momentum_windows 必须是正整数列表")
                for key, default in (("ma_window", 20), ("hold_num", 1)):
                    if int(strategy.get(key, default)) <= 0:
                        raise ValueError(f"策略 etf_rotation.{key} 必须大于 0")
            if name == "logbias":
                if int(strategy.get("ema_window", 20)) <= 0:
                    raise ValueError("策略 logbias.ema_window 必须大于 0")
                stop = float(strategy.get("stop", -5.0))
                entry = float(strategy.get("entry", 5.0))
                overheat = float(strategy.get("overheat", 15.0))
                if not stop < entry < overheat:
                    raise ValueError("策略 logbias 必须满足 stop < entry < overheat")
            if name == "microcap":
                if strategy.get("adjustment", "none") != "none":
                    raise ValueError("微盘股策略只支持 adjustment=none 的完整全市场快照")
                if strategy.get("selection", "pool") not in {
                    "pool",
                    "rank",
                    "rps",
                    "smallest",
                }:
                    raise ValueError("策略 microcap.selection 配置无效")
                if strategy.get("rebalance", "weekly") not in {
                    "daily",
                    "weekly",
                    "monthly",
                }:
                    raise ValueError("策略 microcap.rebalance 配置无效")
                for key, default in (
                    ("pool_size", 400),
                    ("hold_count", 20),
                    ("rank_start", 1),
                    ("rps_lookback_days", 120),
                ):
                    if int(strategy.get(key, default)) <= 0:
                        raise ValueError(f"策略 microcap.{key} 必须大于 0")
                target = float(strategy.get("rps_target", 70))
                if not 0 <= target <= 100:
                    raise ValueError("策略 microcap.rps_target 必须在 0 到 100 之间")
        return self

    def ensure_directories(self) -> None:
        for path in (
            self.paths.data_dir,
            self.paths.artifacts_dir,
            self.paths.runs_dir,
            self.minute.inbox,
            self.minute.archive,
            self.minute.quarantine,
            self.paths.database.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    tushare_token: str = ""
    tushare_http_url: str = ""


def load_config(path: str | Path | None = None) -> AppConfig:
    target = Path(path or os.getenv("QT_CONFIG", "configs/default.yaml"))
    payload: dict[str, Any] = {}
    if target.exists():
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    cfg = AppConfig.model_validate(payload)
    cfg.ensure_directories()
    return cfg
