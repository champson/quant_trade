from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    runs_dir: Path = Path("runs")
    database: Path = Path("data/quant_trade.duckdb")


class RetryConfig(BaseModel):
    attempts: int = 6
    delays: list[float] = Field(default_factory=lambda: [2, 5, 10, 20, 40, 60])
    circuit_failures: int = 5
    circuit_cooldown_seconds: int = 300


class ProvidersConfig(BaseModel):
    priority: list[str] = Field(default_factory=lambda: ["tushare", "baostock", "akshare"])
    allow_fallback: bool = True
    retry: RetryConfig = Field(default_factory=RetryConfig)
    tushare: dict[str, Any] = Field(default_factory=dict)
    baostock: dict[str, Any] = Field(default_factory=dict)
    akshare: dict[str, Any] = Field(default_factory=dict)


class MinuteConfig(BaseModel):
    inbox: Path = Path("data/inbox/minute")
    archive: Path = Path("data/archive/minute")
    quarantine: Path = Path("data/quarantine/minute")
    encoding_candidates: list[str] = Field(
        default_factory=lambda: ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    )
    filename_symbol_regex: str = r"(?P<symbol>\d{6})(?:[._-]?(?P<exchange>SH|SZ|BJ))?"
    timestamp_convention: str = "source"
    compression: str = "zstd"
    compression_level: int = 6
    chunk_rows: int = 250_000
    row_group_rows: int = 250_000


class BacktestConfig(BaseModel):
    initial_cash: float = 1_000_000
    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.0002
    risk_free_annual: float = 0.015


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    minute: MinuteConfig = Field(default_factory=MinuteConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)

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
