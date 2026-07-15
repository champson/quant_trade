from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quant_trade.config import AppConfig


class DataStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_directories()
        self.root = config.paths.data_dir / "processed"
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = config.paths.database
        self._init_db()

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database))

    def _init_db(self) -> None:
        with self.connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS data_fetches (
                    fetched_at TIMESTAMP, dataset VARCHAR, provider VARCHAR,
                    symbols VARCHAR, start_at VARCHAR, end_at VARCHAR,
                    rows BIGINT, status VARCHAR, warnings VARCHAR
                );
                CREATE TABLE IF NOT EXISTS minute_imports (
                    file_hash VARCHAR PRIMARY KEY, file_name VARCHAR,
                    imported_at TIMESTAMP, rows BIGINT, min_time TIMESTAMP,
                    max_time TIMESTAMP, status VARCHAR, details VARCHAR
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY, task VARCHAR, started_at TIMESTAMP,
                    finished_at TIMESTAMP, status VARCHAR, as_of VARCHAR,
                    config_json VARCHAR, details VARCHAR
                );
                """
            )

    @staticmethod
    def safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_").replace(".", "_")

    def daily_path(self, asset_type: str, symbol: str) -> Path:
        return self.root / "daily" / asset_type / f"{self.safe_symbol(symbol)}.parquet"

    def market_snapshot_complete(self, asset_type: str, trade_date: date | str) -> bool:
        stamp = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        return (self.root / "snapshots" / asset_type / f"{stamp}.complete").exists()

    def mark_market_snapshot(self, asset_type: str, trade_date: date | str) -> None:
        stamp = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
        path = self.root / "snapshots" / asset_type / f"{stamp}.complete"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def write_daily(self, df: pd.DataFrame, asset_type: str) -> int:
        count = 0
        for symbol, part in df.groupby("symbol", sort=False):
            path = self.daily_path(asset_type, str(symbol))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = pd.read_parquet(path)
                part = pd.concat([old, part], ignore_index=True)
            part = part.sort_values("trade_date").drop_duplicates(
                ["symbol", "trade_date"], keep="last"
            )
            part.to_parquet(path, index=False)
            count += len(part)
        return count

    def read_daily(
        self, symbols: list[str], start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        daily_root = self.root / "daily"
        for symbol in symbols:
            matches = list(daily_root.glob(f"*/{self.safe_symbol(symbol)}.parquet"))
            for path in matches:
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        if start:
            out = out[out["trade_date"] >= pd.Timestamp(start)]
        if end:
            out = out[out["trade_date"] <= pd.Timestamp(end)]
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def daily_basic_path(self, trade_date: str) -> Path:
        return self.root / "daily_basic" / f"trade_date={trade_date}" / "data.parquet"

    def write_daily_basic(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        work = df.rename(columns={"ts_code": "symbol"}).copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"].astype(str))
        if "total_mv" not in work:
            raise ValueError("daily_basic 缺少 total_mv")
        for trade_date, part in work.groupby(work["trade_date"].dt.strftime("%Y-%m-%d")):
            path = self.daily_basic_path(trade_date)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
            part.drop_duplicates(["symbol", "trade_date"], keep="last").to_parquet(path, index=False)
        return len(work)

    def read_daily_basic(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        paths = list((self.root / "daily_basic").glob("trade_date=*/data.parquet"))
        if not paths:
            return pd.DataFrame()
        out = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        if start:
            out = out[out["trade_date"] >= pd.Timestamp(start)]
        if end:
            out = out[out["trade_date"] <= pd.Timestamp(end)]
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def minute_partition(self, trade_date: str) -> Path:
        return self.root / "minute" / "1min" / f"trade_date={trade_date}" / "bars.parquet"

    def write_minute(self, df: pd.DataFrame) -> int:
        written = 0
        for trade_date, part in df.groupby(df["trade_date"].astype(str)):
            path = self.minute_partition(str(trade_date))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
            part = part.sort_values(["symbol", "bar_time"]).drop_duplicates(
                ["symbol", "bar_time"], keep="last"
            )
            part.to_parquet(path, index=False)
            written += len(part)
        return written

    def minute_imported(self, file_hash: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT status FROM minute_imports WHERE file_hash = ?", [file_hash]
            ).fetchone()
        return bool(row and row[0] == "success")

    def record_minute_import(self, values: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM minute_imports WHERE file_hash = ?", [values["file_hash"]])
            con.execute(
                "INSERT INTO minute_imports VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    values["file_hash"], values["file_name"], datetime.now(),
                    values.get("rows", 0), values.get("min_time"), values.get("max_time"),
                    values["status"], json.dumps(values.get("details", {}), ensure_ascii=False),
                ],
            )

    def record_fetch(self, **values: Any) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO data_fetches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [datetime.now(), values.get("dataset"), values.get("provider"),
                 values.get("symbols"), values.get("start"), values.get("end"),
                 values.get("rows", 0), values.get("status"),
                 json.dumps(values.get("warnings", []), ensure_ascii=False)],
            )
