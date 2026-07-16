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
                CREATE TABLE IF NOT EXISTS minute_sources (
                    source_path VARCHAR, frequency VARCHAR, symbol VARCHAR,
                    asset_type VARCHAR, file_hash VARCHAR, file_size BIGINT,
                    file_mtime_ns BIGINT, imported_at TIMESTAMP,
                    rows_input BIGINT, rows_written BIGINT, rows_filtered BIGINT,
                    min_time TIMESTAMP, max_time TIMESTAMP,
                    status VARCHAR, error VARCHAR,
                    PRIMARY KEY (source_path, frequency)
                );
                CREATE TABLE IF NOT EXISTS minute_partitions (
                    frequency VARCHAR, symbol VARCHAR, year INTEGER,
                    asset_type VARCHAR, path VARCHAR, rows BIGINT,
                    min_time TIMESTAMP, max_time TIMESTAMP,
                    source_hash VARCHAR, updated_at TIMESTAMP,
                    PRIMARY KEY (frequency, symbol, year)
                );
                CREATE TABLE IF NOT EXISTS minute_import_runs (
                    run_id VARCHAR PRIMARY KEY, source_root VARCHAR,
                    frequency VARCHAR, started_at TIMESTAMP,
                    finished_at TIMESTAMP, status VARCHAR,
                    files_total BIGINT, files_success BIGINT,
                    files_skipped BIGINT, files_empty BIGINT,
                    files_failed BIGINT, rows_written BIGINT,
                    rows_filtered BIGINT, details VARCHAR
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id VARCHAR PRIMARY KEY, task VARCHAR, started_at TIMESTAMP,
                    finished_at TIMESTAMP, status VARCHAR, as_of VARCHAR,
                    config_json VARCHAR, details VARCHAR
                );
                ALTER TABLE data_fetches ADD COLUMN IF NOT EXISTS adjustment VARCHAR;
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

    @staticmethod
    def _with_adjustment(df: pd.DataFrame) -> pd.DataFrame:
        """Rows written before the adjustment column existed count as unadjusted."""
        if "adjustment" not in df:
            df = df.assign(adjustment="none")
        else:
            df = df.copy()
            df["adjustment"] = df["adjustment"].fillna("none")
        return df

    def write_daily(self, df: pd.DataFrame, asset_type: str) -> int:
        count = 0
        for symbol, part in self._with_adjustment(df).groupby("symbol", sort=False):
            path = self.daily_path(asset_type, str(symbol))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                old = pd.read_parquet(path)
                part = self._with_adjustment(pd.concat([old, part], ignore_index=True))
            part = part.sort_values("trade_date").drop_duplicates(
                ["symbol", "trade_date", "adjustment"], keep="last"
            )
            part.to_parquet(path, index=False)
            count += len(part)
        return count

    def read_daily(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        adjustment: str = "none",
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        daily_root = self.root / "daily"
        for symbol in symbols:
            matches = list(daily_root.glob(f"*/{self.safe_symbol(symbol)}.parquet"))
            for path in matches:
                frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame()
        out = self._with_adjustment(pd.concat(frames, ignore_index=True))
        out = out[out["adjustment"] == adjustment]
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
        """Legacy 1-minute layout kept for backward compatibility."""
        return self.root / "minute" / "1min" / f"trade_date={trade_date}" / "bars.parquet"

    def minute_symbol_year_path(self, frequency: str, symbol: str, year: int) -> Path:
        return (
            self.root / "minute" / f"frequency={frequency}"
            / f"symbol={self.safe_symbol(symbol)}" / f"year={year}.parquet"
        )

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

    def commit_minute_symbol(
        self,
        *,
        frequency: str,
        symbol: str,
        asset_type: str,
        staged: dict[int, Path],
        statistics: dict[int, dict[str, Any]],
        source_hash: str,
    ) -> None:
        """Atomically replace each symbol/year file, then update the catalog."""
        target_dir = self.minute_symbol_year_path(frequency, symbol, 2000).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        new_years = set(staged)
        existing = set()
        for path in target_dir.glob("year=*.parquet"):
            try:
                existing.add(int(path.stem.split("=", 1)[1]))
            except (IndexError, ValueError):
                continue
        for year, staged_path in staged.items():
            target = self.minute_symbol_year_path(frequency, symbol, year)
            staged_path.replace(target)
        for stale_year in existing - new_years:
            self.minute_symbol_year_path(frequency, symbol, stale_year).unlink(missing_ok=True)

        with self.connect() as con:
            con.execute(
                "DELETE FROM minute_partitions WHERE frequency = ? AND symbol = ?",
                [frequency, symbol],
            )
            for year in sorted(new_years):
                stat = statistics[year]
                target = self.minute_symbol_year_path(frequency, symbol, year).resolve()
                con.execute(
                    "INSERT INTO minute_partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        frequency, symbol, year, asset_type, str(target), stat["rows"],
                        stat["min_time"], stat["max_time"], source_hash, datetime.now(),
                    ],
                )

    def read_minute(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        frequency: str = "5min",
        asset_types: list[str] | None = None,
    ) -> pd.DataFrame:
        if not symbols:
            raise ValueError("分钟查询必须指定至少一个证券代码")
        start_at, end_at = pd.Timestamp(start), pd.Timestamp(end)
        placeholders = ",".join("?" for _ in symbols)
        query = (
            "SELECT path FROM minute_partitions "
            f"WHERE frequency = ? AND symbol IN ({placeholders}) "
            "AND max_time >= ? AND min_time <= ?"
        )
        params: list[Any] = [frequency, *symbols, start_at, end_at]
        if asset_types:
            query += f" AND asset_type IN ({','.join('?' for _ in asset_types)})"
            params.extend(asset_types)
        with self.connect() as con:
            paths = [row[0] for row in con.execute(query, params).fetchall()]
            if not paths:
                return pd.DataFrame()
            return con.execute(
                """
                SELECT * FROM read_parquet(?)
                WHERE bar_time >= ? AND bar_time <= ?
                ORDER BY symbol, bar_time
                """,
                [paths, start_at, end_at],
            ).df()

    def minute_source_unchanged(
        self, source_path: str, frequency: str, file_hash: str
    ) -> bool:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT status, file_hash FROM minute_sources
                WHERE source_path = ? AND frequency = ?
                """,
                [source_path, frequency],
            ).fetchone()
        return bool(row and row[0] in {"success", "empty"} and row[1] == file_hash)

    def minute_source_stat_unchanged(
        self,
        source_path: str,
        frequency: str,
        file_size: int,
        file_mtime_ns: int,
    ) -> bool:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT status, file_size, file_mtime_ns FROM minute_sources
                WHERE source_path = ? AND frequency = ?
                """,
                [source_path, frequency],
            ).fetchone()
        return bool(
            row and row[0] in {"success", "empty"}
            and row[1] == file_size and row[2] == file_mtime_ns
        )

    def record_minute_source(self, values: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                "DELETE FROM minute_sources WHERE source_path = ? AND frequency = ?",
                [values["source_path"], values["frequency"]],
            )
            con.execute(
                "INSERT INTO minute_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    values["source_path"], values["frequency"], values.get("symbol"),
                    values.get("asset_type"), values.get("file_hash"),
                    values.get("file_size", 0), values.get("file_mtime_ns", 0),
                    datetime.now(), values.get("rows_input", 0),
                    values.get("rows_written", 0), values.get("rows_filtered", 0),
                    values.get("min_time"), values.get("max_time"), values["status"],
                    values.get("error"),
                ],
            )

    def start_minute_import_run(
        self, run_id: str, source_root: str, frequency: str, files_total: int
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO minute_import_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, source_root, frequency, datetime.now(), None, "running",
                 files_total, 0, 0, 0, 0, 0, 0, "{}"],
            )

    def finish_minute_import_run(self, run_id: str, values: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE minute_import_runs SET finished_at = ?, status = ?,
                  files_success = ?, files_skipped = ?, files_empty = ?,
                  files_failed = ?, rows_written = ?, rows_filtered = ?, details = ?
                WHERE run_id = ?
                """,
                [
                    datetime.now(), values["status"], values.get("files_success", 0),
                    values.get("files_skipped", 0), values.get("files_empty", 0),
                    values.get("files_failed", 0), values.get("rows_written", 0),
                    values.get("rows_filtered", 0),
                    json.dumps(values.get("details", {}), ensure_ascii=False), run_id,
                ],
            )

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
                """
                INSERT INTO data_fetches
                    (fetched_at, dataset, provider, symbols, start_at, end_at,
                     rows, status, warnings, adjustment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [datetime.now(), values.get("dataset"), values.get("provider"),
                 values.get("symbols"), values.get("start"), values.get("end"),
                 values.get("rows", 0), values.get("status"),
                 json.dumps(values.get("warnings", []), ensure_ascii=False),
                 values.get("adjustment", "none")],
            )
