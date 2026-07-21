from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quant_trade.config import AppConfig
from quant_trade.data.quality import DataQualityError, validate_bars
from quant_trade.models import Adjustment, AssetType


class DataStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.ensure_directories()
        self.root = config.paths.data_dir / "processed"
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = config.paths.database
        self._snapshot_validation_cache: dict[tuple[str, str, date], float] = {}
        self._init_db()
        self._recover_minute_commits()

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
                CREATE TABLE IF NOT EXISTS minute_archive_imports (
                    file_hash VARCHAR, frequency VARCHAR, asset_type VARCHAR,
                    file_name VARCHAR, imported_at TIMESTAMP, rows BIGINT,
                    min_time TIMESTAMP, max_time TIMESTAMP, status VARCHAR,
                    details VARCHAR,
                    PRIMARY KEY (file_hash, frequency, asset_type)
                );
                CREATE TABLE IF NOT EXISTS minute_archive_members (
                    file_hash VARCHAR, frequency VARCHAR, asset_type VARCHAR,
                    symbol VARCHAR, year INTEGER, rows BIGINT,
                    min_time TIMESTAMP, max_time TIMESTAMP,
                    partition_size BIGINT, partition_mtime_ns BIGINT,
                    PRIMARY KEY (file_hash, frequency, asset_type, symbol, year)
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
                CREATE TABLE IF NOT EXISTS minute_source_members (
                    source_path VARCHAR, frequency VARCHAR, symbol VARCHAR,
                    year INTEGER, rows BIGINT, min_time TIMESTAMP,
                    max_time TIMESTAMP, partition_size BIGINT,
                    partition_mtime_ns BIGINT,
                    PRIMARY KEY (source_path, frequency, symbol, year)
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
                CREATE TABLE IF NOT EXISTS daily_coverage (
                    asset_type VARCHAR, adjustment VARCHAR, symbol VARCHAR,
                    trade_date DATE, provider VARCHAR, covered_at TIMESTAMP,
                    PRIMARY KEY (asset_type, adjustment, symbol, trade_date)
                );
                CREATE TABLE IF NOT EXISTS trade_calendar (
                    cal_date DATE PRIMARY KEY, source VARCHAR, updated_at TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS calendar_coverage (
                    cal_date DATE PRIMARY KEY, source VARCHAR, updated_at TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    asset_type VARCHAR, adjustment VARCHAR, trade_date DATE,
                    row_count BIGINT, symbol_count BIGINT, expected_symbols BIGINT,
                    provider VARCHAR, status VARCHAR, checked_at TIMESTAMP,
                    details VARCHAR,
                    PRIMARY KEY (asset_type, adjustment, trade_date)
                );
                CREATE TABLE IF NOT EXISTS market_snapshot_members (
                    asset_type VARCHAR, adjustment VARCHAR, trade_date DATE,
                    symbol VARCHAR, file_size BIGINT, file_mtime_ns BIGINT,
                    PRIMARY KEY (asset_type, adjustment, trade_date, symbol)
                );
                CREATE TABLE IF NOT EXISTS daily_basic_snapshots (
                    trade_date DATE PRIMARY KEY, row_count BIGINT,
                    symbol_count BIGINT, expected_symbols BIGINT,
                    provider VARCHAR, status VARCHAR, checked_at TIMESTAMP,
                    details VARCHAR
                );
                CREATE TABLE IF NOT EXISTS minute_commit_log (
                    commit_id VARCHAR PRIMARY KEY, status VARCHAR,
                    created_at TIMESTAMP, updated_at TIMESTAMP
                );
                ALTER TABLE data_fetches ADD COLUMN IF NOT EXISTS adjustment VARCHAR;
                ALTER TABLE minute_archive_members ADD COLUMN IF NOT EXISTS rows BIGINT;
                ALTER TABLE minute_archive_members ADD COLUMN IF NOT EXISTS min_time TIMESTAMP;
                ALTER TABLE minute_archive_members ADD COLUMN IF NOT EXISTS max_time TIMESTAMP;
                ALTER TABLE minute_archive_members
                    ADD COLUMN IF NOT EXISTS partition_size BIGINT;
                ALTER TABLE minute_archive_members
                    ADD COLUMN IF NOT EXISTS partition_mtime_ns BIGINT;
                ALTER TABLE minute_source_members ADD COLUMN IF NOT EXISTS rows BIGINT;
                ALTER TABLE minute_source_members ADD COLUMN IF NOT EXISTS min_time TIMESTAMP;
                ALTER TABLE minute_source_members ADD COLUMN IF NOT EXISTS max_time TIMESTAMP;
                ALTER TABLE minute_source_members ADD COLUMN IF NOT EXISTS partition_size BIGINT;
                ALTER TABLE minute_source_members
                    ADD COLUMN IF NOT EXISTS partition_mtime_ns BIGINT;
                ALTER TABLE market_snapshot_members ADD COLUMN IF NOT EXISTS file_size BIGINT;
                ALTER TABLE market_snapshot_members ADD COLUMN IF NOT EXISTS file_mtime_ns BIGINT;
                """
            )

    @contextmanager
    def minute_write_lock(self):
        """Serialize minute recovery, merging and commits across processes."""
        lock_path = self.root / ".staging" / "minute-write.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _recover_minute_commits(self) -> None:
        with self.minute_write_lock():
            self._recover_minute_commits_locked()

    def _recover_minute_commits_locked(self) -> None:
        """Finish cleanup or roll back a minute commit interrupted by process death."""
        root = self.root / ".staging" / "minute-commit"
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            states = dict(con.execute("SELECT commit_id, status FROM minute_commit_log").fetchall())
        seen: set[str] = set()
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            commit_id = directory.name
            seen.add(commit_id)
            journal_path = directory / "journal.json"
            if not journal_path.exists():
                if states.get(commit_id) == "committed":
                    shutil.rmtree(directory, ignore_errors=True)
                    if not directory.exists():
                        with self.connect() as con:
                            con.execute(
                                "DELETE FROM minute_commit_log WHERE commit_id = ?", [commit_id]
                            )
                elif commit_id not in states:
                    shutil.rmtree(directory, ignore_errors=True)
                else:
                    raise RuntimeError(f"分钟提交 {commit_id} 缺少恢复日志")
                continue
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            committed = states.get(commit_id) == "committed"
            if not committed:
                for operation in reversed(journal.get("operations", [])):
                    target = Path(operation["target"])
                    backup_value = operation.get("backup")
                    backup = Path(backup_value) if backup_value else None
                    if backup is None:
                        target.unlink(missing_ok=True)
                    elif backup.exists():
                        target.unlink(missing_ok=True)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        backup.replace(target)
                with self.connect() as con:
                    con.execute("DELETE FROM minute_commit_log WHERE commit_id = ?", [commit_id])
            shutil.rmtree(directory, ignore_errors=True)
            if committed and not directory.exists():
                with self.connect() as con:
                    con.execute("DELETE FROM minute_commit_log WHERE commit_id = ?", [commit_id])
        stale_logs = set(states) - seen
        if stale_logs:
            preparing = [commit_id for commit_id in stale_logs if states[commit_id] != "committed"]
            if preparing:
                raise RuntimeError(f"分钟提交缺少恢复目录: {preparing}")
            with self.connect() as con:
                con.executemany(
                    "DELETE FROM minute_commit_log WHERE commit_id = ? AND status = 'committed'",
                    [(commit_id,) for commit_id in stale_logs],
                )

    @staticmethod
    def safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_").replace(".", "_")

    @staticmethod
    def _symbol_digest(symbols: set[str] | list[str]) -> str:
        return hashlib.sha256("\n".join(sorted(set(symbols))).encode()).hexdigest()

    def _atomic_parquet_write(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def daily_path(
        self,
        asset_type: AssetType | str,
        symbol: str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> Path:
        asset = AssetType(asset_type).value
        mode = Adjustment(adjustment).value
        return (
            self.root
            / "daily"
            / asset
            / f"adjustment={mode}"
            / f"{self.safe_symbol(symbol)}.parquet"
        )

    def market_snapshot_complete(
        self,
        asset_type: AssetType | str,
        trade_date: date | str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> bool:
        asset = AssetType(asset_type).value
        mode = Adjustment(adjustment).value
        day = pd.Timestamp(trade_date).date()
        cache_key = (asset, mode, day)
        cached_at = self._snapshot_validation_cache.get(cache_key)
        cache_ttl = self.config.providers.market_snapshot_cache_ttl_seconds
        if cached_at is not None and time.monotonic() - cached_at < cache_ttl:
            return True
        with self.connect() as con:
            row = con.execute(
                """
                SELECT status, symbol_count FROM market_snapshots
                WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                """,
                [
                    asset,
                    mode,
                    day,
                ],
            ).fetchone()
            if not row or row[0] != "complete":
                return False
            if int(row[1]) <= 0:
                return False
            members = con.execute(
                """
                    SELECT symbol, file_size, file_mtime_ns FROM market_snapshot_members
                    WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                    """,
                [asset, mode, day],
            ).fetchall()
        if len(members) != int(row[1]):
            return False
        missing_fingerprints = any(item[1] is None or item[2] is None for item in members)
        fingerprints: list[tuple[int, int, str]] = []
        for symbol, expected_size, expected_mtime_ns in members:
            path = self.daily_path(asset, symbol, mode)
            if not path.exists():
                return False
            stat = path.stat()
            if not missing_fingerprints and (
                stat.st_size != int(expected_size) or stat.st_mtime_ns != int(expected_mtime_ns)
            ):
                return False
            fingerprints.append((stat.st_size, stat.st_mtime_ns, str(symbol)))
        if missing_fingerprints:
            if not self._deep_market_snapshot_valid(
                asset, mode, day, [row[2] for row in fingerprints]
            ):
                return False
            with self.connect() as con:
                con.executemany(
                    """
                    UPDATE market_snapshot_members SET file_size = ?, file_mtime_ns = ?
                    WHERE asset_type = ? AND adjustment = ? AND trade_date = ? AND symbol = ?
                    """,
                    [
                        (size, mtime_ns, asset, mode, day, symbol)
                        for size, mtime_ns, symbol in fingerprints
                    ],
                )
        self._snapshot_validation_cache[cache_key] = time.monotonic()
        return True

    def incomplete_market_snapshot_dates(
        self,
        asset_type: AssetType | str,
        trade_dates: list[date],
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> list[date]:
        """Validate a date range while stat-ing each distinct member file only once."""
        days = sorted(set(trade_dates))
        if not days:
            return []
        asset = AssetType(asset_type).value
        mode = Adjustment(adjustment).value
        requested = pd.DataFrame({"trade_date": days})
        with self.connect() as con:
            con.register("requested_snapshot_days", requested)
            snapshot_rows = con.execute(
                """
                SELECT r.trade_date, s.status, s.symbol_count, COUNT(m.symbol) AS members,
                       SUM(CASE WHEN m.file_size IS NULL OR m.file_mtime_ns IS NULL
                           THEN 1 ELSE 0 END) AS missing_fingerprints
                FROM requested_snapshot_days r
                LEFT JOIN market_snapshots s ON s.trade_date = r.trade_date
                  AND s.asset_type = ? AND s.adjustment = ?
                LEFT JOIN market_snapshot_members m ON m.trade_date = r.trade_date
                  AND m.asset_type = ? AND m.adjustment = ?
                GROUP BY r.trade_date, s.status, s.symbol_count
                """,
                [asset, mode, asset, mode],
            ).fetchall()
            fingerprints = con.execute(
                """
                SELECT m.symbol, MIN(m.file_size), MAX(m.file_size),
                       MIN(m.file_mtime_ns), MAX(m.file_mtime_ns)
                FROM market_snapshot_members m
                JOIN requested_snapshot_days r ON r.trade_date = m.trade_date
                WHERE m.asset_type = ? AND m.adjustment = ?
                GROUP BY m.symbol
                """,
                [asset, mode],
            ).fetchall()
        invalid = {
            pd.Timestamp(day).date()
            for day, status, symbol_count, members, missing in snapshot_rows
            if status != "complete" or not symbol_count or int(members) != int(symbol_count)
        }
        legacy_days = {
            pd.Timestamp(day).date()
            for day, status, symbol_count, members, missing in snapshot_rows
            if status == "complete"
            and symbol_count
            and int(members) == int(symbol_count)
            and int(missing or 0) > 0
        }
        bad_symbols: list[str] = []
        for symbol, min_size, max_size, min_mtime, max_mtime in fingerprints:
            if min_size is None or min_mtime is None:
                continue
            if min_size != max_size or min_mtime != max_mtime:
                bad_symbols.append(str(symbol))
                continue
            try:
                stat = self.daily_path(asset, str(symbol), mode).stat()
            except OSError:
                bad_symbols.append(str(symbol))
                continue
            if stat.st_size != int(min_size) or stat.st_mtime_ns != int(min_mtime):
                bad_symbols.append(str(symbol))
        if bad_symbols:
            with self.connect() as con:
                con.register("requested_snapshot_days", requested)
                con.register("bad_snapshot_symbols", pd.DataFrame({"symbol": bad_symbols}))
                rows = con.execute(
                    """
                    SELECT DISTINCT m.trade_date
                    FROM market_snapshot_members m
                    JOIN requested_snapshot_days r ON r.trade_date = m.trade_date
                    JOIN bad_snapshot_symbols b ON b.symbol = m.symbol
                    WHERE m.asset_type = ? AND m.adjustment = ?
                    """,
                    [asset, mode],
                ).fetchall()
            invalid.update(pd.Timestamp(row[0]).date() for row in rows)
        for day in legacy_days:
            if not self.market_snapshot_complete(asset, day, mode):
                invalid.add(day)
        now = time.monotonic()
        for day in set(days) - invalid:
            self._snapshot_validation_cache[(asset, mode, day)] = now
        return sorted(invalid)

    def _deep_market_snapshot_valid(
        self,
        asset_type: AssetType | str,
        adjustment: Adjustment | str,
        trade_date: date | str,
        symbols: list[str],
    ) -> bool:
        if not symbols:
            return False
        paths = [self.daily_path(asset_type, symbol, adjustment) for symbol in symbols]
        if any(not path.exists() for path in paths):
            return False
        try:
            with duckdb.connect() as con:
                frame = con.execute(
                    """
                    SELECT *
                    FROM read_parquet(?, hive_partitioning = false, union_by_name = true)
                    WHERE CAST(trade_date AS DATE) = ?
                    """,
                    [[str(path) for path in paths], pd.Timestamp(trade_date).date()],
                ).df()
            validate_bars(frame)
            actual = set(frame["symbol"].astype(str))
            if "asset_type" in frame and set(frame["asset_type"].astype(str)) != {
                AssetType(asset_type).value
            }:
                return False
            if "adjustment" in frame and set(frame["adjustment"].astype(str)) != {
                Adjustment(adjustment).value
            }:
                return False
        except Exception:
            return False
        return actual == set(symbols)

    @staticmethod
    def _market_snapshot_validation_sample(
        symbols: list[str], trade_date: date, sample_size: int | None
    ) -> list[str]:
        """Choose a stable content-validation sample independent of input order."""
        if sample_size is None or sample_size >= len(symbols):
            return sorted(symbols)
        if sample_size <= 0:
            return []
        return sorted(
            symbols,
            key=lambda symbol: hashlib.sha256(
                f"{trade_date.isoformat()}:{symbol}".encode()
            ).digest(),
        )[:sample_size]

    def market_snapshot_symbols(
        self,
        asset_type: AssetType | str,
        trade_date: date | str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> set[str]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT symbol FROM market_snapshot_members
                WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                """,
                [
                    AssetType(asset_type).value,
                    Adjustment(adjustment).value,
                    pd.Timestamp(trade_date).date(),
                ],
            ).fetchall()
        return {str(row[0]) for row in rows}

    def mark_market_snapshot(
        self,
        asset_type: AssetType | str,
        trade_date: date | str,
        adjustment: Adjustment | str = Adjustment.NONE,
        *,
        row_count: int = 0,
        symbol_count: int = 0,
        expected_symbols: int = 0,
        provider: str = "manual",
        status: str = "complete",
        details: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        validation_sample_size: int | None = None,
        preserve_existing_complete: bool = False,
    ) -> bool:
        if status not in {"complete", "incomplete"}:
            raise ValueError("快照状态只支持 complete/incomplete")
        asset = AssetType(asset_type).value
        mode = Adjustment(adjustment).value
        day = pd.Timestamp(trade_date).date()
        member_symbols = sorted(set(symbols or []))
        fingerprints: list[tuple[str, int, int]] = []
        validation_details = dict(details or {})
        if status == "complete":
            sample = self._market_snapshot_validation_sample(
                member_symbols, day, validation_sample_size
            )
            validation_details.update(
                {
                    "content_validation": (
                        "full"
                        if len(sample) == len(member_symbols)
                        else "sampled"
                        if sample
                        else "existence_only"
                    ),
                    "validation_sample_size": len(sample),
                }
            )
            paths_exist = True
            try:
                for symbol in member_symbols:
                    stat = self.daily_path(asset, symbol, mode).stat()
                    fingerprints.append((symbol, stat.st_size, stat.st_mtime_ns))
            except OSError:
                paths_exist = False
            if (
                len(member_symbols) != int(symbol_count)
                or not paths_exist
                or (sample and not self._deep_market_snapshot_valid(asset, mode, day, sample))
            ):
                status = "incomplete"
                validation_details["content_validation_failed"] = True
                fingerprints = []
        if status == "incomplete" and preserve_existing_complete:
            with self.connect() as con:
                existing = con.execute(
                    """
                    SELECT status FROM market_snapshots
                    WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                    """,
                    [asset, mode, day],
                ).fetchone()
            if existing and existing[0] == "complete":
                return False
        self._snapshot_validation_cache.pop((asset, mode, day), None)
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO market_snapshots
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    asset,
                    mode,
                    day,
                    row_count,
                    symbol_count,
                    expected_symbols,
                    provider,
                    status,
                    datetime.now(),
                    json.dumps(validation_details, ensure_ascii=False),
                ],
            )
            con.execute(
                """
                DELETE FROM market_snapshot_members
                WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                """,
                [
                    asset,
                    mode,
                    day,
                ],
            )
            if member_symbols:
                fingerprint_by_symbol = {
                    symbol: (file_size, file_mtime_ns)
                    for symbol, file_size, file_mtime_ns in fingerprints
                }
                con.executemany(
                    "INSERT INTO market_snapshot_members VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            asset,
                            mode,
                            day,
                            symbol,
                            fingerprint_by_symbol.get(symbol, (None, None))[0],
                            fingerprint_by_symbol.get(symbol, (None, None))[1],
                        )
                        for symbol in member_symbols
                    ],
                )
        return status == "complete"

    def audit_market_snapshots(
        self,
        asset_type: AssetType | str,
        adjustment: Adjustment | str = Adjustment.NONE,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        mark_incomplete: bool = True,
    ) -> list[dict[str, Any]]:
        """Deep-audit every member of completed snapshots and refresh fingerprints."""
        asset = AssetType(asset_type).value
        mode = Adjustment(adjustment).value
        clauses = ["asset_type = ?", "adjustment = ?", "status = 'complete'"]
        params: list[Any] = [asset, mode]
        if start is not None:
            clauses.append("trade_date >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            clauses.append("trade_date <= ?")
            params.append(pd.Timestamp(end).date())
        with self.connect() as con:
            snapshots = con.execute(
                "SELECT trade_date, symbol_count, details FROM market_snapshots WHERE "
                + " AND ".join(clauses)
                + " ORDER BY trade_date",
                params,
            ).fetchall()
        results: list[dict[str, Any]] = []
        for raw_day, expected_count, raw_details in snapshots:
            day = pd.Timestamp(raw_day).date()
            symbols = sorted(self.market_snapshot_symbols(asset, day, mode))
            valid = len(symbols) == int(expected_count) and self._deep_market_snapshot_valid(
                asset, mode, day, symbols
            )
            result = {
                "trade_date": day.isoformat(),
                "symbol_count": int(expected_count),
                "status": "valid" if valid else "invalid",
            }
            results.append(result)
            cache_key = (asset, mode, day)
            self._snapshot_validation_cache.pop(cache_key, None)
            with self.connect() as con:
                if valid:
                    fingerprints = []
                    for symbol in symbols:
                        stat = self.daily_path(asset, symbol, mode).stat()
                        fingerprints.append((stat.st_size, stat.st_mtime_ns, symbol))
                    con.executemany(
                        """
                        UPDATE market_snapshot_members SET file_size = ?, file_mtime_ns = ?
                        WHERE asset_type = ? AND adjustment = ? AND trade_date = ? AND symbol = ?
                        """,
                        [
                            (size, mtime, asset, mode, day, symbol)
                            for size, mtime, symbol in fingerprints
                        ],
                    )
                    self._snapshot_validation_cache[cache_key] = time.monotonic()
                elif mark_incomplete:
                    try:
                        details = json.loads(raw_details or "{}")
                    except json.JSONDecodeError:
                        details = {}
                    details.update(
                        {
                            "deep_audit_failed": True,
                            "deep_audited_at": datetime.now().isoformat(),
                        }
                    )
                    con.execute(
                        """
                        UPDATE market_snapshots SET status = 'incomplete', checked_at = ?,
                          details = ?
                        WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                        """,
                        [datetime.now(), json.dumps(details, ensure_ascii=False), asset, mode, day],
                    )
        return results

    def market_snapshot_symbol_count(
        self,
        asset_type: AssetType | str,
        trade_date: date | str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> int:
        if not self.market_snapshot_complete(asset_type, trade_date, adjustment):
            return 0
        with self.connect() as con:
            row = con.execute(
                """
                SELECT symbol_count FROM market_snapshots
                WHERE asset_type = ? AND adjustment = ? AND trade_date = ?
                  AND status = 'complete'
                """,
                [
                    AssetType(asset_type).value,
                    Adjustment(adjustment).value,
                    pd.Timestamp(trade_date).date(),
                ],
            ).fetchone()
        return int(row[0]) if row else 0

    def latest_complete_snapshot_symbol_count(
        self,
        asset_type: AssetType | str,
        adjustment: Adjustment | str,
        before: date,
    ) -> int:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT symbol_count FROM market_snapshots
                WHERE asset_type = ? AND adjustment = ? AND trade_date < ?
                  AND status = 'complete'
                ORDER BY trade_date DESC LIMIT 1
                """,
                [AssetType(asset_type).value, Adjustment(adjustment).value, before],
            ).fetchone()
        return int(row[0]) if row else 0

    def latest_complete_market_snapshot_date(
        self,
        asset_type: AssetType | str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> date | None:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT trade_date FROM market_snapshots
                WHERE asset_type = ? AND adjustment = ? AND status = 'complete'
                ORDER BY trade_date DESC
                """,
                [AssetType(asset_type).value, Adjustment(adjustment).value],
            ).fetchall()
        for (trade_date,) in rows:
            day = pd.Timestamp(trade_date).date()
            if self.market_snapshot_complete(asset_type, day, adjustment):
                return day
        return None

    def daily_basic_symbol_count(self, trade_date: date | str) -> int:
        path = self.daily_basic_path(pd.Timestamp(trade_date).strftime("%Y-%m-%d"))
        if not path.exists():
            return 0
        frame = pd.read_parquet(path, columns=["symbol"])
        return int(frame["symbol"].nunique())

    def daily_basic_symbols(self, trade_date: date | str) -> set[str]:
        path = self.daily_basic_path(pd.Timestamp(trade_date).strftime("%Y-%m-%d"))
        if not path.exists():
            return set()
        frame = pd.read_parquet(path, columns=["symbol"])
        return set(frame["symbol"].dropna().astype(str))

    def daily_basic_complete(self, trade_date: date | str) -> bool:
        day = pd.Timestamp(trade_date).date()
        path = self.daily_basic_path(str(day))
        if not path.exists():
            return False
        with self.connect() as con:
            row = con.execute(
                "SELECT status, symbol_count, details FROM daily_basic_snapshots "
                "WHERE trade_date = ?",
                [day],
            ).fetchone()
        if not row or row[0] != "complete":
            return False
        try:
            frame = pd.read_parquet(path, columns=["symbol", "trade_date", "total_mv"])
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
            frame["total_mv"] = pd.to_numeric(frame["total_mv"], errors="coerce")
            invalid = (
                frame[["symbol", "trade_date", "total_mv"]].isna().any(axis=1)
                | frame["symbol"].astype(str).str.strip().eq("")
                | ~np.isfinite(frame["total_mv"])
                | frame["total_mv"].le(0)
            )
            if (
                frame.empty
                or invalid.any()
                or frame.duplicated(["symbol", "trade_date"]).any()
                or set(frame["trade_date"].dt.date) != {day}
            ):
                return False
            symbols = set(frame["symbol"].astype(str))
        except Exception:
            return False
        try:
            details = json.loads(row[2] or "{}")
        except json.JSONDecodeError:
            return False
        return (
            int(row[1]) > 0
            and len(symbols) == int(row[1])
            and details.get("symbol_digest") == self._symbol_digest(symbols)
        )

    def incomplete_daily_basic_dates(self, trade_dates: list[date]) -> list[date]:
        days = sorted(set(trade_dates))
        if not days:
            return []
        requested = pd.DataFrame({"trade_date": days})
        with self.connect() as con:
            con.register("requested_basic_days", requested)
            rows = con.execute(
                """
                SELECT r.trade_date, s.status, s.symbol_count, s.details
                FROM requested_basic_days r
                LEFT JOIN daily_basic_snapshots s USING (trade_date)
                """
            ).fetchall()
        invalid: set[date] = set()
        metadata: dict[date, tuple[int, str]] = {}
        paths: list[str] = []
        path_days: dict[str, date] = {}
        for raw_day, status, symbol_count, raw_details in rows:
            day = pd.Timestamp(raw_day).date()
            path = self.daily_basic_path(str(day)).resolve()
            try:
                details = json.loads(raw_details or "{}")
            except json.JSONDecodeError:
                details = {}
            digest = details.get("symbol_digest")
            if status != "complete" or not symbol_count or not digest or not path.exists():
                invalid.add(day)
                continue
            metadata[day] = (int(symbol_count), str(digest))
            paths.append(str(path))
            path_days[str(path)] = day
        if paths:
            try:
                with duckdb.connect() as con:
                    aggregates = con.execute(
                        """
                        SELECT filename, MIN(CAST(trade_date AS DATE)),
                          MAX(CAST(trade_date AS DATE)), COUNT(*),
                          COUNT(DISTINCT symbol),
                          COUNT_IF(symbol IS NULL OR TRIM(CAST(symbol AS VARCHAR)) = ''
                            OR trade_date IS NULL OR total_mv IS NULL
                            OR NOT isfinite(TRY_CAST(total_mv AS DOUBLE))
                            OR TRY_CAST(total_mv AS DOUBLE) <= 0),
                          sha256(string_agg(CAST(symbol AS VARCHAR), chr(10)
                            ORDER BY CAST(symbol AS VARCHAR)))
                        FROM read_parquet(?, hive_partitioning = false,
                          union_by_name = true, filename = true)
                        GROUP BY filename
                        """,
                        [paths],
                    ).fetchall()
                seen: set[date] = set()
                for filename, min_day, max_day, count, unique_count, bad, digest in aggregates:
                    day = path_days[str(Path(filename).resolve())]
                    seen.add(day)
                    expected_count, expected_digest = metadata[day]
                    if (
                        pd.Timestamp(min_day).date() != day
                        or pd.Timestamp(max_day).date() != day
                        or int(count) != expected_count
                        or int(unique_count) != expected_count
                        or int(bad or 0)
                        or digest != expected_digest
                    ):
                        invalid.add(day)
                invalid.update(set(metadata) - seen)
            except Exception:
                invalid.update(day for day in metadata if not self.daily_basic_complete(day))
        return sorted(invalid)

    def audit_daily_basic_snapshots(
        self,
        *,
        start: date | str | None = None,
        end: date | str | None = None,
        mark_incomplete: bool = True,
    ) -> list[dict[str, Any]]:
        clauses = ["status = 'complete'"]
        params: list[Any] = []
        if start is not None:
            clauses.append("trade_date >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            clauses.append("trade_date <= ?")
            params.append(pd.Timestamp(end).date())
        with self.connect() as con:
            rows = con.execute(
                "SELECT trade_date FROM daily_basic_snapshots WHERE "
                + " AND ".join(clauses)
                + " ORDER BY trade_date",
                params,
            ).fetchall()
        results = []
        for (raw_day,) in rows:
            day = pd.Timestamp(raw_day).date()
            valid = self.daily_basic_complete(day)
            results.append(
                {"trade_date": day.isoformat(), "status": "valid" if valid else "invalid"}
            )
            if not valid and mark_incomplete:
                with self.connect() as con:
                    con.execute(
                        "UPDATE daily_basic_snapshots SET status = 'incomplete', checked_at = ? "
                        "WHERE trade_date = ?",
                        [datetime.now(), day],
                    )
        return results

    @classmethod
    def symbol_digest(cls, symbols: set[str]) -> str:
        return cls._symbol_digest(symbols)

    def mark_daily_basic_snapshot(
        self,
        trade_date: date | str,
        *,
        row_count: int,
        symbol_count: int,
        expected_symbols: int,
        provider: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if status not in {"observed", "complete", "incomplete"}:
            raise ValueError("daily_basic 状态只支持 observed/complete/incomplete")
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO daily_basic_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    pd.Timestamp(trade_date).date(),
                    row_count,
                    symbol_count,
                    expected_symbols,
                    provider,
                    status,
                    datetime.now(),
                    json.dumps(details or {}, ensure_ascii=False),
                ],
            )

    @staticmethod
    def _with_daily_metadata(
        df: pd.DataFrame, asset_type: AssetType | str | None = None
    ) -> pd.DataFrame:
        if "adjustment" not in df:
            df = df.assign(adjustment="none")
        else:
            df = df.copy()
            df["adjustment"] = df["adjustment"].fillna("none")
        df["adjustment"] = df["adjustment"].map(lambda value: Adjustment(value).value)
        if asset_type is not None:
            asset = AssetType(asset_type).value
            if "asset_type" in df:
                actual = set(df["asset_type"].dropna().astype(str).unique())
                if actual and actual != {asset}:
                    raise ValueError(f"行情资产类型 {sorted(actual)} 与写入目录 {asset} 不一致")
            df["asset_type"] = asset
        return df

    def write_daily(self, df: pd.DataFrame, asset_type: AssetType | str) -> int:
        count = 0
        asset = AssetType(asset_type).value
        fingerprints: list[tuple[int, int, str, str, str]] = []
        work = self._with_daily_metadata(df, asset_type)
        for (symbol, adjustment), part in work.groupby(["symbol", "adjustment"], sort=False):
            path = self.daily_path(asset_type, str(symbol), str(adjustment))
            if path.exists():
                old = pd.read_parquet(path)
                part = self._with_daily_metadata(
                    pd.concat([old, part], ignore_index=True), asset_type
                )
            part = part.sort_values("trade_date").drop_duplicates(
                ["symbol", "trade_date", "adjustment"], keep="last"
            )
            self._atomic_parquet_write(part, path)
            stat = path.stat()
            fingerprints.append(
                (stat.st_size, stat.st_mtime_ns, asset, str(adjustment), str(symbol))
            )
            count += len(part)
        if fingerprints:
            with self.connect() as con:
                con.executemany(
                    """
                    UPDATE market_snapshot_members SET file_size = ?, file_mtime_ns = ?
                    WHERE asset_type = ? AND adjustment = ? AND symbol = ?
                    """,
                    fingerprints,
                )
            self._snapshot_validation_cache.clear()
        return count

    def replace_daily(self, df: pd.DataFrame, asset_type: AssetType | str) -> int:
        """Atomically replace each symbol/adjustment file represented by *df*."""
        if df is None or df.empty:
            return 0
        count = 0
        asset = AssetType(asset_type).value
        fingerprints: list[tuple[int, int, str, str, str]] = []
        work = self._with_daily_metadata(df, asset_type)
        for (symbol, adjustment), part in work.groupby(["symbol", "adjustment"], sort=False):
            part = part.sort_values("trade_date").drop_duplicates(
                ["symbol", "trade_date", "adjustment"], keep="last"
            )
            path = self.daily_path(asset_type, str(symbol), str(adjustment))
            self._atomic_parquet_write(part, path)
            stat = path.stat()
            fingerprints.append(
                (stat.st_size, stat.st_mtime_ns, asset, str(adjustment), str(symbol))
            )
            count += len(part)
        with self.connect() as con:
            con.executemany(
                """
                UPDATE market_snapshot_members SET file_size = ?, file_mtime_ns = ?
                WHERE asset_type = ? AND adjustment = ? AND symbol = ?
                """,
                fingerprints,
            )
        self._snapshot_validation_cache.clear()
        return count

    def read_daily(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        *,
        asset_type: AssetType | str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> pd.DataFrame:
        asset = AssetType(asset_type)
        mode = Adjustment(adjustment)
        daily_root = self.root / "daily" / asset.value / f"adjustment={mode.value}"
        paths = (
            [self.daily_path(asset, symbol, mode) for symbol in symbols]
            if symbols
            else list(daily_root.glob("*.parquet"))
        )
        frames = [pd.read_parquet(path) for path in paths if path.exists()]
        if not frames:
            return pd.DataFrame()
        out = self._with_daily_metadata(pd.concat(frames, ignore_index=True), asset)
        out = out[out["adjustment"] == mode.value]
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        if start:
            out = out[out["trade_date"] >= pd.Timestamp(start)]
        if end:
            out = out[out["trade_date"] <= pd.Timestamp(end)]
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def read_daily_dates(
        self,
        symbols: list[str],
        trade_dates: list[date | str],
        *,
        asset_type: AssetType | str,
        adjustment: Adjustment | str = Adjustment.NONE,
    ) -> pd.DataFrame:
        """Read only selected daily dates, with filtering pushed into DuckDB/Parquet."""
        days = sorted({pd.Timestamp(value).date() for value in trade_dates})
        if not days:
            return pd.DataFrame()
        asset = AssetType(asset_type)
        mode = Adjustment(adjustment)
        daily_root = self.root / "daily" / asset.value / f"adjustment={mode.value}"
        paths = (
            [self.daily_path(asset, symbol, mode) for symbol in symbols]
            if symbols
            else list(daily_root.glob("*.parquet"))
        )
        existing = [str(path) for path in paths if path.exists()]
        if not existing:
            return pd.DataFrame()
        placeholders = ", ".join("?" for _ in days)
        with duckdb.connect() as con:
            out = con.execute(
                f"""
                SELECT * FROM read_parquet(?, hive_partitioning = false, union_by_name = true)
                WHERE CAST(trade_date AS DATE) IN ({placeholders})
                """,
                [existing, *days],
            ).df()
        if out.empty:
            return out
        out = self._with_daily_metadata(out, asset)
        out = out[out["adjustment"] == mode.value]
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def confirmed_empty_daily_dates(
        self,
        asset_type: AssetType | str,
        adjustment: Adjustment | str,
        symbol: str,
        start: date,
        end: date,
    ) -> set[date]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT trade_date FROM daily_coverage
                WHERE asset_type = ? AND adjustment = ? AND symbol = ?
                  AND trade_date BETWEEN ? AND ?
                  AND provider = 'empty'
                """,
                [
                    AssetType(asset_type).value,
                    Adjustment(adjustment).value,
                    symbol,
                    start,
                    end,
                ],
            ).fetchall()
        return {pd.Timestamp(row[0]).date() for row in rows}

    def mark_daily_empty_dates(
        self,
        asset_type: AssetType | str,
        adjustment: Adjustment | str,
        symbol: str,
        days: list[date],
    ) -> None:
        if not days:
            return
        asset, mode = AssetType(asset_type).value, Adjustment(adjustment).value
        now = datetime.now()
        with self.connect() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO daily_coverage
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(asset, mode, symbol, day, "empty", now) for day in days],
            )

    def covered_calendar_dates(
        self,
        start: date,
        end: date,
        *,
        mutable_from: date,
        mutable_ttl: timedelta,
        now: datetime | None = None,
    ) -> set[date]:
        cutoff = (now or datetime.now()) - mutable_ttl
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT cal_date FROM calendar_coverage
                WHERE cal_date BETWEEN ? AND ?
                  AND (cal_date < ? OR updated_at >= ?)
                """,
                [start, end, mutable_from, cutoff],
            ).fetchall()
        return {pd.Timestamp(row[0]).date() for row in rows}

    def read_trading_days(self, start: date, end: date) -> list[date]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT cal_date FROM trade_calendar
                WHERE cal_date BETWEEN ? AND ? ORDER BY cal_date
                """,
                [start, end],
            ).fetchall()
        return [pd.Timestamp(row[0]).date() for row in rows]

    def calendar_range_complete(self, start: date, end: date) -> bool:
        if start > end:
            return True
        with self.connect() as con:
            count = con.execute(
                """
                SELECT count(DISTINCT cal_date) FROM calendar_coverage
                WHERE cal_date BETWEEN ? AND ?
                """,
                [start, end],
            ).fetchone()[0]
        return int(count) == (end - start).days + 1

    def write_trade_calendar(
        self,
        open_days: list[date],
        covered_days: list[date],
        source: str,
    ) -> None:
        covered_days = sorted(set(covered_days))
        open_days = sorted(set(open_days))
        if not covered_days:
            return
        now = datetime.now()
        first, last = min(covered_days), max(covered_days)
        with self.connect() as con:
            con.execute(
                "DELETE FROM trade_calendar WHERE cal_date BETWEEN ? AND ?",
                [first, last],
            )
            if open_days:
                con.executemany(
                    "INSERT INTO trade_calendar VALUES (?, ?, ?)",
                    [(day, source, now) for day in open_days],
                )
            con.executemany(
                "INSERT OR REPLACE INTO calendar_coverage VALUES (?, ?, ?)",
                [(day, source, now) for day in covered_days],
            )

    def daily_basic_path(self, trade_date: str) -> Path:
        return self.root / "daily_basic" / f"trade_date={trade_date}" / "data.parquet"

    def write_daily_basic(self, df: pd.DataFrame, *, replace_dates: bool = False) -> int:
        if df is None or df.empty:
            return 0
        work = df.rename(columns={"ts_code": "symbol"}).copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"].astype(str))
        if "total_mv" not in work:
            raise ValueError("daily_basic 缺少 total_mv")
        for trade_date, part in work.groupby(work["trade_date"].dt.strftime("%Y-%m-%d")):
            path = self.daily_basic_path(trade_date)
            if path.exists() and not replace_dates:
                part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
            self._atomic_parquet_write(
                part.drop_duplicates(["symbol", "trade_date"], keep="last"), path
            )
        return len(work)

    def read_daily_basic(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        start_day = pd.Timestamp(start).date() if start else None
        end_day = pd.Timestamp(end).date() if end else None
        paths = []
        for path in (self.root / "daily_basic").glob("trade_date=*/data.parquet"):
            try:
                day = pd.Timestamp(path.parent.name.split("=", 1)[1]).date()
            except (IndexError, ValueError):
                continue
            if (start_day is None or day >= start_day) and (end_day is None or day <= end_day):
                paths.append(path)
        if not paths:
            return pd.DataFrame()
        out = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        if start:
            out = out[out["trade_date"] >= pd.Timestamp(start)]
        if end:
            out = out[out["trade_date"] <= pd.Timestamp(end)]
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    def audit_minute_partitions(self, frequency: str) -> list[dict[str, Any]]:
        """Deep-check catalog metadata, keys and market-data quality for minute files."""
        minutes = int(frequency.removesuffix("min"))
        with self.connect() as con:
            catalog = con.execute(
                """
                SELECT symbol, year, asset_type, path, rows, min_time, max_time
                FROM minute_partitions WHERE frequency = ? ORDER BY symbol, year
                """,
                [frequency],
            ).fetchall()
        results: list[dict[str, Any]] = []
        for symbol, year, asset_type, raw_path, rows, min_time, max_time in catalog:
            path = Path(raw_path)
            reason = ""
            try:
                if not path.exists():
                    raise DataQualityError("文件不存在")
                with duckdb.connect() as con:
                    values = con.execute(
                        """
                        SELECT COUNT(*), MIN(bar_time), MAX(bar_time),
                          COUNT(DISTINCT symbol), MIN(symbol), MAX(symbol),
                          COUNT(DISTINCT frequency), MIN(frequency), MAX(frequency),
                          COUNT(DISTINCT asset_type), MIN(asset_type), MAX(asset_type),
                          MIN(YEAR(bar_time)), MAX(YEAR(bar_time)),
                          COUNT_IF(symbol IS NULL OR bar_time IS NULL OR trade_date IS NULL
                            OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                            OR volume IS NULL),
                          COUNT_IF(NOT isfinite(open) OR NOT isfinite(high)
                            OR NOT isfinite(low) OR NOT isfinite(close)
                            OR NOT isfinite(volume) OR open <= 0 OR high <= 0
                            OR low <= 0 OR close <= 0 OR volume < 0
                            OR high < GREATEST(open, close, low)
                            OR low > LEAST(open, close, high)
                            OR (amount IS NOT NULL AND
                              (NOT isfinite(amount) OR amount < 0))),
                          COUNT_IF(CAST(trade_date AS DATE) != CAST(bar_time AS DATE)),
                          COUNT_IF(NOT (
                            (asset_type = 'stock'
                              AND EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) = 570)
                            OR (
                              EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) > 570
                              AND EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) <= 690
                              AND MOD(EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) - 570, ?) = 0
                            ) OR (
                              EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) >= 780
                              AND EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) <= 900
                              AND MOD(EXTRACT(HOUR FROM bar_time) * 60
                                + EXTRACT(MINUTE FROM bar_time) - 780, ?) = 0
                            )
                          ) OR EXTRACT(SECOND FROM bar_time) != 0),
                          COUNT_IF(is_auction IS DISTINCT FROM (
                            asset_type = 'stock'
                            AND EXTRACT(HOUR FROM bar_time) = 9
                            AND EXTRACT(MINUTE FROM bar_time) = 30
                            AND EXTRACT(SECOND FROM bar_time) = 0
                          ))
                        FROM read_parquet(?, hive_partitioning = false)
                        """,
                        [minutes, minutes, str(path)],
                    ).fetchone()
                    duplicates = con.execute(
                        """
                        SELECT COUNT(*) FROM (
                          SELECT symbol, bar_time FROM read_parquet(?, hive_partitioning = false)
                          GROUP BY symbol, bar_time HAVING COUNT(*) > 1
                        )
                        """,
                        [str(path)],
                    ).fetchone()[0]
                actual_rows = int(values[0])
                actual_min, actual_max = values[1], values[2]
                if actual_rows != int(rows) or actual_rows <= 0:
                    raise DataQualityError(f"行数不匹配: catalog={rows}, file={actual_rows}")
                if pd.Timestamp(actual_min) != pd.Timestamp(min_time) or pd.Timestamp(
                    actual_max
                ) != pd.Timestamp(max_time):
                    raise DataQualityError("起止时间与目录不匹配")
                if values[3:6] != (1, symbol, symbol):
                    raise DataQualityError("证券代码与目录不匹配")
                if values[6:9] != (1, frequency, frequency):
                    raise DataQualityError("频率与目录不匹配")
                if values[9:12] != (1, asset_type, asset_type):
                    raise DataQualityError("资产类型与目录不匹配")
                if values[12:14] != (int(year), int(year)):
                    raise DataQualityError("年份与分区不匹配")
                if int(values[14] or 0) or int(values[15] or 0):
                    raise DataQualityError("存在空关键字段或非法 OHLCV")
                if int(values[16] or 0):
                    raise DataQualityError("trade_date 与 bar_time 日期不一致")
                if int(values[17] or 0):
                    raise DataQualityError(f"存在不符合{frequency}交易时段的时间")
                if int(values[18] or 0):
                    raise DataQualityError("is_auction 与时间或资产类型不一致")
                if int(duplicates):
                    raise DataQualityError(f"存在 {duplicates} 组重复时间键")
            except Exception as exc:
                reason = str(exc)
            results.append(
                {
                    "frequency": frequency,
                    "symbol": str(symbol),
                    "year": int(year),
                    "path": str(path),
                    "status": "invalid" if reason else "valid",
                    "reason": reason,
                }
            )
        return results

    def minute_partition(self, trade_date: str) -> Path:
        """Legacy 1-minute layout kept for backward compatibility."""
        return self.root / "minute" / "1min" / f"trade_date={trade_date}" / "bars.parquet"

    def minute_symbol_year_path(self, frequency: str, symbol: str, year: int) -> Path:
        return (
            self.root
            / "minute"
            / f"frequency={frequency}"
            / f"symbol={self.safe_symbol(symbol)}"
            / f"year={year}.parquet"
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
        source_record: dict[str, Any] | None = None,
    ) -> None:
        """Replace one symbol mirror through the shared batch commit path."""
        self.commit_minute_batch(
            [
                {
                    "frequency": frequency,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "staged": staged,
                    "statistics": statistics,
                    "source_hash": source_hash,
                    "replace_symbol": True,
                }
            ],
            source_record=source_record,
        )

    def commit_minute_batch(
        self,
        entries: list[dict[str, Any]],
        *,
        assume_locked: bool = False,
        source_record: dict[str, Any] | None = None,
        archive_record: dict[str, Any] | None = None,
    ) -> None:
        if assume_locked:
            self._commit_minute_batch_locked(
                entries, source_record=source_record, archive_record=archive_record
            )
            return
        with self.minute_write_lock():
            self._commit_minute_batch_locked(
                entries, source_record=source_record, archive_record=archive_record
            )

    def _commit_minute_batch_locked(
        self,
        entries: list[dict[str, Any]],
        *,
        source_record: dict[str, Any] | None = None,
        archive_record: dict[str, Any] | None = None,
    ) -> None:
        """Commit staged symbol/year files and their catalog rows as one recoverable batch."""
        if not entries:
            return
        commit_id = uuid.uuid4().hex
        backup_root = self.root / ".staging" / "minute-commit" / commit_id
        operations: list[dict[str, str | None]] = []
        seen_targets: set[Path] = set()
        for entry in entries:
            frequency = str(entry["frequency"])
            symbol = str(entry["symbol"])
            target_dir = self.minute_symbol_year_path(frequency, symbol, 2000).parent
            existing_years: set[int] = set()
            if entry.get("replace_symbol", True):
                for path in target_dir.glob("year=*.parquet"):
                    try:
                        existing_years.add(int(path.stem.split("=", 1)[1]))
                    except (IndexError, ValueError):
                        continue
            for year in sorted(set(entry["staged"]) | existing_years):
                target = self.minute_symbol_year_path(frequency, symbol, year).resolve()
                if target in seen_targets:
                    raise ValueError(f"分钟提交包含重复目标分区: {target}")
                seen_targets.add(target)
                backup = backup_root / "backups" / f"{len(operations):08d}.parquet"
                operations.append(
                    {
                        "target": str(target),
                        "backup": str(backup) if target.exists() else None,
                    }
                )
        backup_root.mkdir(parents=True, exist_ok=True)
        journal_path = backup_root / "journal.json"
        journal_tmp = journal_path.with_suffix(".tmp")
        journal_tmp.write_text(
            json.dumps({"commit_id": commit_id, "operations": operations}, ensure_ascii=False),
            encoding="utf-8",
        )
        journal_tmp.replace(journal_path)
        with self.connect() as log_con:
            log_con.execute(
                "INSERT INTO minute_commit_log VALUES (?, 'preparing', ?, ?)",
                [commit_id, datetime.now(), datetime.now()],
            )
        con = self.connect()
        try:
            operation_by_target = {Path(item["target"]): item for item in operations}
            for entry in entries:
                frequency = str(entry["frequency"])
                symbol = str(entry["symbol"])
                staged: dict[int, Path] = entry["staged"]
                target_dir = self.minute_symbol_year_path(frequency, symbol, 2000).parent
                target_dir.mkdir(parents=True, exist_ok=True)
                new_years = set(staged)
                existing_years: set[int] = set()
                for path in target_dir.glob("year=*.parquet"):
                    try:
                        existing_years.add(int(path.stem.split("=", 1)[1]))
                    except (IndexError, ValueError):
                        continue
                affected = set(new_years)
                if entry.get("replace_symbol", True):
                    affected |= existing_years
                for year in sorted(affected):
                    target = self.minute_symbol_year_path(frequency, symbol, year).resolve()
                    if target.exists():
                        backup_value = operation_by_target[target]["backup"]
                        if backup_value is None:
                            raise RuntimeError(f"分钟提交日志缺少备份路径: {target}")
                        backup = Path(backup_value)
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        target.replace(backup)
                for year, staged_path in staged.items():
                    target = self.minute_symbol_year_path(frequency, symbol, year).resolve()
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staged_path.replace(target)

            con.execute("BEGIN TRANSACTION")
            for entry in entries:
                frequency = str(entry["frequency"])
                symbol = str(entry["symbol"])
                years = sorted(entry["staged"])
                if entry.get("replace_symbol", True):
                    con.execute(
                        "DELETE FROM minute_partitions WHERE frequency = ? AND symbol = ?",
                        [frequency, symbol],
                    )
                    # A mirror replacement cannot prove that rows contributed by
                    # earlier ZIP archives survived, even when row counts match.
                    con.execute(
                        "DELETE FROM minute_archive_members WHERE frequency = ? AND symbol = ?",
                        [frequency, symbol],
                    )
                elif years:
                    placeholders = ",".join("?" for _ in years)
                    con.execute(
                        "DELETE FROM minute_partitions WHERE frequency = ? AND symbol = ? "
                        f"AND year IN ({placeholders})",
                        [frequency, symbol, *years],
                    )
                for year in years:
                    stat = entry["statistics"][year]
                    target = self.minute_symbol_year_path(frequency, symbol, year).resolve()
                    con.execute(
                        "INSERT INTO minute_partitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            frequency,
                            symbol,
                            year,
                            entry["asset_type"],
                            str(target),
                            stat["rows"],
                            stat["min_time"],
                            stat["max_time"],
                            entry["source_hash"],
                            datetime.now(),
                        ],
                    )
                    partition_stat = target.stat()
                    con.execute(
                        """
                        UPDATE minute_archive_members
                        SET partition_size = ?, partition_mtime_ns = ?
                        WHERE frequency = ? AND symbol = ? AND year = ?
                        """,
                        [
                            partition_stat.st_size,
                            partition_stat.st_mtime_ns,
                            frequency,
                            symbol,
                            year,
                        ],
                    )
            if source_record is not None:
                values = {key: value for key, value in source_record.items() if key != "statistics"}
                source_statistics = source_record.get("statistics", {})
                values["members"] = []
                for year in sorted(source_statistics):
                    stat = source_statistics[year]
                    partition = self.minute_symbol_year_path(
                        values["frequency"], values["symbol"], int(year)
                    )
                    partition_stat = partition.stat()
                    values["members"].append(
                        (
                            int(year),
                            int(stat["rows"]),
                            stat["min_time"],
                            stat["max_time"],
                            partition_stat.st_size,
                            partition_stat.st_mtime_ns,
                        )
                    )
                self._record_minute_source_con(con, values)
            if archive_record is not None:
                values = dict(archive_record)
                values["members"] = []
                for symbol, year, rows, min_time, max_time in archive_record.get("members", []):
                    partition = self.minute_symbol_year_path(
                        str(values["frequency"]), symbol, int(year)
                    )
                    partition_stat = partition.stat()
                    values["members"].append(
                        (
                            symbol,
                            int(year),
                            int(rows),
                            min_time,
                            max_time,
                            partition_stat.st_size,
                            partition_stat.st_mtime_ns,
                        )
                    )
                self._record_minute_import_con(con, values)
            con.execute(
                "UPDATE minute_commit_log SET status = 'committed', updated_at = ? "
                "WHERE commit_id = ?",
                [datetime.now(), commit_id],
            )
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            for operation in reversed(operations):
                target = Path(str(operation["target"]))
                backup_value = operation.get("backup")
                backup = Path(str(backup_value)) if backup_value else None
                if backup is None:
                    target.unlink(missing_ok=True)
                elif backup.exists():
                    target.unlink(missing_ok=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup.replace(target)
            with self.connect() as log_con:
                log_con.execute("DELETE FROM minute_commit_log WHERE commit_id = ?", [commit_id])
            shutil.rmtree(backup_root, ignore_errors=True)
            raise
        finally:
            con.close()
        shutil.rmtree(backup_root, ignore_errors=True)
        if not backup_root.exists():
            with self.connect() as log_con:
                log_con.execute("DELETE FROM minute_commit_log WHERE commit_id = ?", [commit_id])

    def merge_minute_partition(
        self,
        existing: Path,
        incoming: Path,
        output: Path,
    ) -> dict[str, Any]:
        """Merge an incremental minute partition without accepting conflicting bars."""
        output.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect() as con:
            con.read_parquet(
                [str(existing), str(incoming)],
                hive_partitioning=False,
                union_by_name=True,
            ).create_view("combined")
            conflicts = con.execute(
                """
                SELECT symbol, bar_time FROM combined
                GROUP BY symbol, bar_time
                HAVING count(DISTINCT struct_pack(
                    open := open, high := high, low := low, close := close,
                    volume := volume, amount := amount
                )) > 1
                LIMIT 3
                """
            ).fetchall()
            if conflicts:
                raise ValueError(f"增量分钟数据与已有分区冲突: {conflicts}")
            relation = con.sql(
                """
                SELECT * EXCLUDE (row_number) FROM (
                    SELECT *, row_number() OVER (
                        PARTITION BY symbol, bar_time ORDER BY source DESC
                    ) AS row_number
                    FROM combined
                ) WHERE row_number = 1
                ORDER BY symbol, bar_time
                """
            )
            relation.write_parquet(
                str(output),
                compression=self.config.minute.compression,
                row_group_size=self.config.minute.row_group_rows,
            )
            row = con.execute(
                "SELECT count(*), min(bar_time), max(bar_time) FROM read_parquet(?)",
                [str(output)],
            ).fetchone()
        return {"rows": int(row[0]), "min_time": row[1], "max_time": row[2]}

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
                SELECT * FROM read_parquet(?, hive_partitioning = false)
                WHERE bar_time >= ? AND bar_time <= ?
                ORDER BY symbol, bar_time
                """,
                [paths, start_at, end_at],
            ).df()

    def minute_source_unchanged(self, source_path: str, frequency: str, file_hash: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT status, file_hash, symbol FROM minute_sources
                WHERE source_path = ? AND frequency = ?
                """,
                [source_path, frequency],
            ).fetchone()
        return bool(
            row
            and row[0] in {"success", "empty"}
            and row[1] == file_hash
            and (
                row[0] == "empty"
                or self.minute_source_members_available(source_path, frequency, row[2])
            )
        )

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
                SELECT status, file_size, file_mtime_ns, symbol FROM minute_sources
                WHERE source_path = ? AND frequency = ?
                """,
                [source_path, frequency],
            ).fetchone()
        return bool(
            row
            and row[0] in {"success", "empty"}
            and row[1] == file_size
            and row[2] == file_mtime_ns
            and (
                row[0] == "empty"
                or self.minute_source_members_available(source_path, frequency, row[3])
            )
        )

    def minute_source_members_available(
        self, source_path: str, frequency: str, symbol: str | None
    ) -> bool:
        if not symbol:
            return False
        with self.connect() as con:
            members = con.execute(
                """
                    SELECT year, rows, min_time, max_time,
                           partition_size, partition_mtime_ns
                    FROM minute_source_members
                    WHERE source_path = ? AND frequency = ? AND symbol = ?
                    """,
                [source_path, frequency, symbol],
            ).fetchall()
            catalog_years = {
                int(row[0])
                for row in con.execute(
                    "SELECT year FROM minute_partitions WHERE frequency = ? AND symbol = ?",
                    [frequency, symbol],
                ).fetchall()
            }
        if not members or {int(row[0]) for row in members} - catalog_years:
            return False
        with duckdb.connect() as con:
            for year, expected_rows, min_time, max_time, file_size, file_mtime_ns in members:
                path = self.minute_symbol_year_path(frequency, symbol, int(year))
                if (
                    not path.exists()
                    or expected_rows is None
                    or min_time is None
                    or max_time is None
                    or file_size is None
                    or file_mtime_ns is None
                ):
                    return False
                stat = path.stat()
                if stat.st_size != int(file_size) or stat.st_mtime_ns != int(file_mtime_ns):
                    return False
                actual = con.execute(
                    """
                    SELECT count(*) FROM read_parquet(?, hive_partitioning = false)
                    WHERE symbol = ? AND bar_time BETWEEN ? AND ?
                    """,
                    [str(path), symbol, min_time, max_time],
                ).fetchone()[0]
                if int(actual) != int(expected_rows):
                    return False
        return True

    def record_minute_source(self, values: dict[str, Any]) -> None:
        with self.connect() as con:
            self._record_minute_source_con(con, values)

    @staticmethod
    def _record_minute_source_con(con: duckdb.DuckDBPyConnection, values: dict[str, Any]) -> None:
        con.execute(
            "DELETE FROM minute_sources WHERE source_path = ? AND frequency = ?",
            [values["source_path"], values["frequency"]],
        )
        con.execute(
            "INSERT INTO minute_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                values["source_path"],
                values["frequency"],
                values.get("symbol"),
                values.get("asset_type"),
                values.get("file_hash"),
                values.get("file_size", 0),
                values.get("file_mtime_ns", 0),
                datetime.now(),
                values.get("rows_input", 0),
                values.get("rows_written", 0),
                values.get("rows_filtered", 0),
                values.get("min_time"),
                values.get("max_time"),
                values["status"],
                values.get("error"),
            ],
        )
        con.execute(
            "DELETE FROM minute_source_members WHERE source_path = ? AND frequency = ?",
            [values["source_path"], values["frequency"]],
        )
        members = sorted(set(values.get("members", [])))
        if values["status"] == "success" and members:
            con.executemany(
                "INSERT INTO minute_source_members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        values["source_path"],
                        values["frequency"],
                        values["symbol"],
                        *member,
                    )
                    for member in members
                ],
            )

    def start_minute_import_run(
        self, run_id: str, source_root: str, frequency: str, files_total: int
    ) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO minute_import_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    source_root,
                    frequency,
                    datetime.now(),
                    None,
                    "running",
                    files_total,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    "{}",
                ],
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
                    datetime.now(),
                    values["status"],
                    values.get("files_success", 0),
                    values.get("files_skipped", 0),
                    values.get("files_empty", 0),
                    values.get("files_failed", 0),
                    values.get("rows_written", 0),
                    values.get("rows_filtered", 0),
                    json.dumps(values.get("details", {}), ensure_ascii=False),
                    run_id,
                ],
            )

    def minute_imported(self, file_hash: str, frequency: str, asset_type: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT status FROM minute_archive_imports
                WHERE file_hash = ? AND frequency = ? AND asset_type = ?
                """,
                [file_hash, frequency, asset_type],
            ).fetchone()
            members = con.execute(
                """
                SELECT symbol, year, rows, min_time, max_time,
                       partition_size, partition_mtime_ns
                FROM minute_archive_members
                WHERE file_hash = ? AND frequency = ? AND asset_type = ?
                """,
                [file_hash, frequency, asset_type],
            ).fetchall()
            catalog_members = {
                (str(symbol), int(year))
                for symbol, year in con.execute(
                    "SELECT symbol, year FROM minute_partitions WHERE frequency = ?",
                    [frequency],
                ).fetchall()
            }
        if not row or row[0] != "success" or not members:
            return False
        for (
            symbol,
            year,
            expected_rows,
            min_time,
            max_time,
            expected_size,
            expected_mtime_ns,
        ) in members:
            path = self.minute_symbol_year_path(frequency, symbol, int(year))
            if (
                (str(symbol), int(year)) not in catalog_members
                or not path.exists()
                or expected_rows is None
                or min_time is None
                or max_time is None
                or expected_size is None
                or expected_mtime_ns is None
            ):
                return False
            partition_stat = path.stat()
            if partition_stat.st_size != int(expected_size) or partition_stat.st_mtime_ns != int(
                expected_mtime_ns
            ):
                return False
        return True

    def record_minute_import(self, values: dict[str, Any]) -> None:
        with self.connect() as con:
            self._record_minute_import_con(con, values)

    @staticmethod
    def _record_minute_import_con(con: duckdb.DuckDBPyConnection, values: dict[str, Any]) -> None:
        con.execute(
            """
                INSERT OR REPLACE INTO minute_archive_imports
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
            [
                values["file_hash"],
                values["frequency"],
                values["asset_type"],
                values["file_name"],
                datetime.now(),
                values.get("rows", 0),
                values.get("min_time"),
                values.get("max_time"),
                values["status"],
                json.dumps(values.get("details", {}), ensure_ascii=False),
            ],
        )
        con.execute(
            """
                DELETE FROM minute_archive_members
                WHERE file_hash = ? AND frequency = ? AND asset_type = ?
                """,
            [values["file_hash"], values["frequency"], values["asset_type"]],
        )
        members = sorted(set(values.get("members", [])))
        if values["status"] == "success" and members:
            con.executemany(
                "INSERT INTO minute_archive_members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        values["file_hash"],
                        values["frequency"],
                        values["asset_type"],
                        *member,
                    )
                    for member in members
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
                [
                    datetime.now(),
                    values.get("dataset"),
                    values.get("provider"),
                    values.get("symbols"),
                    values.get("start"),
                    values.get("end"),
                    values.get("rows", 0),
                    values.get("status"),
                    json.dumps(values.get("warnings", []), ensure_ascii=False),
                    values.get("adjustment", "none"),
                ],
            )
