from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_trade.config import AppConfig
from quant_trade.data.minute_archive import MinuteArchiveImporter
from quant_trade.data.quality import DataQualityError, validate_bars
from quant_trade.data.storage import DataStore


ASSET_TYPES = {"stock": "stock", "etf": "etf", "index": "index"}
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True)
class DirectorySource:
    path: Path
    relative_path: str
    symbol: str
    asset_type: str
    expected_rows: int | None = None


@dataclass
class DirectoryProfile:
    root: str
    files: int
    files_by_type: dict[str, int]
    expected_rows: int
    manifest: str | None
    bad_headers: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    unknown_asset_types: list[str] = field(default_factory=list)
    duplicate_symbols: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not (
            self.bad_headers
            or self.missing_files
            or self.unknown_asset_types
            or self.duplicate_symbols
        )


@dataclass
class FileImportResult:
    relative_path: str
    symbol: str
    asset_type: str
    status: str
    rows_input: int = 0
    rows_written: int = 0
    rows_filtered: int = 0
    min_time: str | None = None
    max_time: str | None = None
    error: str | None = None


@dataclass
class DirectoryImportResult:
    run_id: str
    source_root: str
    frequency: str
    files_total: int
    files_success: int = 0
    files_skipped: int = 0
    files_empty: int = 0
    files_failed: int = 0
    rows_written: int = 0
    rows_filtered: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "success" if self.files_failed == 0 else "partial_failed"


class MinuteDirectoryImporter:
    """Read a directory of per-symbol CSVs without mutating the source tree."""

    def __init__(self, config: AppConfig, store: DataStore):
        self.config = config
        self.store = store

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _normalize_symbol(value: str) -> str:
        return MinuteArchiveImporter._normalize_symbol(str(value).strip())

    def _filename_symbol(self, path: Path) -> str:
        match = re.search(self.config.minute.filename_symbol_regex, path.stem, re.IGNORECASE)
        if not match:
            return self._normalize_symbol(path.stem)
        code = match.group("symbol")
        exchange = match.groupdict().get("exchange")
        return self._normalize_symbol(f"{code}.{exchange}" if exchange else code)

    def _manifest_sources(self, root: Path) -> tuple[list[DirectorySource], Path | None]:
        manifest = root / "manifest.csv"
        if not manifest.exists():
            return [], None
        frame = pd.read_csv(manifest, dtype=str, encoding="utf-8-sig").fillna("")
        required = {"category", "ts_code", "relative_file"}
        if not required <= set(frame.columns):
            raise DataQualityError("manifest.csv 缺少 category/ts_code/relative_file")
        sources = []
        for row in frame.to_dict("records"):
            category = row["category"].strip().lower()
            rows = int(row["rows"]) if row.get("rows", "").strip().isdigit() else None
            sources.append(
                DirectorySource(
                    path=root / row["relative_file"],
                    relative_path=row["relative_file"],
                    symbol=self._normalize_symbol(row["ts_code"]),
                    asset_type=ASSET_TYPES.get(category, category),
                    expected_rows=rows,
                )
            )
        return sources, manifest

    def discover(self, root: str | Path) -> tuple[list[DirectorySource], Path | None]:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise DataQualityError(f"分钟数据目录不存在: {root}")
        sources, manifest = self._manifest_sources(root)
        if sources:
            return sources, manifest
        discovered = []
        for path in sorted(root.rglob("*.csv")):
            if path.name in {"manifest.csv"}:
                continue
            relative = path.relative_to(root)
            category = relative.parts[0].lower() if len(relative.parts) > 1 else ""
            discovered.append(
                DirectorySource(
                    path=path,
                    relative_path=str(relative),
                    symbol=self._filename_symbol(path),
                    asset_type=ASSET_TYPES.get(category, "unknown"),
                )
            )
        if not discovered:
            raise DataQualityError("目录中没有证券CSV")
        return discovered, None

    def _header(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        for encoding in self.config.minute.encoding_candidates:
            try:
                with path.open("r", encoding=encoding, newline="") as handle:
                    return next(csv.reader(handle))
            except UnicodeDecodeError:
                continue
            except StopIteration:
                return []
        return []

    def inspect_directory(self, root: str | Path) -> DirectoryProfile:
        root_path = Path(root).expanduser().resolve()
        sources, manifest = self.discover(root_path)
        bad_headers, missing, unknown = [], [], []
        symbol_counts: dict[str, int] = {}
        by_type: dict[str, int] = {}
        expected_rows = 0
        for source in sources:
            symbol_counts[source.symbol] = symbol_counts.get(source.symbol, 0) + 1
            by_type[source.asset_type] = by_type.get(source.asset_type, 0) + 1
            expected_rows += source.expected_rows or 0
            if not source.path.exists():
                missing.append(source.relative_path)
                continue
            header = self._header(source.path)
            mapping = MinuteArchiveImporter._mapping(header)
            required = {"symbol", "bar_time", "open", "high", "low", "close", "volume"}
            if not required <= set(mapping.values()):
                bad_headers.append(source.relative_path)
            if source.asset_type not in ASSET_TYPES.values():
                unknown.append(source.relative_path)
        return DirectoryProfile(
            root=str(root_path),
            files=len(sources),
            files_by_type=by_type,
            expected_rows=expected_rows,
            manifest=str(manifest) if manifest else None,
            bad_headers=bad_headers,
            missing_files=missing,
            unknown_asset_types=unknown,
            duplicate_symbols=sorted(
                symbol for symbol, count in symbol_counts.items() if count > 1
            ),
        )

    def _encoding_and_mapping(self, path: Path) -> tuple[str, dict[str, str]]:
        header = self._header(path)
        if not header:
            # A zero-byte file is treated as empty, while header-only files are read normally.
            return self.config.minute.encoding_candidates[0], {}
        mapping = MinuteArchiveImporter._mapping(header)
        required = {"symbol", "bar_time", "open", "high", "low", "close", "volume"}
        if not required <= set(mapping.values()):
            raise DataQualityError(f"CSV字段不完整: {path.name}: {header}")
        for encoding in self.config.minute.encoding_candidates:
            try:
                with path.open("r", encoding=encoding, errors="strict") as handle:
                    handle.read(128 * 1024)
                return encoding, mapping
            except UnicodeDecodeError:
                continue
        raise DataQualityError(f"无法识别CSV编码: {path.name}")

    @staticmethod
    def _placeholder_mask(frame: pd.DataFrame, asset_type: str) -> pd.Series:
        if asset_type != "etf":
            return pd.Series(False, index=frame.index)
        clock = frame["bar_time"].dt.strftime("%H:%M:%S").eq("09:31:00")
        one_price = frame[["open", "high", "low", "close"]].eq(1.0).all(axis=1)
        zero_trade = frame["volume"].eq(0) & frame["amount"].fillna(0).eq(0)
        return clock & one_price & zero_trade

    @staticmethod
    def _validate_time_grid(frame: pd.DataFrame, frequency: str, asset_type: str) -> None:
        minutes = int(frequency.removesuffix("min"))
        timestamps = frame["bar_time"]
        clock_minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
        auction = asset_type == "stock" and clock_minutes.eq(9 * 60 + 30)
        am_elapsed = clock_minutes - (9 * 60 + 30)
        pm_elapsed = clock_minutes - (13 * 60)
        regular = ((am_elapsed > 0) & (am_elapsed <= 120) & am_elapsed.mod(minutes).eq(0)) | (
            (pm_elapsed >= 0) & (pm_elapsed <= 120) & pm_elapsed.mod(minutes).eq(0)
        )
        invalid = ~(regular | auction)
        if invalid.any():
            examples = frame.loc[invalid, "bar_time"].head(3).astype(str).tolist()
            raise DataQualityError(f"存在不符合{frequency}交易时段的时间: {examples}")

    @staticmethod
    def _deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        duplicate = frame.duplicated("bar_time", keep=False)
        if not duplicate.any():
            return frame, 0
        values = ["open", "high", "low", "close", "volume", "amount"]
        conflicts = (
            frame.loc[duplicate].groupby("bar_time")[values].nunique(dropna=False).max(axis=1) > 1
        )
        if conflicts.any():
            examples = conflicts[conflicts].index[:3].astype(str).tolist()
            raise DataQualityError(f"同时间K线数值冲突: {examples}")
        before = len(frame)
        return frame.drop_duplicates("bar_time", keep="last"), before - frame["bar_time"].nunique()

    def _normalize_chunk(
        self,
        chunk: pd.DataFrame,
        source: DirectorySource,
        mapping: dict[str, str],
        frequency: str,
    ) -> tuple[pd.DataFrame, int]:
        frame = chunk.rename(columns=mapping).copy()
        frame["symbol"] = frame["symbol"].astype(str).map(self._normalize_symbol)
        mismatched = frame["symbol"].ne(source.symbol)
        if mismatched.any():
            raise DataQualityError(
                f"文件名/manifest与CSV代码不一致: {source.symbol} != "
                f"{frame.loc[mismatched, 'symbol'].iloc[0]}"
            )
        frame["bar_time"] = pd.to_datetime(frame["bar_time"], errors="coerce")
        for column in NUMERIC_COLUMNS:
            if column not in frame:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        required = ["bar_time", "open", "high", "low", "close", "volume"]
        invalid = frame[required].isna().any(axis=1)
        if invalid.any():
            raise DataQualityError(f"存在 {int(invalid.sum())} 条无法解析的记录")
        placeholder = self._placeholder_mask(frame, source.asset_type)
        filtered = int(placeholder.sum())
        frame = frame.loc[~placeholder].copy()
        frame, duplicates = self._deduplicate(frame)
        filtered += duplicates
        if frame.empty:
            return frame, filtered
        if not frame["bar_time"].is_monotonic_increasing:
            raise DataQualityError("CSV未按trade_time升序排列")
        self._validate_time_grid(frame, frequency, source.asset_type)
        frame["trade_date"] = frame["bar_time"].dt.date
        frame["asset_type"] = source.asset_type
        frame["frequency"] = frequency
        frame["is_auction"] = (source.asset_type == "stock") & frame["bar_time"].dt.strftime(
            "%H:%M:%S"
        ).eq("09:30:00")
        frame["source"] = "tushare_directory"
        validate_bars(frame.assign(trade_date=frame["trade_date"].astype(str)), minute=True)
        return frame[
            [
                "symbol",
                "asset_type",
                "frequency",
                "trade_date",
                "bar_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "is_auction",
                "source",
            ]
        ], filtered

    @staticmethod
    def _arrow_schema() -> pa.Schema:
        return pa.schema(
            [
                ("symbol", pa.string()),
                ("asset_type", pa.string()),
                ("frequency", pa.string()),
                ("trade_date", pa.date32()),
                ("bar_time", pa.timestamp("ns")),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("volume", pa.float64()),
                ("amount", pa.float64()),
                ("is_auction", pa.bool_()),
                ("source", pa.string()),
            ]
        )

    def _import_file(
        self,
        source: DirectorySource,
        frequency: str,
        run_id: str,
        resume: bool,
    ) -> FileImportResult:
        stat = source.path.stat()
        source_path = str(source.path.resolve())
        if resume and self.store.minute_source_stat_unchanged(
            source_path, frequency, stat.st_size, stat.st_mtime_ns
        ):
            return FileImportResult(
                source.relative_path, source.symbol, source.asset_type, "skipped"
            )
        file_hash = self._hash_file(source.path)
        if resume and self.store.minute_source_unchanged(source_path, frequency, file_hash):
            return FileImportResult(
                source.relative_path, source.symbol, source.asset_type, "skipped"
            )
        result = FileImportResult(source.relative_path, source.symbol, source.asset_type, "failed")
        staging = (
            self.store.root / ".staging" / "minute" / run_id / self.store.safe_symbol(source.symbol)
        )
        staging.mkdir(parents=True, exist_ok=True)
        writers: dict[int, pq.ParquetWriter] = {}
        staged: dict[int, Path] = {}
        statistics: dict[int, dict[str, Any]] = {}
        previous_time: pd.Timestamp | None = None
        previous_values: tuple[Any, ...] | None = None
        try:
            encoding, mapping = self._encoding_and_mapping(source.path)
            if not mapping:
                result.status = "empty"
            else:
                chunks = pd.read_csv(
                    source.path,
                    encoding=encoding,
                    dtype=str,
                    chunksize=self.config.minute.chunk_rows,
                    low_memory=False,
                )
                for chunk in chunks:
                    result.rows_input += len(chunk)
                    frame, filtered = self._normalize_chunk(chunk, source, mapping, frequency)
                    result.rows_filtered += filtered
                    if frame.empty:
                        continue
                    if previous_time is not None:
                        first_time = frame["bar_time"].iloc[0]
                        if first_time < previous_time:
                            raise DataQualityError("分块边界时间倒序")
                        if first_time == previous_time:
                            columns = ["open", "high", "low", "close", "volume", "amount"]
                            current_values = tuple(frame.iloc[0][columns])
                            if current_values != previous_values:
                                raise DataQualityError(f"分块边界K线冲突: {first_time}")
                            frame = frame.iloc[1:].copy()
                            result.rows_filtered += 1
                            if frame.empty:
                                continue
                    previous_time = frame["bar_time"].iloc[-1]
                    previous_values = tuple(
                        frame.iloc[-1][["open", "high", "low", "close", "volume", "amount"]]
                    )
                    result.rows_written += len(frame)
                    current_min, current_max = frame["bar_time"].min(), frame["bar_time"].max()
                    result.min_time = (
                        str(current_min)
                        if result.min_time is None
                        else str(min(pd.Timestamp(result.min_time), current_min))
                    )
                    result.max_time = (
                        str(current_max)
                        if result.max_time is None
                        else str(max(pd.Timestamp(result.max_time), current_max))
                    )
                    for year, part in frame.groupby(frame["bar_time"].dt.year):
                        year = int(year)
                        path = staging / f"year={year}.parquet"
                        if year not in writers:
                            writers[year] = pq.ParquetWriter(
                                path,
                                self._arrow_schema(),
                                compression=self.config.minute.compression,
                                compression_level=self.config.minute.compression_level,
                                use_dictionary=True,
                                write_statistics=True,
                            )
                            staged[year] = path
                            statistics[year] = {
                                "rows": 0,
                                "min_time": part["bar_time"].min(),
                                "max_time": part["bar_time"].max(),
                            }
                        table = pa.Table.from_pandas(
                            part, schema=self._arrow_schema(), preserve_index=False, safe=False
                        )
                        writers[year].write_table(
                            table, row_group_size=self.config.minute.row_group_rows
                        )
                        statistics[year]["rows"] += len(part)
                        statistics[year]["min_time"] = min(
                            statistics[year]["min_time"], part["bar_time"].min()
                        )
                        statistics[year]["max_time"] = max(
                            statistics[year]["max_time"], part["bar_time"].max()
                        )
                if source.expected_rows is not None and result.rows_input != source.expected_rows:
                    raise DataQualityError(
                        f"实际行数 {result.rows_input} 与manifest {source.expected_rows} 不一致"
                    )
                for writer in writers.values():
                    writer.close()
                writers.clear()
                if staged:
                    self.store.commit_minute_symbol(
                        frequency=frequency,
                        symbol=source.symbol,
                        asset_type=source.asset_type,
                        staged=staged,
                        statistics=statistics,
                        source_hash=file_hash,
                    )
                    result.status = "success"
                else:
                    # An empty/truncated source is not proof that previously
                    # imported history should be deleted.
                    result.status = "empty"
            self.store.record_minute_source(
                {
                    "source_path": source_path,
                    "frequency": frequency,
                    "symbol": source.symbol,
                    "asset_type": source.asset_type,
                    "file_hash": file_hash,
                    "file_size": stat.st_size,
                    "file_mtime_ns": stat.st_mtime_ns,
                    "rows_input": result.rows_input,
                    "rows_written": result.rows_written,
                    "rows_filtered": result.rows_filtered,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "status": result.status,
                }
            )
            return result
        except Exception as exc:
            result.error = str(exc)
            for writer in writers.values():
                writer.close()
            self.store.record_minute_source(
                {
                    "source_path": source_path,
                    "frequency": frequency,
                    "symbol": source.symbol,
                    "asset_type": source.asset_type,
                    "file_hash": file_hash,
                    "file_size": stat.st_size,
                    "file_mtime_ns": stat.st_mtime_ns,
                    "rows_input": result.rows_input,
                    "rows_written": 0,
                    "rows_filtered": result.rows_filtered,
                    "status": "failed",
                    "error": result.error,
                }
            )
            return result
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def import_directory(
        self,
        root: str | Path,
        frequency: str = "5min",
        *,
        resume: bool = True,
        progress: Callable[[int, int, FileImportResult], None] | None = None,
    ) -> DirectoryImportResult:
        if frequency not in {"1min", "5min", "15min", "30min", "60min"}:
            raise ValueError("不支持的分钟频率")
        profile = self.inspect_directory(root)
        if not profile.valid:
            raise DataQualityError(json.dumps(asdict(profile), ensure_ascii=False))
        sources, _ = self.discover(root)
        run_id = f"minute-{uuid.uuid4().hex[:12]}"
        result = DirectoryImportResult(run_id, profile.root, frequency, len(sources))
        self.store.start_minute_import_run(run_id, profile.root, frequency, len(sources))
        for index, source in enumerate(sources, 1):
            item = self._import_file(source, frequency, run_id, resume)
            if item.status == "success":
                result.files_success += 1
            elif item.status == "skipped":
                result.files_skipped += 1
            elif item.status == "empty":
                result.files_empty += 1
            else:
                result.files_failed += 1
                result.failures.append({"file": item.relative_path, "error": item.error or ""})
            result.rows_written += item.rows_written
            result.rows_filtered += item.rows_filtered
            if progress:
                progress(index, len(sources), item)
        self.store.finish_minute_import_run(
            run_id,
            {
                "status": result.status,
                "files_success": result.files_success,
                "files_skipped": result.files_skipped,
                "files_empty": result.files_empty,
                "files_failed": result.files_failed,
                "rows_written": result.rows_written,
                "rows_filtered": result.rows_filtered,
                "details": {"failures": result.failures[:100]},
            },
        )
        return result
