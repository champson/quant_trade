from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_trade.config import AppConfig
from quant_trade.data.quality import DataQualityError, validate_bars
from quant_trade.data.storage import DataStore


ALIASES = {
    "symbol": {"symbol", "ts_code", "code", "证券代码", "股票代码", "代码"},
    "bar_time": {"bar_time", "trade_time", "datetime", "time", "交易时间", "时间"},
    "trade_date": {"trade_date", "date", "交易日期", "日期"},
    "open": {"open", "开盘", "开盘价"},
    "high": {"high", "最高", "最高价"},
    "low": {"low", "最低", "最低价"},
    "close": {"close", "收盘", "收盘价"},
    "volume": {"volume", "vol", "成交量"},
    "amount": {"amount", "成交额", "成交金额"},
}


def apply_timestamp_convention(timestamps: pd.Series, frequency: str, convention: str) -> pd.Series:
    """Normalize imported timestamps to the canonical bar-end convention."""
    if convention == "bar_start":
        return timestamps + pd.Timedelta(minutes=int(frequency.removesuffix("min")))
    if convention in {"source", "bar_end"}:
        return timestamps
    raise ValueError(f"不支持的 timestamp_convention: {convention}")


@dataclass
class ArchiveProfile:
    encoding: str
    delimiter: str
    members: list[str]
    columns: dict[str, str]
    sample_rows: int
    timestamp_convention: str = "source"
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    file_hash: str
    file_name: str
    status: str
    rows: int = 0
    min_time: str | None = None
    max_time: str | None = None
    warnings: list[str] = field(default_factory=list)


class MinuteArchiveImporter:
    max_member_bytes = 5 * 1024**3
    max_ratio = 500

    def __init__(self, config: AppConfig, store: DataStore):
        self.config = config
        self.store = store
        self.pattern = re.compile(config.minute.filename_symbol_regex, re.IGNORECASE)

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _safe_members(self, archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        csvs = []
        for info in archive.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise DataQualityError(f"ZIP 包含不安全路径: {info.filename}")
            if info.file_size > self.max_member_bytes:
                raise DataQualityError(f"ZIP 成员过大: {info.filename}")
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > self.max_ratio:
                raise DataQualityError(f"ZIP 压缩比异常: {info.filename}")
            if not info.is_dir() and info.filename.lower().endswith(".csv"):
                csvs.append(info)
        if not csvs:
            raise DataQualityError("ZIP 中没有 CSV 文件")
        return csvs

    def _decode_sample(self, raw: bytes) -> tuple[str, str]:
        for encoding in self.config.minute.encoding_candidates:
            try:
                return raw.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        raise DataQualityError("无法识别 CSV 编码")

    @staticmethod
    def _mapping(headers: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        normalized = {str(h).strip().lower(): h for h in headers}
        for target, aliases in ALIASES.items():
            for alias in aliases:
                if alias.lower() in normalized:
                    result[normalized[alias.lower()]] = target
                    break
        return result

    def inspect(self, path: str | Path) -> ArchiveProfile:
        path = Path(path)
        with zipfile.ZipFile(path) as archive:
            members = sorted(self._safe_members(archive), key=lambda item: item.filename)
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise DataQualityError("ZIP 包含重复成员路径")
            with archive.open(members[0]) as handle:
                raw = handle.read(128 * 1024)
            sample, encoding = self._decode_sample(raw)
            try:
                dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",\t;|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
            header = next(csv.reader(io.StringIO(sample), delimiter=delimiter))
            mapping = self._mapping(header)
            required = {"open", "high", "low", "close", "volume"}
            if not required <= set(mapping.values()):
                raise DataQualityError("分钟 CSV 缺少 OHLCV 字段；检测到: " + ", ".join(header))
            if "bar_time" not in mapping.values() and "trade_date" not in mapping.values():
                raise DataQualityError("分钟 CSV 缺少交易时间")
            warnings = []
            if "symbol" not in mapping.values():
                warnings.append("CSV 无证券代码，将从文件名提取")
            return ArchiveProfile(
                encoding=encoding,
                delimiter=delimiter,
                members=names,
                columns=mapping,
                sample_rows=max(0, len(sample.splitlines()) - 1),
                timestamp_convention=self.config.minute.timestamp_convention,
                warnings=warnings,
            )

    def _symbol_from_name(self, name: str) -> str:
        match = self.pattern.search(Path(name).stem)
        if not match:
            raise DataQualityError(f"无法从文件名提取证券代码: {name}")
        code = match.group("symbol")
        exchange = match.groupdict().get("exchange")
        if not exchange:
            exchange = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code}.{exchange.upper()}"

    @staticmethod
    def _asset_type(symbol: str, requested: str) -> str:
        if requested != "auto":
            if requested not in {"stock", "etf", "index"}:
                raise ValueError("asset_type 只支持 auto/stock/etf/index")
            return requested
        code, _, exchange = symbol.partition(".")
        if code.startswith(("399", "93")) or (exchange == "SH" and code.startswith("000")):
            return "index"
        if code.startswith(("5", "15", "16")):
            return "etf"
        return "stock"

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

    def _normalize_chunk(
        self,
        chunk: pd.DataFrame,
        member: str,
        profile: ArchiveProfile,
        frequency: str,
        asset_type: str,
    ) -> pd.DataFrame:
        out = chunk.rename(columns=profile.columns).copy()
        if "symbol" not in out:
            out["symbol"] = self._symbol_from_name(member)
        out["symbol"] = out["symbol"].astype(str).str.strip().map(self._normalize_symbol)
        if "bar_time" in out:
            raw_time = out["bar_time"].astype(str).str.strip()
            if "trade_date" in out:
                only_clock = raw_time.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?").fillna(False)
                raw_time = raw_time.where(
                    ~only_clock,
                    out["trade_date"].astype(str).str.strip() + " " + raw_time,
                )
            out["bar_time"] = pd.to_datetime(raw_time, errors="coerce")
        elif "trade_date" in out:
            out["bar_time"] = pd.to_datetime(out["trade_date"], errors="coerce")
        out["bar_time"] = apply_timestamp_convention(
            out["bar_time"], frequency, profile.timestamp_convention
        )
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in out:
                out[col] = pd.NA
            out[col] = pd.to_numeric(out[col], errors="coerce")
        invalid = (
            out[["symbol", "bar_time", "open", "high", "low", "close", "volume"]].isna().any(axis=1)
        )
        invalid |= ~out["symbol"].str.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", na=False)
        if invalid.any():
            raise DataQualityError(f"{member} 存在 {int(invalid.sum())} 条无法解析的记录")
        out["trade_date"] = out["bar_time"].dt.date
        member_category = next(
            (
                part.lower()
                for part in Path(member).parts[:-1]
                if part.lower() in {"stock", "etf", "index"}
            ),
            asset_type,
        )
        requested_asset = member_category if asset_type == "auto" else asset_type
        out["asset_type"] = out["symbol"].map(
            lambda symbol: self._asset_type(symbol, requested_asset)
        )
        out["frequency"] = frequency
        out["is_auction"] = out["asset_type"].eq("stock") & out["bar_time"].dt.strftime(
            "%H:%M:%S"
        ).eq("09:30:00")
        out["source"] = "tushare_zip"
        return out[
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
        ]

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
    def _normalize_symbol(value: str) -> str:
        value = value.upper().replace(" ", "")
        if "." in value:
            code, exchange = value.split(".", 1)
            if code in {"SH", "SZ", "BJ"}:
                code, exchange = exchange, code
            # Tushare index archives sometimes use the vendor-only CSI suffix.
            # Persist the canonical exchange suffix used by the rest of the store.
            if exchange == "CSI":
                exchange = "SH"
            return f"{code}.{exchange}"
        return f"{value}.{'SH' if value.startswith(('5', '6', '9')) else 'SZ'}"

    def import_archive(
        self,
        path: str | Path,
        profile: ArchiveProfile | None = None,
        *,
        frequency: str = "1min",
        asset_type: str = "auto",
    ) -> ImportResult:
        if frequency not in {"1min", "5min", "15min", "30min", "60min"}:
            raise ValueError("不支持的分钟频率")
        if asset_type not in {"auto", "stock", "etf", "index"}:
            raise ValueError("asset_type 只支持 auto/stock/etf/index")
        path = Path(path)
        file_hash = self.hash_file(path)
        if self.store.minute_imported(file_hash, frequency, asset_type):
            destination = self.config.minute.archive / path.name
            if path.resolve() != destination.resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    destination = destination.with_name(f"{file_hash[:8]}_{destination.name}")
                shutil.move(str(path), destination)
            return ImportResult(file_hash, path.name, "skipped", warnings=["文件已导入"])
        try:
            profile = profile or self.inspect(path)
        except Exception as exc:
            result = ImportResult(file_hash, path.name, "failed", warnings=[str(exc)])
            self.store.record_minute_import(
                {
                    "file_hash": file_hash,
                    "frequency": frequency,
                    "asset_type": asset_type,
                    "file_name": path.name,
                    "status": "failed",
                    "details": {"warnings": result.warnings, "phase": "preflight"},
                    "members": [],
                }
            )
            destination = self.config.minute.quarantine / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != destination.resolve():
                if destination.exists():
                    destination = destination.with_name(f"{file_hash[:8]}_{destination.name}")
                try:
                    shutil.move(str(path), destination)
                except OSError as move_exc:
                    result.warnings.append(f"隔离失败: {move_exc}")
                    self.store.record_minute_import(
                        {
                            "file_hash": file_hash,
                            "frequency": frequency,
                            "asset_type": asset_type,
                            "file_name": path.name,
                            "status": "failed",
                            "details": {
                                "warnings": result.warnings,
                                "phase": "preflight",
                            },
                            "members": [],
                        }
                    )
            raise DataQualityError("; ".join(result.warnings)) from exc
        result = ImportResult(file_hash, path.name, "failed", warnings=list(profile.warnings))
        min_time = None
        max_time = None
        staging = self.store.root / ".staging" / "minute-archive" / uuid.uuid4().hex
        writers: dict[tuple[str, int], pq.ParquetWriter] = {}
        staged: dict[tuple[str, int], Path] = {}
        statistics: dict[tuple[str, int], dict] = {}
        symbol_assets: dict[str, str] = {}
        previous_times: dict[str, pd.Timestamp] = {}
        committed_members: list[tuple[str, int, int, pd.Timestamp, pd.Timestamp]] = []
        archive_recorded = False
        try:
            staging.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                members = {m.filename: m for m in self._safe_members(archive)}
                for name in profile.members:
                    with archive.open(members[name]) as raw:
                        text = io.TextIOWrapper(raw, encoding=profile.encoding, newline="")
                        for chunk in pd.read_csv(
                            text,
                            sep=profile.delimiter,
                            chunksize=200_000,
                            low_memory=False,
                            dtype=str,
                        ):
                            normalized = self._normalize_chunk(
                                chunk, name, profile, frequency, asset_type
                            )
                            if not normalized.empty:
                                result.warnings.extend(
                                    validate_bars(
                                        normalized.assign(
                                            trade_date=normalized["trade_date"].astype(str)
                                        ),
                                        minute=True,
                                    )
                                )
                                for symbol, symbol_frame in normalized.groupby(
                                    "symbol", sort=False
                                ):
                                    if not symbol_frame["bar_time"].is_monotonic_increasing:
                                        raise DataQualityError(
                                            f"{symbol} 的分钟K线未按时间升序排列"
                                        )
                                    actual_asset = str(symbol_frame["asset_type"].iloc[0])
                                    self._validate_time_grid(symbol_frame, frequency, actual_asset)
                                    if symbol in previous_times and (
                                        symbol_frame["bar_time"].iloc[0] <= previous_times[symbol]
                                    ):
                                        raise DataQualityError(
                                            f"{symbol} 在ZIP成员或分块边界存在重复/倒序K线"
                                        )
                                    previous_times[symbol] = symbol_frame["bar_time"].iloc[-1]
                                    symbol_assets[symbol] = actual_asset
                                    for year, part in symbol_frame.groupby(
                                        symbol_frame["bar_time"].dt.year
                                    ):
                                        key = (str(symbol), int(year))
                                        if key not in writers:
                                            staged_path = (
                                                staging
                                                / self.store.safe_symbol(str(symbol))
                                                / f"year={int(year)}.parquet"
                                            )
                                            staged_path.parent.mkdir(parents=True, exist_ok=True)
                                            writers[key] = pq.ParquetWriter(
                                                staged_path,
                                                self._arrow_schema(),
                                                compression=self.config.minute.compression,
                                                compression_level=self.config.minute.compression_level,
                                                use_dictionary=True,
                                                write_statistics=True,
                                            )
                                            staged[key] = staged_path
                                            statistics[key] = {
                                                "rows": 0,
                                                "min_time": part["bar_time"].min(),
                                                "max_time": part["bar_time"].max(),
                                            }
                                        writers[key].write_table(
                                            pa.Table.from_pandas(
                                                part,
                                                schema=self._arrow_schema(),
                                                preserve_index=False,
                                                safe=False,
                                            ),
                                            row_group_size=self.config.minute.row_group_rows,
                                        )
                                        statistics[key]["rows"] += len(part)
                                        statistics[key]["min_time"] = min(
                                            statistics[key]["min_time"],
                                            part["bar_time"].min(),
                                        )
                                        statistics[key]["max_time"] = max(
                                            statistics[key]["max_time"],
                                            part["bar_time"].max(),
                                        )
                                result.rows += len(normalized)
                                chunk_min = normalized["bar_time"].min()
                                chunk_max = normalized["bar_time"].max()
                                min_time = (
                                    chunk_min if min_time is None else min(min_time, chunk_min)
                                )
                                max_time = (
                                    chunk_max if max_time is None else max(max_time, chunk_max)
                                )
            if result.rows == 0:
                raise DataQualityError("ZIP 中没有可导入的分钟记录")
            for writer in writers.values():
                writer.close()
            writers.clear()
            archive_statistics = {key: dict(value) for key, value in statistics.items()}
            with self.store.minute_write_lock():
                for (symbol, year), incoming in list(staged.items()):
                    existing = self.store.minute_symbol_year_path(frequency, symbol, year)
                    if not existing.exists():
                        continue
                    merged = (
                        staging / "merged" / self.store.safe_symbol(symbol) / f"year={year}.parquet"
                    )
                    statistics[(symbol, year)] = self.store.merge_minute_partition(
                        existing, incoming, merged
                    )
                    staged[(symbol, year)] = merged
                entries = []
                for symbol in sorted(symbol_assets):
                    years = {
                        year: staged[(symbol, year)]
                        for item_symbol, year in staged
                        if item_symbol == symbol
                    }
                    stats = {
                        year: statistics[(symbol, year)]
                        for item_symbol, year in statistics
                        if item_symbol == symbol
                    }
                    entries.append(
                        {
                            "frequency": frequency,
                            "symbol": symbol,
                            "asset_type": symbol_assets[symbol],
                            "staged": years,
                            "statistics": stats,
                            "source_hash": file_hash,
                            # A ZIP may be an incremental delivery. Preserve years
                            # not represented by this archive.
                            "replace_symbol": False,
                        }
                    )
                committed_members = [
                    (
                        symbol,
                        year,
                        int(archive_statistics[(symbol, year)]["rows"]),
                        archive_statistics[(symbol, year)]["min_time"],
                        archive_statistics[(symbol, year)]["max_time"],
                    )
                    for symbol, year in sorted(staged)
                ]
                result.min_time = str(min_time)
                result.max_time = str(max_time)
                archive_record = {
                    "file_hash": file_hash,
                    "frequency": frequency,
                    "asset_type": asset_type,
                    "file_name": path.name,
                    "rows": result.rows,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "status": "success",
                    "details": {"warnings": result.warnings, "profile": asdict(profile)},
                    "members": committed_members,
                }
                self.store.commit_minute_batch(
                    entries,
                    assume_locked=True,
                    archive_record=archive_record,
                )
                archive_recorded = True
                result.status = "success"
            destination = self.config.minute.archive / path.name
        except Exception as exc:
            result.status = "failed"
            result.warnings.append(str(exc))
            destination = self.config.minute.quarantine / path.name
        finally:
            for writer in writers.values():
                writer.close()
            shutil.rmtree(staging, ignore_errors=True)
        if not archive_recorded:
            self.store.record_minute_import(
                {
                    "file_hash": file_hash,
                    "frequency": frequency,
                    "asset_type": asset_type,
                    "file_name": path.name,
                    "rows": result.rows,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "status": result.status,
                    "details": {"warnings": result.warnings, "profile": asdict(profile)},
                    "members": committed_members,
                }
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != destination.resolve():
            if destination.exists():
                destination = destination.with_name(f"{file_hash[:8]}_{destination.name}")
            shutil.move(str(path), destination)
        if result.status == "failed":
            raise DataQualityError("; ".join(result.warnings))
        return result

    def import_inbox(
        self, *, frequency: str = "1min", asset_type: str = "auto"
    ) -> list[ImportResult]:
        results = []
        for path in sorted(self.config.minute.inbox.glob("*.zip")):
            try:
                results.append(
                    self.import_archive(path, frequency=frequency, asset_type=asset_type)
                )
            except Exception as exc:
                results.append(ImportResult("", path.name, "failed", warnings=[str(exc)]))
        return results

    def write_profile(self, path: str | Path, profile: ArchiveProfile) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


def resample_minutes(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate minute bars by exchange session without crossing lunch."""
    if frequency not in {"5min", "15min", "30min", "60min"}:
        raise ValueError("只支持 5min/15min/30min/60min")
    value = int(frequency.removesuffix("min"))
    work = df.copy()
    work["bar_time"] = pd.to_datetime(work["bar_time"])
    if "is_auction" not in work:
        work["is_auction"] = False
    auctions = work[work["is_auction"]].copy()
    work = work[~work["is_auction"]].copy()
    work["session"] = work["bar_time"].dt.hour.lt(12).map({True: "am", False: "pm"})
    rows = []
    for (symbol, trade_date, session), part in work.groupby(["symbol", "trade_date", "session"]):
        part = part.sort_values("bar_time").copy()
        day = pd.Timestamp(trade_date).normalize()
        session_open = (
            day + pd.Timedelta(hours=9, minutes=30)
            if session == "am"
            else day + pd.Timedelta(hours=13)
        )
        elapsed = (part["bar_time"] - session_open).dt.total_seconds().div(60)
        # A source may contain a 13:00 reopening bar; fold it into the first
        # regular afternoon target bar rather than emitting a zero-length bin.
        bucket = ((elapsed.clip(lower=1) + value - 1) // value).astype(int) * value
        part["target_time"] = session_open + pd.to_timedelta(bucket, unit="m")
        agg = (
            part.groupby("target_time", sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                amount=("amount", "sum"),
            )
            .dropna(subset=["open", "close"])
            .rename_axis("bar_time")
        )
        agg["symbol"] = symbol
        agg["trade_date"] = trade_date
        agg["frequency"] = frequency
        agg["is_auction"] = False
        rows.append(agg.reset_index())
    if not auctions.empty:
        auctions["frequency"] = frequency
        rows.append(auctions)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["symbol", "bar_time"])
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
