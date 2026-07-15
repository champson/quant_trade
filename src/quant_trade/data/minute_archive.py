from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

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
            members = self._safe_members(archive)
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
                raise DataQualityError(
                    "分钟 CSV 缺少 OHLCV 字段；检测到: " + ", ".join(header)
                )
            if "bar_time" not in mapping.values() and "trade_date" not in mapping.values():
                raise DataQualityError("分钟 CSV 缺少交易时间")
            warnings = []
            if "symbol" not in mapping.values():
                warnings.append("CSV 无证券代码，将从文件名提取")
            return ArchiveProfile(
                encoding=encoding, delimiter=delimiter,
                members=[m.filename for m in members], columns=mapping,
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

    def _normalize_chunk(self, chunk: pd.DataFrame, member: str, profile: ArchiveProfile, file_hash: str) -> pd.DataFrame:
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
        if "trade_date" in out and out["bar_time"].isna().any():
            out["bar_time"] = pd.to_datetime(out["trade_date"].astype(str), errors="coerce")
        out["trade_date"] = out["bar_time"].dt.strftime("%Y-%m-%d")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in out:
                out[col] = pd.NA
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out["source"] = "tushare_zip"
        out["source_file_hash"] = file_hash
        return out[["symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount", "source", "source_file_hash"]].dropna(
            subset=["symbol", "bar_time", "open", "high", "low", "close", "volume"]
        )

    @staticmethod
    def _normalize_symbol(value: str) -> str:
        value = value.upper().replace(" ", "")
        if "." in value:
            code, exchange = value.split(".", 1)
            if code in {"SH", "SZ", "BJ"}:
                code, exchange = exchange, code
            return f"{code}.{exchange}"
        return f"{value}.{'SH' if value.startswith(('5', '6', '9')) else 'SZ'}"

    def import_archive(self, path: str | Path, profile: ArchiveProfile | None = None) -> ImportResult:
        path = Path(path)
        file_hash = self.hash_file(path)
        if self.store.minute_imported(file_hash):
            return ImportResult(file_hash, path.name, "skipped", warnings=["文件已导入"])
        profile = profile or self.inspect(path)
        result = ImportResult(file_hash, path.name, "failed", warnings=list(profile.warnings))
        min_time = None
        max_time = None
        try:
            with zipfile.ZipFile(path) as archive:
                members = {m.filename: m for m in self._safe_members(archive)}
                for name in profile.members:
                    with archive.open(members[name]) as raw:
                        text = io.TextIOWrapper(raw, encoding=profile.encoding, newline="")
                        for chunk in pd.read_csv(
                            text, sep=profile.delimiter, chunksize=200_000,
                            low_memory=False, dtype=str,
                        ):
                            normalized = self._normalize_chunk(chunk, name, profile, file_hash)
                            if not normalized.empty:
                                result.warnings.extend(validate_bars(normalized, minute=True))
                                self.store.write_minute(normalized)
                                result.rows += len(normalized)
                                chunk_min = normalized["bar_time"].min()
                                chunk_max = normalized["bar_time"].max()
                                min_time = chunk_min if min_time is None else min(min_time, chunk_min)
                                max_time = chunk_max if max_time is None else max(max_time, chunk_max)
            if result.rows == 0:
                raise DataQualityError("ZIP 中没有可导入的分钟记录")
            result.status = "success"
            result.min_time = str(min_time)
            result.max_time = str(max_time)
            destination = self.config.minute.archive / path.name
        except Exception as exc:
            result.warnings.append(str(exc))
            destination = self.config.minute.quarantine / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != destination.resolve():
            if destination.exists():
                destination = destination.with_name(f"{file_hash[:8]}_{destination.name}")
            shutil.move(str(path), destination)
        self.store.record_minute_import({
            "file_hash": file_hash, "file_name": path.name, "rows": result.rows,
            "min_time": result.min_time, "max_time": result.max_time,
            "status": result.status, "details": {"warnings": result.warnings, "profile": asdict(profile)},
        })
        if result.status == "failed":
            raise DataQualityError("; ".join(result.warnings))
        return result

    def import_inbox(self) -> list[ImportResult]:
        results = []
        for path in sorted(self.config.minute.inbox.glob("*.zip")):
            try:
                results.append(self.import_archive(path))
            except DataQualityError as exc:
                results.append(ImportResult("", path.name, "failed", warnings=[str(exc)]))
        return results

    def write_profile(self, path: str | Path, profile: ArchiveProfile) -> Path:
        target = Path(path)
        target.write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def resample_minutes(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate 1-minute bars without crossing the A-share lunch break."""
    if frequency not in {"5min", "15min", "30min", "60min"}:
        raise ValueError("只支持 5min/15min/30min/60min")
    value = int(frequency.removesuffix("min"))
    work = df.copy()
    work["bar_time"] = pd.to_datetime(work["bar_time"])
    work["session"] = work["bar_time"].dt.hour.lt(12).map({True: "am", False: "pm"})
    rows = []
    for (symbol, trade_date, session), part in work.groupby(["symbol", "trade_date", "session"]):
        part = part.set_index("bar_time").sort_index()
        agg = part.resample(f"{value}min", origin="start_day", label="right", closed="right").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), amount=("amount", "sum"),
        ).dropna(subset=["open", "close"])
        agg["symbol"] = symbol
        agg["trade_date"] = trade_date
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
