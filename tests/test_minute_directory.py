from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from quant_trade.data.minute_directory import MinuteDirectoryImporter
from quant_trade.data.quality import DataQualityError
from quant_trade.data.storage import DataStore


HEADER = "ts_code,trade_time,open,close,high,low,vol,amount\n"


def test_minute_member_catalog_migrates_from_legacy_schema(app_config):
    app_config.ensure_directories()
    with duckdb.connect(str(app_config.paths.database)) as con:
        con.execute(
            """
            CREATE TABLE minute_source_members (
                source_path VARCHAR, frequency VARCHAR, symbol VARCHAR, year INTEGER,
                PRIMARY KEY (source_path, frequency, symbol, year)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE minute_archive_members (
                file_hash VARCHAR, frequency VARCHAR, asset_type VARCHAR,
                symbol VARCHAR, year INTEGER,
                PRIMARY KEY (file_hash, frequency, asset_type, symbol, year)
            )
            """
        )
    store = DataStore(app_config)
    with store.connect() as con:
        source_columns = {
            row[1] for row in con.execute("PRAGMA table_info('minute_source_members')").fetchall()
        }
        archive_columns = {
            row[1] for row in con.execute("PRAGMA table_info('minute_archive_members')").fetchall()
        }
    assert {
        "rows",
        "min_time",
        "max_time",
        "partition_size",
        "partition_mtime_ns",
    } <= source_columns
    assert {
        "rows",
        "min_time",
        "max_time",
        "partition_size",
        "partition_mtime_ns",
    } <= archive_columns


def _write_dataset(root: Path) -> None:
    (root / "stock").mkdir(parents=True)
    (root / "etf").mkdir()
    (root / "index").mkdir()
    (root / "stock" / "000001.SZ.csv").write_text(
        HEADER
        + "000001.SZ,2024-01-02 09:30:00,10,10,10,10,0,0\n"
        + "000001.SZ,2024-01-02 09:35:00,10,10.1,10.2,9.9,100,1000\n"
        + "000001.SZ,2025-01-02 09:35:00,11,11.1,11.2,10.9,200,2200\n",
        encoding="utf-8-sig",
    )
    (root / "etf" / "510300.SH.csv").write_text(
        HEADER
        + "510300.SH,2025-12-31 09:31:00,1,1,1,1,0,0\n"
        + "510300.SH,2026-01-05 09:35:00,4,4.1,4.2,3.9,10000,41000\n"
        + "510300.SH,2026-01-05 13:00:00,4.1,4.2,4.3,4.0,20000,84000\n",
        encoding="utf-8-sig",
    )
    (root / "index" / "000001.SH.csv").write_text(HEADER, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "category": "stock",
                "ts_code": "000001.SZ",
                "relative_file": "stock/000001.SZ.csv",
                "rows": 3,
            },
            {
                "category": "etf",
                "ts_code": "510300.SH",
                "relative_file": "etf/510300.SH.csv",
                "rows": 3,
            },
            {
                "category": "index",
                "ts_code": "000001.SH",
                "relative_file": "index/000001.SH.csv",
                "rows": 0,
            },
        ]
    ).to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")


def test_directory_import_is_flat_audited_and_resumable(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    importer = MinuteDirectoryImporter(app_config, store)
    profile = importer.inspect_directory(source)
    assert profile.valid
    assert profile.files_by_type == {"stock": 1, "etf": 1, "index": 1}
    assert profile.expected_rows == 6

    result = importer.import_directory(source, "5min")
    assert result.status == "success"
    assert result.files_success == 2
    assert result.files_empty == 1
    assert result.rows_written == 5
    assert result.rows_filtered == 1

    stock_path = store.minute_symbol_year_path("5min", "000001.SZ", 2024)
    etf_path = store.minute_symbol_year_path("5min", "510300.SH", 2026)
    assert stock_path.exists() and etf_path.exists()
    assert "stock" not in stock_path.parts and "etf" not in etf_path.parts
    stock = pd.read_parquet(stock_path)
    etf = pd.read_parquet(etf_path)
    assert stock["asset_type"].unique().tolist() == ["stock"]
    assert bool(stock.loc[stock["bar_time"].dt.strftime("%H:%M") == "09:30", "is_auction"].iloc[0])
    assert etf["bar_time"].dt.strftime("%H:%M").tolist() == ["09:35", "13:00"]

    queried = store.read_minute(["000001.SZ"], "2024-01-02 09:00", "2024-01-02 10:00", "5min")
    assert len(queried) == 2
    assert queried["asset_type"].unique().tolist() == ["stock"]

    again = importer.import_directory(source, "5min")
    assert again.files_skipped == 3
    assert again.rows_written == 0


def test_directory_import_rejects_conflicting_duplicate(app_config, tmp_path):
    source = tmp_path / "source"
    (source / "stock").mkdir(parents=True)
    path = source / "stock" / "000001.SZ.csv"
    path.write_text(
        HEADER
        + "000001.SZ,2024-01-02 09:35:00,10,10,10,10,100,1000\n"
        + "000001.SZ,2024-01-02 09:35:00,10,11,11,10,100,1000\n",
        encoding="utf-8-sig",
    )
    result = MinuteDirectoryImporter(app_config, DataStore(app_config)).import_directory(
        source, "5min"
    )
    assert result.status == "partial_failed"
    assert result.files_failed == 1
    assert "冲突" in result.failures[0]["error"]


def test_empty_source_does_not_delete_existing_partitions(app_config, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "000001.SZ.csv"
    path.write_text(
        HEADER + "000001.SZ,2024-01-02 09:35:00,10,10,10,10,100,1000\n",
        encoding="utf-8-sig",
    )
    manifest = source / "manifest.csv"
    pd.DataFrame(
        [
            {
                "category": "stock",
                "ts_code": "000001.SZ",
                "relative_file": path.name,
                "rows": 1,
            }
        ]
    ).to_csv(manifest, index=False, encoding="utf-8-sig")
    store = DataStore(app_config)
    importer = MinuteDirectoryImporter(app_config, store)
    first = importer.import_directory(source, "5min")
    partition = store.minute_symbol_year_path("5min", "000001.SZ", 2024)
    assert first.files_success == 1
    assert partition.exists()

    path.write_text(HEADER, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "category": "stock",
                "ts_code": "000001.SZ",
                "relative_file": path.name,
                "rows": 0,
            }
        ]
    ).to_csv(manifest, index=False, encoding="utf-8-sig")
    second = importer.import_directory(source, "5min", resume=False)
    assert second.files_empty == 1
    assert partition.exists()
    assert len(pd.read_parquet(partition)) == 1


def test_directory_rejects_multiple_files_for_same_symbol(app_config, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for year in (2024, 2025):
        (source / f"000001.SZ-{year}.csv").write_text(
            HEADER + f"000001.SZ,{year}-01-02 09:35:00,10,10,10,10,100,1000\n",
            encoding="utf-8-sig",
        )
    pd.DataFrame(
        [
            {
                "category": "stock",
                "ts_code": "000001.SZ",
                "relative_file": f"000001.SZ-{year}.csv",
                "rows": 1,
            }
            for year in (2024, 2025)
        ]
    ).to_csv(source / "manifest.csv", index=False, encoding="utf-8-sig")
    importer = MinuteDirectoryImporter(app_config, DataStore(app_config))
    profile = importer.inspect_directory(source)
    assert profile.duplicate_symbols == ["000001.SZ"]
    with pytest.raises(DataQualityError, match="duplicate_symbols"):
        importer.import_directory(source, "5min")


def test_resume_repairs_deleted_physical_partition(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    importer = MinuteDirectoryImporter(app_config, store)
    assert importer.import_directory(source, "5min").files_success == 2
    partition = store.minute_symbol_year_path("5min", "000001.SZ", 2024)
    partition.unlink()

    repaired = importer.import_directory(source, "5min")
    assert repaired.files_success == 1
    assert repaired.files_skipped == 2
    assert partition.exists()


def test_resume_repairs_partition_rewritten_with_same_rows_and_times(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    importer = MinuteDirectoryImporter(app_config, store)
    assert importer.import_directory(source, "5min").files_success == 2
    partition = store.minute_symbol_year_path("5min", "000001.SZ", 2024)
    corrupted = pd.read_parquet(partition).assign(close=999.0)
    corrupted.to_parquet(partition, index=False)

    repaired = importer.import_directory(source, "5min")
    assert repaired.files_success == 1
    assert repaired.files_skipped == 2
    assert 999.0 not in set(pd.read_parquet(partition)["close"])


def test_source_metadata_failure_rolls_back_partition_and_catalog(
    app_config, tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source = source / "stock"
    source.mkdir(parents=True)
    path = source / "000001.SZ.csv"
    path.write_text(
        HEADER + "000001.SZ,2024-01-02 09:35:00,10,10,10,10,100,1000\n",
        encoding="utf-8-sig",
    )
    store = DataStore(app_config)
    original = DataStore._record_minute_source_con

    def fail_success_record(con, values):
        if values["status"] == "success":
            raise RuntimeError("source metadata failed")
        return original(con, values)

    monkeypatch.setattr(DataStore, "_record_minute_source_con", staticmethod(fail_success_record))
    result = MinuteDirectoryImporter(app_config, store).import_directory(source.parent, "5min")

    assert result.status == "partial_failed"
    assert not store.minute_symbol_year_path("5min", "000001.SZ", 2024).exists()
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM minute_partitions").fetchone()[0] == 0
        assert con.execute("SELECT status FROM minute_sources").fetchone()[0] == "failed"


def test_directory_batch_finishes_when_file_metadata_read_raises(app_config, tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    importer = MinuteDirectoryImporter(app_config, store)
    original = importer._import_file

    def fail_one(item, frequency, run_id, resume):
        if item.symbol == "000001.SZ":
            raise OSError("stat failed")
        return original(item, frequency, run_id, resume)

    monkeypatch.setattr(importer, "_import_file", fail_one)
    result = importer.import_directory(source, "5min")
    assert result.status == "partial_failed"
    assert result.files_failed == 1
    assert result.files_success == 1
    with store.connect() as con:
        status = con.execute(
            "SELECT status FROM minute_import_runs WHERE run_id = ?", [result.run_id]
        ).fetchone()[0]
    assert status == "partial_failed"


def test_directory_run_is_finished_when_progress_callback_raises(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    importer = MinuteDirectoryImporter(app_config, store)

    def fail_progress(*_args):
        raise RuntimeError("output closed")

    with pytest.raises(RuntimeError, match="output closed"):
        importer.import_directory(source, "5min", progress=fail_progress)
    with store.connect() as con:
        status, details = con.execute("SELECT status, details FROM minute_import_runs").fetchone()
    assert status == "failed"
    assert "output closed" in details


def test_flat_directory_without_manifest_infers_asset_types(app_config, tmp_path):
    source = tmp_path / "flat"
    source.mkdir()
    (source / "000001.SZ.csv").write_text(
        HEADER + "000001.SZ,2024-01-02 09:35:00,10,10,10,10,100,1000\n",
        encoding="utf-8-sig",
    )
    (source / "510300.SH.csv").write_text(
        HEADER + "510300.SH,2024-01-02 09:35:00,4,4,4,4,100,400\n",
        encoding="utf-8-sig",
    )

    result = MinuteDirectoryImporter(app_config, DataStore(app_config)).import_directory(
        source, "5min"
    )

    assert result.status == "success"
    assert result.files_success == 2


def test_directory_import_applies_bar_start_timestamp_convention(app_config, tmp_path):
    app_config.minute.timestamp_convention = "bar_start"
    source = tmp_path / "source" / "stock"
    source.mkdir(parents=True)
    (source / "000001.SZ.csv").write_text(
        HEADER + "000001.SZ,2024-01-02 09:30:00,10,10,10,10,100,1000\n",
        encoding="utf-8-sig",
    )
    store = DataStore(app_config)
    result = MinuteDirectoryImporter(app_config, store).import_directory(source.parent, "5min")
    assert result.status == "success"
    frame = pd.read_parquet(store.minute_symbol_year_path("5min", "000001.SZ", 2024))
    assert frame["bar_time"].iloc[0] == pd.Timestamp("2024-01-02 09:35:00")


def test_minute_partition_audit_checks_ohlcv_content(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    MinuteDirectoryImporter(app_config, store).import_directory(source, "5min")
    partition = store.minute_symbol_year_path("5min", "000001.SZ", 2024)
    pd.read_parquet(partition).assign(close=-1.0).to_parquet(partition, index=False)

    invalid = [
        item for item in store.audit_minute_partitions("5min") if item["status"] == "invalid"
    ]
    assert any(item["symbol"] == "000001.SZ" for item in invalid)


def test_minute_partition_audit_checks_date_grid_and_auction_semantics(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    MinuteDirectoryImporter(app_config, store).import_directory(source, "5min")
    partition = store.minute_symbol_year_path("5min", "000001.SZ", 2024)

    frame = pd.read_parquet(partition)
    frame.loc[0, "trade_date"] = date(2024, 1, 3)
    frame.to_parquet(partition, index=False)
    result = next(
        item
        for item in store.audit_minute_partitions("5min")
        if item["symbol"] == "000001.SZ" and item["year"] == 2024
    )
    assert "trade_date 与 bar_time" in result["reason"]

    frame["trade_date"] = frame["bar_time"].dt.date
    frame.loc[0, "is_auction"] = False
    frame.to_parquet(partition, index=False)
    result = next(
        item
        for item in store.audit_minute_partitions("5min")
        if item["symbol"] == "000001.SZ" and item["year"] == 2024
    )
    assert "is_auction" in result["reason"]

    frame.loc[0, "is_auction"] = True
    frame.loc[1, "bar_time"] = pd.Timestamp("2024-01-02 09:36:00")
    frame.loc[1, "trade_date"] = date(2024, 1, 2)
    frame.to_parquet(partition, index=False)
    with store.connect() as con:
        con.execute(
            "UPDATE minute_partitions SET max_time = ? "
            "WHERE frequency = '5min' AND symbol = '000001.SZ' AND year = 2024",
            [pd.Timestamp("2024-01-02 09:36:00")],
        )
    result = next(
        item
        for item in store.audit_minute_partitions("5min")
        if item["symbol"] == "000001.SZ" and item["year"] == 2024
    )
    assert "不符合5min交易时段" in result["reason"]


def test_startup_rolls_back_interrupted_minute_commit(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    MinuteDirectoryImporter(app_config, store).import_directory(source, "5min")
    target = store.minute_symbol_year_path("5min", "000001.SZ", 2024).resolve()
    original = pd.read_parquet(target)

    commit_id = "interrupted"
    recovery = store.root / ".staging" / "minute-commit" / commit_id
    backup = recovery / "backups" / "00000000.parquet"
    backup.parent.mkdir(parents=True)
    target.replace(backup)
    replacement = original.assign(close=999.0)
    replacement.to_parquet(target, index=False)
    (recovery / "journal.json").write_text(
        json.dumps(
            {
                "commit_id": commit_id,
                "operations": [{"target": str(target), "backup": str(backup)}],
            }
        ),
        encoding="utf-8",
    )
    with store.connect() as con:
        con.execute(
            "INSERT INTO minute_commit_log VALUES (?, 'preparing', current_timestamp, "
            "current_timestamp)",
            [commit_id],
        )

    recovered = DataStore(app_config)
    pd.testing.assert_frame_equal(pd.read_parquet(target), original)
    with recovered.connect() as con:
        assert con.execute("SELECT count(*) FROM minute_commit_log").fetchone()[0] == 0
    assert not recovery.exists()


def test_startup_keeps_committed_minute_files_and_cleans_backup(app_config, tmp_path):
    source = tmp_path / "source"
    _write_dataset(source)
    store = DataStore(app_config)
    MinuteDirectoryImporter(app_config, store).import_directory(source, "5min")
    target = store.minute_symbol_year_path("5min", "000001.SZ", 2024).resolve()
    original = pd.read_parquet(target)

    commit_id = "committed"
    recovery = store.root / ".staging" / "minute-commit" / commit_id
    backup = recovery / "backups" / "00000000.parquet"
    backup.parent.mkdir(parents=True)
    target.replace(backup)
    original.assign(close=888.0).to_parquet(target, index=False)
    (recovery / "journal.json").write_text(
        json.dumps(
            {
                "commit_id": commit_id,
                "operations": [{"target": str(target), "backup": str(backup)}],
            }
        ),
        encoding="utf-8",
    )
    with store.connect() as con:
        con.execute(
            "INSERT INTO minute_commit_log VALUES (?, 'committed', current_timestamp, "
            "current_timestamp)",
            [commit_id],
        )

    recovered = DataStore(app_config)
    assert set(pd.read_parquet(target)["close"]) == {888.0}
    with recovered.connect() as con:
        assert con.execute("SELECT count(*) FROM minute_commit_log").fetchone()[0] == 0
    assert not recovery.exists()
