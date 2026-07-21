from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from quant_trade.data.minute_archive import MinuteArchiveImporter, resample_minutes
from quant_trade.data.quality import DataQualityError
from quant_trade.data.storage import DataStore


def _zip(path, member="000001.SZ.csv"):
    csv = """ts_code,trade_time,open,high,low,close,vol,amount
000001.SZ,2024-01-02 09:31:00,10,10.2,9.9,10.1,100,1010
000001.SZ,2024-01-02 09:32:00,10.1,10.3,10,10.2,200,2030
000001.SZ,2024-01-02 13:01:00,10.2,10.4,10.1,10.3,300,3090
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, csv)


def test_minute_zip_is_profiled_imported_and_deduplicated(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "sample.zip"
    _zip(path)
    store = DataStore(app_config)
    importer = MinuteArchiveImporter(app_config, store)
    profile = importer.inspect(path)
    assert profile.columns["trade_time"] == "bar_time"
    result = importer.import_archive(path, profile)
    assert result.status == "success"
    assert result.rows == 3
    assert (app_config.minute.archive / "sample.zip").exists()
    partition = store.minute_symbol_year_path("1min", "000001.SZ", 2024)
    data = store.read_minute(["000001.SZ"], "2024-01-02 09:00", "2024-01-02 14:00", "1min")
    assert partition.exists()
    assert len(data) == 3
    again = importer.import_archive(app_config.minute.archive / "sample.zip", profile)
    assert again.status == "skipped"
    partition.unlink()
    repaired = importer.import_archive(app_config.minute.archive / "sample.zip", profile)
    assert repaired.status == "success"
    assert partition.exists()


def test_resampling_does_not_bridge_lunch(app_config):
    times = pd.to_datetime(["2024-01-02 11:29", "2024-01-02 11:30", "2024-01-02 13:01"])
    data = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 3,
            "trade_date": ["2024-01-02"] * 3,
            "bar_time": times,
            "open": [1, 2, 10],
            "high": [2, 3, 11],
            "low": [1, 2, 10],
            "close": [2, 3, 11],
            "volume": [1, 1, 1],
            "amount": [1, 1, 1],
        }
    )
    result = resample_minutes(data, "5min")
    assert len(result) == 2
    assert set(result["open"]) == {1, 10}


def test_resampling_folds_1300_reopen_into_first_afternoon_bucket():
    data = pd.DataFrame(
        {
            "symbol": ["510300.SH"] * 4,
            "trade_date": ["2024-01-02"] * 4,
            "bar_time": pd.to_datetime(
                [
                    "2024-01-02 13:00",
                    "2024-01-02 13:05",
                    "2024-01-02 13:10",
                    "2024-01-02 13:15",
                ]
            ),
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [1, 2, 3, 4],
            "close": [2, 3, 4, 5],
            "volume": [1, 1, 1, 1],
            "amount": [1, 1, 1, 1],
        }
    )
    result = resample_minutes(data, "15min")
    assert len(result) == 1
    assert result.iloc[0]["bar_time"] == pd.Timestamp("2024-01-02 13:15")
    assert result.iloc[0]["open"] == 1


def test_separate_date_and_clock_columns_are_combined(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "split-time.zip"
    csv = """code,date,time,open,high,low,close,volume,amount
000001,2024-01-03,09:31:00,10,11,9,10.5,100,1000
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("000001.csv", csv)
    store = DataStore(app_config)
    result = MinuteArchiveImporter(app_config, store).import_archive(path)
    assert result.status == "success"
    data = store.read_minute(["000001.SZ"], "2024-01-03 09:00", "2024-01-03 10:00", "1min")
    assert data.iloc[0]["bar_time"] == pd.Timestamp("2024-01-03 09:31:00")
    assert data.iloc[0]["symbol"] == "000001.SZ"


def test_zip_frequency_is_catalogued_in_unified_storage(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "five-minute.zip"
    csv = """ts_code,trade_time,open,high,low,close,vol,amount
510300.SH,2024-01-02 09:35:00,4,4.1,3.9,4.0,100,400
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("510300.SH.csv", csv)
    store = DataStore(app_config)
    result = MinuteArchiveImporter(app_config, store).import_archive(path, frequency="5min")
    assert result.status == "success"
    data = store.read_minute(["510300.SH"], "2024-01-02 09:00", "2024-01-02 10:00", "5min")
    assert len(data) == 1
    assert data.iloc[0]["asset_type"] == "etf"


def test_auto_asset_type_recognizes_sz_lof_and_csi_index():
    assert MinuteArchiveImporter._asset_type("160106.SZ", "auto") == "etf"
    assert MinuteArchiveImporter._asset_type("930050.CSI", "auto") == "index"


def test_csi_index_suffix_is_canonicalized_during_import(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "csi-index.zip"
    csv = """ts_code,trade_time,open,high,low,close,vol,amount
930050.CSI,2024-01-02 09:35:00,100,101,99,100,10,1000
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("930050.CSI.csv", csv)
    store = DataStore(app_config)
    result = MinuteArchiveImporter(app_config, store).import_archive(path, frequency="5min")
    assert result.status == "success"
    data = store.read_minute(["930050.SH"], "2024-01-02", "2024-01-03", "5min")
    assert data[["symbol", "asset_type"]].iloc[0].tolist() == ["930050.SH", "index"]


def test_invalid_zip_preflight_is_recorded_and_quarantined(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "not-a-zip.zip"
    path.write_text("broken", encoding="utf-8")
    store = DataStore(app_config)
    with pytest.raises(DataQualityError):
        MinuteArchiveImporter(app_config, store).import_archive(path)
    assert (app_config.minute.quarantine / path.name).exists()
    with store.connect() as con:
        status, details = con.execute(
            "SELECT status, details FROM minute_archive_imports"
        ).fetchone()
    assert status == "failed"
    assert "preflight" in details


def test_import_inbox_continues_after_filesystem_error(app_config, monkeypatch):
    app_config.ensure_directories()
    first = app_config.minute.inbox / "a.zip"
    second = app_config.minute.inbox / "b.zip"
    _zip(first)
    _zip(second)
    importer = MinuteArchiveImporter(app_config, DataStore(app_config))

    def fail(path, **kwargs):
        raise OSError(f"cannot move {path.name}")

    monkeypatch.setattr(importer, "import_archive", fail)
    results = importer.import_inbox()
    assert [item.status for item in results] == ["failed", "failed"]
    assert all("cannot move" in item.warnings[0] for item in results)


def test_zip_failure_rolls_back_all_staged_symbols(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "broken.zip"
    good = """ts_code,trade_time,open,high,low,close,vol,amount
000001.SZ,2024-01-02 09:31:00,10,10.2,9.9,10.1,100,1010
"""
    bad = """ts_code,trade_time,open,high,low,close,vol,amount
000002.SZ,2024-01-02 09:31:00,10,9,9.9,10.1,100,1010
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("000001.SZ.csv", good)
        archive.writestr("000002.SZ.csv", bad)
    store = DataStore(app_config)
    with pytest.raises(DataQualityError):
        MinuteArchiveImporter(app_config, store).import_archive(path)
    assert store.read_minute(["000001.SZ"], "2024-01-02 09:00", "2024-01-02 10:00", "1min").empty


def test_incremental_zip_merges_existing_symbol_year(app_config):
    app_config.ensure_directories()
    store = DataStore(app_config)
    importer = MinuteArchiveImporter(app_config, store)
    for name, day, price in (("jan.zip", "2024-01-02", 10), ("feb.zip", "2024-02-02", 11)):
        path = app_config.minute.inbox / name
        csv = f"""ts_code,trade_time,open,high,low,close,vol,amount
000001.SZ,{day} 09:31:00,{price},{price},{price},{price},100,1000
"""
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("000001.SZ.csv", csv)
        assert importer.import_archive(path).status == "success"

    data = store.read_minute(["000001.SZ"], "2024-01-01", "2024-12-31 23:59", frequency="1min")
    assert data["bar_time"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-02-02"]


def test_overlapping_incremental_zip_keeps_older_archive_resumable(app_config):
    app_config.ensure_directories()
    store = DataStore(app_config)
    importer = MinuteArchiveImporter(app_config, store)
    archives = []
    payloads = {
        "base.zip": [
            ("09:31:00", 10),
            ("09:33:00", 12),
        ],
        "gap.zip": [("09:32:00", 11)],
    }
    for name, rows in payloads.items():
        path = app_config.minute.inbox / name
        csv = "ts_code,trade_time,open,high,low,close,vol,amount\n" + "".join(
            f"000001.SZ,2024-01-02 {clock},{price},{price},{price},{price},100,1000\n"
            for clock, price in rows
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("000001.SZ.csv", csv)
        file_hash = importer.hash_file(path)
        importer.import_archive(path)
        archives.append(file_hash)

    assert store.minute_imported(archives[0], "1min", "auto")
    assert store.minute_imported(archives[1], "1min", "auto")
    data = store.read_minute(["000001.SZ"], "2024-01-02 09:30", "2024-01-02 09:34", "1min")
    assert data["bar_time"].dt.strftime("%H:%M").tolist() == ["09:31", "09:32", "09:33"]


def test_zip_resume_detects_partition_value_rewrite(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "fingerprint.zip"
    _zip(path)
    store = DataStore(app_config)
    importer = MinuteArchiveImporter(app_config, store)
    file_hash = importer.hash_file(path)
    importer.import_archive(path)
    assert store.minute_imported(file_hash, "1min", "auto")

    partition = store.minute_symbol_year_path("1min", "000001.SZ", 2024)
    pd.read_parquet(partition).assign(close=999.0).to_parquet(partition, index=False)
    assert not store.minute_imported(file_hash, "1min", "auto")


def test_zip_members_are_imported_in_deterministic_name_order(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "reversed-members.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for year in (2025, 2024):
            archive.writestr(
                f"000001.SZ-{year}.csv",
                "ts_code,trade_time,open,high,low,close,vol,amount\n"
                f"000001.SZ,{year}-01-02 09:31:00,10,10,10,10,100,1000\n",
            )

    store = DataStore(app_config)
    result = MinuteArchiveImporter(app_config, store).import_archive(path)

    assert result.status == "success"
    data = store.read_minute(["000001.SZ"], "2024-01-01", "2025-12-31", "1min")
    assert data["bar_time"].dt.year.tolist() == [2024, 2025]


def test_zip_rejects_unparseable_rows_instead_of_dropping_them(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "invalid-row.zip"
    csv = """ts_code,trade_time,open,high,low,close,vol,amount
000001.SZ,2024-01-02 09:31:00,10,10,10,10,100,1000
000001.SZ,BAD_TIME,10,10,10,10,100,1000
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("000001.SZ.csv", csv)
    store = DataStore(app_config)
    with pytest.raises(DataQualityError, match="无法解析"):
        MinuteArchiveImporter(app_config, store).import_archive(path)
    assert store.read_minute(["000001.SZ"], "2024-01-01", "2024-12-31", "1min").empty


def test_zip_rejects_timestamp_grid_that_does_not_match_frequency(app_config):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "wrong-grid.zip"
    _zip(path)
    with pytest.raises(DataQualityError, match="不符合5min"):
        MinuteArchiveImporter(app_config, DataStore(app_config)).import_archive(
            path, frequency="5min"
        )


def test_archive_metadata_failure_rolls_back_partition_and_success_record(app_config, monkeypatch):
    app_config.ensure_directories()
    path = app_config.minute.inbox / "metadata-failure.zip"
    _zip(path)
    store = DataStore(app_config)
    original = DataStore._record_minute_import_con

    def fail_success_record(con, values):
        if values["status"] == "success":
            raise RuntimeError("archive metadata failed")
        return original(con, values)

    monkeypatch.setattr(DataStore, "_record_minute_import_con", staticmethod(fail_success_record))
    with pytest.raises(DataQualityError, match="archive metadata failed"):
        MinuteArchiveImporter(app_config, store).import_archive(path)

    assert not store.minute_symbol_year_path("1min", "000001.SZ", 2024).exists()
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM minute_partitions").fetchone()[0] == 0
        assert con.execute("SELECT status FROM minute_archive_imports").fetchone()[0] == "failed"
