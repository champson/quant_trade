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
