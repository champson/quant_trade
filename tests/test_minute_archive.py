from __future__ import annotations

import zipfile

import pandas as pd

from quant_trade.data.minute_archive import MinuteArchiveImporter, resample_minutes
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
    partition = store.minute_partition("2024-01-02")
    data = pd.read_parquet(partition)
    assert len(data) == 3
    again = importer.import_archive(app_config.minute.archive / "sample.zip", profile)
    assert again.status == "skipped"


def test_resampling_does_not_bridge_lunch(app_config):
    times = pd.to_datetime(["2024-01-02 11:29", "2024-01-02 11:30", "2024-01-02 13:01"])
    data = pd.DataFrame({
        "symbol": ["000001.SZ"] * 3, "trade_date": ["2024-01-02"] * 3,
        "bar_time": times, "open": [1, 2, 10], "high": [2, 3, 11],
        "low": [1, 2, 10], "close": [2, 3, 11], "volume": [1, 1, 1], "amount": [1, 1, 1],
    })
    result = resample_minutes(data, "5min")
    assert len(result) == 2
    assert set(result["open"]) == {1, 10}


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
    data = pd.read_parquet(store.minute_partition("2024-01-03"))
    assert data.iloc[0]["bar_time"] == pd.Timestamp("2024-01-03 09:31:00")
    assert data.iloc[0]["symbol"] == "000001.SZ"
