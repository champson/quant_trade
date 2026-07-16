from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_trade.data.minute_directory import MinuteDirectoryImporter
from quant_trade.data.storage import DataStore


HEADER = "ts_code,trade_time,open,close,high,low,vol,amount\n"


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
    pd.DataFrame([
        {"category": "stock", "ts_code": "000001.SZ", "relative_file": "stock/000001.SZ.csv", "rows": 3},
        {"category": "etf", "ts_code": "510300.SH", "relative_file": "etf/510300.SH.csv", "rows": 3},
        {"category": "index", "ts_code": "000001.SH", "relative_file": "index/000001.SH.csv", "rows": 0},
    ]).to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")


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

    queried = store.read_minute(
        ["000001.SZ"], "2024-01-02 09:00", "2024-01-02 10:00", "5min"
    )
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
