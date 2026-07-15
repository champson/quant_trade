from __future__ import annotations

from pathlib import Path

import pytest

from quant_trade.config import AppConfig, MinuteConfig, PathsConfig


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
            runs_dir=tmp_path / "runs",
            database=tmp_path / "data" / "test.duckdb",
        ),
        minute=MinuteConfig(
            inbox=tmp_path / "data" / "inbox" / "minute",
            archive=tmp_path / "data" / "archive" / "minute",
            quarantine=tmp_path / "data" / "quarantine" / "minute",
        ),
    )

