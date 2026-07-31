from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import duckdb
import pytest
from streamlit.testing.v1 import AppTest

from quant_trade.data.storage import DataStore
from quant_trade.dashboard.app import (
    _latest_daily_generation,
    _latest_review_files,
    _review_history,
)


def test_dashboard_resolves_one_manifest_generation(tmp_path):
    names = {
        "summary": "market_summary_2024-01-02.json",
        "csv": "market_breadth_2024-01-02.csv",
        "png": "market_breadth_2024-01-02.png",
    }
    for filename in names.values():
        (tmp_path / filename).write_text("data", encoding="utf-8")
    (tmp_path / "market_breadth_2024-01-03.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "review_manifest_2024-01-02.json").write_text(
        json.dumps({"as_of": "2024-01-02", "files": names}),
        encoding="utf-8",
    )

    resolved = _latest_review_files(tmp_path)

    assert {name: path.name for name, path in resolved.items()} == names


def test_dashboard_does_not_fallback_to_mixed_files_during_publish(tmp_path):
    names = {
        "summary": "market_summary_2024-01-02.json",
        "csv": "market_breadth_2024-01-02.csv",
    }
    checksums = {}
    for name, filename in names.items():
        payload = filename.encode()
        (tmp_path / filename).write_bytes(payload)
        checksums[name] = sha256(payload).hexdigest()
    (tmp_path / "review_manifest_2024-01-02.json").write_text(
        json.dumps({"as_of": "2024-01-02", "files": names, "sha256": checksums}),
        encoding="utf-8",
    )
    (tmp_path / names["csv"]).write_text("new generation", encoding="utf-8")

    assert _latest_review_files(tmp_path) == {}


def test_dashboard_chooses_review_by_as_of_not_file_mtime(tmp_path):
    for stamp in ("2026-07-21", "2020-01-02"):
        summary = tmp_path / f"market_summary_{stamp}.json"
        summary.write_text(json.dumps({"as_of": stamp}), encoding="utf-8")
        (tmp_path / f"review_manifest_{stamp}.json").write_text(
            json.dumps(
                {
                    "as_of": stamp,
                    "files": {"summary": summary.name},
                }
            ),
            encoding="utf-8",
        )

    resolved = _latest_review_files(tmp_path)

    assert resolved["summary"].name == "market_summary_2026-07-21.json"


def test_dashboard_resolves_checksum_valid_daily_generation(tmp_path):
    review = tmp_path / "reviews" / "market_summary_2024-01-08.json"
    signal = tmp_path / "signals" / "logbias" / "signal_20240108.csv"
    review.parent.mkdir(parents=True)
    signal.parent.mkdir(parents=True)
    review.write_text(json.dumps({"as_of": "2024-01-08"}), encoding="utf-8")
    signal.write_text("symbol,target_weight\n510300.SH,1\n", encoding="utf-8")
    manifest_dir = tmp_path / "daily"
    manifest_dir.mkdir()
    manifest = {
        "as_of": "2024-01-08",
        "generated_at": "2024-01-08T18:00:00+08:00",
        "review_files": {"summary": str(review.relative_to(tmp_path))},
        "signal_files": {"logbias": {"csv": str(signal.relative_to(tmp_path))}},
        "sha256": {
            str(review.relative_to(tmp_path)): sha256(review.read_bytes()).hexdigest(),
            str(signal.relative_to(tmp_path)): sha256(signal.read_bytes()).hexdigest(),
        },
    }
    (manifest_dir / "daily_manifest_2024-01-08.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    resolved = _latest_daily_generation(tmp_path)

    assert resolved is not None
    assert resolved["review_files"]["summary"] == review
    assert resolved["signal_files"]["logbias"]["csv"] == signal


def test_review_note_round_trip_and_update(app_config):
    store = DataStore(app_config)

    first = store.save_review_note(
        "2026-07-29",
        {
            "headline": "缩量震荡，控制仓位",
            "market_state": "震荡偏弱",
            "sentiment_score": 2,
            "discipline_score": 4,
            "position_pct": 0.35,
            "portfolio_return": -0.003,
            "market_observation": "指数分化。",
            "trade_review": "没有追高。",
            "lessons": "等待确认。",
            "next_plan": "观察量能。",
            "tags": ["缩量", "防守", "缩量"],
        },
    )
    updated = store.save_review_note(
        "2026-07-29",
        {
            **first,
            "headline": "缩量震荡，计划保持防守",
            "tags": "缩量, 防守",
        },
    )

    assert updated["headline"] == "缩量震荡，计划保持防守"
    assert updated["tags"] == ["缩量", "防守"]
    assert updated["created_at"] == first["created_at"]
    notes = store.list_review_notes()
    assert notes["trade_date"].dt.date.tolist() == [first["trade_date"]]


def test_review_note_validates_scores_and_position(app_config):
    store = DataStore(app_config)

    with pytest.raises(ValueError, match="市场体感评分"):
        store.save_review_note("2026-07-29", {"sentiment_score": 6})
    with pytest.raises(ValueError, match="仓位比例"):
        store.save_review_note("2026-07-29", {"position_pct": 1.2})


def test_dashboard_history_lists_every_valid_review_date(tmp_path):
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    for stamp in ("2026-07-28", "2026-07-29"):
        summary = review_dir / f"market_summary_{stamp}.json"
        summary.write_text(json.dumps({"as_of": stamp}), encoding="utf-8")
        (review_dir / f"review_manifest_{stamp}.json").write_text(
            json.dumps(
                {
                    "as_of": stamp,
                    "files": {"summary": summary.name},
                }
            ),
            encoding="utf-8",
        )

    history = _review_history(tmp_path)

    assert [item["as_of"].date().isoformat() for item in history] == [
        "2026-07-29",
        "2026-07-28",
    ]


def test_dashboard_app_starts_with_empty_workspace(tmp_path, monkeypatch):
    config = tmp_path / "dashboard.yaml"
    config.write_text(
        f"""
paths:
  data_dir: {tmp_path / "data"}
  artifacts_dir: {tmp_path / "artifacts"}
  runs_dir: {tmp_path / "runs"}
  database: {tmp_path / "data" / "dashboard.duckdb"}
strategies: {{}}
review:
  indices: {{}}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("QT_CONFIG", str(config))
    app_path = Path(__file__).parents[1] / "src" / "quant_trade" / "dashboard" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert app.title == []
    assert app.info
    assert app.text_input[0].label == "一句话结论"

    app.text_input[0].input("测试当天手动复盘")
    app.button[0].click()
    app.run()

    assert not app.exception
    with duckdb.connect(str(tmp_path / "data" / "dashboard.duckdb"), read_only=True) as con:
        assert con.execute("SELECT headline FROM review_notes").fetchone() == ("测试当天手动复盘",)


def test_dashboard_renders_historical_review(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    review_dir = artifact_root / "reviews"
    review_dir.mkdir(parents=True)
    stamp = "2026-07-29"
    summary = review_dir / f"market_summary_{stamp}.json"
    matrix = review_dir / f"daily_review_{stamp}.csv"
    summary.write_text(
        json.dumps(
            {
                "as_of": stamp,
                "stocks": 5500,
                "up": 3200,
                "down": 2200,
                "mean_return": 0.012,
                "median_return": 0.008,
            }
        ),
        encoding="utf-8",
    )
    matrix.write_text(
        "名称,今年,本月,本周,当天,BIAS25\n"
        "涨幅>10%,1%,1%,1%,1%,\n"
        "涨幅>5%到10%,2%,2%,2%,2%,\n"
        "涨幅>0%到5%,3%,3%,3%,50%,\n"
        "涨幅>-5%到0%,4%,4%,4%,40%,\n"
        "涨幅>-10%到-5%,5%,5%,5%,5%,\n"
        "涨幅小于-10%,6%,6%,6%,2%,\n"
        "A股上涨比例,50%,50%,50%,58%,\n"
        "A股下跌比例,50%,50%,50%,40%,\n"
        "A股涨幅中位数,1%,1%,1%,0.8%,\n",
        encoding="utf-8",
    )
    (review_dir / f"review_manifest_{stamp}.json").write_text(
        json.dumps(
            {
                "as_of": stamp,
                "files": {
                    "summary": summary.name,
                    "daily_review": matrix.name,
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "history-dashboard.yaml"
    config.write_text(
        f"""
paths:
  data_dir: {tmp_path / "data"}
  artifacts_dir: {artifact_root}
  runs_dir: {tmp_path / "runs"}
  database: {tmp_path / "data" / "dashboard.duckdb"}
strategies: {{}}
review:
  indices: {{}}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("QT_CONFIG", str(config))
    app_path = Path(__file__).parents[1] / "src" / "quant_trade" / "dashboard" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=10).run()

    assert not app.exception
    assert len(app.metric) == 8
    assert app.selectbox[0].label == "历史复盘日期"
