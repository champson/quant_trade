from __future__ import annotations

import json
from hashlib import sha256

from quant_trade.dashboard.app import _latest_daily_generation, _latest_review_files


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
