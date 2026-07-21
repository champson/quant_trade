from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from quant_trade.config import load_config


def _latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_child(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"产物路径越界: {value}") from exc
    return path


def _latest_review_files(review_dir: Path) -> dict[str, Path]:
    """Resolve one fully published review generation."""
    manifests: list[tuple[pd.Timestamp, Path, dict]] = []
    for manifest in review_dir.glob("review_manifest_*.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifests.append((pd.Timestamp(payload["as_of"]).normalize(), manifest, payload))
        except (KeyError, OSError, TypeError, ValueError):
            continue
    for _, manifest, payload in sorted(manifests, key=lambda item: item[0], reverse=True):
        try:
            resolved = {
                str(name): _resolved_child(review_dir, str(filename))
                for name, filename in payload["files"].items()
            }
            checksums = payload.get("sha256", {})
            files_valid = resolved.get("summary", Path()).is_file() and all(
                path.is_file() for path in resolved.values()
            )
            checksums_valid = not checksums or (
                set(checksums) == set(resolved)
                and all(_sha256(resolved[name]) == value for name, value in checksums.items())
            )
            if files_valid and checksums_valid:
                return resolved
        except (KeyError, OSError, TypeError, ValueError):
            continue
    if manifests:
        # Do not mix independently selected legacy files while a manifest-based
        # generation is incomplete or corrupt.
        return {}

    # Backward-compatible fallback for reports created before manifests existed.
    summaries = sorted(review_dir.glob("market_summary_*.json"), key=lambda path: path.stem)
    summary = summaries[-1] if summaries else None
    if summary is None:
        return {}
    stamp = summary.stem.removeprefix("market_summary_")
    candidates = {
        "summary": summary,
        "csv": review_dir / f"market_breadth_{stamp}.csv",
        "png": review_dir / f"market_breadth_{stamp}.png",
    }
    return {name: path for name, path in candidates.items() if path.is_file()}


def _latest_daily_generation(artifact_root: Path) -> dict | None:
    """Return the newest checksum-valid daily review/signal generation."""
    candidates: list[tuple[pd.Timestamp, str, Path, dict]] = []
    for manifest in (artifact_root / "daily").glob("daily_manifest_*.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            candidates.append(
                (
                    pd.Timestamp(payload["as_of"]).normalize(),
                    str(payload.get("generated_at", "")),
                    manifest,
                    payload,
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
    for _, _, manifest, payload in sorted(
        candidates, key=lambda item: (item[0], item[1]), reverse=True
    ):
        try:
            checksums = {
                _resolved_child(artifact_root, relative): expected
                for relative, expected in payload["sha256"].items()
            }
            if not checksums or not all(
                path.is_file() and _sha256(path) == expected for path, expected in checksums.items()
            ):
                continue
            review_files = {
                name: _resolved_child(artifact_root, relative)
                for name, relative in payload["review_files"].items()
            }
            signal_files = {
                strategy: {
                    kind: _resolved_child(artifact_root, relative)
                    for kind, relative in files.items()
                }
                for strategy, files in payload["signal_files"].items()
            }
            if not all(path.is_file() for path in review_files.values()):
                continue
            if not all(
                path.is_file() for files in signal_files.values() for path in files.values()
            ):
                continue
            return {
                "as_of": pd.Timestamp(payload["as_of"]).normalize(),
                "manifest": manifest,
                "review_files": review_files,
                "signal_files": signal_files,
            }
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return None


def _latest_signal_csv(strategy_dir: Path) -> Path | None:
    candidates = sorted(strategy_dir.glob("signal_*.csv"), key=lambda path: path.stem)
    return candidates[-1] if candidates else None


def _review_date(files: dict[str, Path]) -> pd.Timestamp | None:
    summary = files.get("summary")
    if summary is None:
        return None
    try:
        return pd.Timestamp(json.loads(summary.read_text(encoding="utf-8"))["as_of"]).normalize()
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _signal_date(path: Path | None) -> pd.Timestamp | None:
    if path is None:
        return None
    try:
        return pd.to_datetime(path.stem.removeprefix("signal_"), format="%Y%m%d").normalize()
    except ValueError:
        return None


def main() -> None:
    cfg = load_config()
    daily_generation = _latest_daily_generation(cfg.paths.artifacts_dir)
    st.set_page_config(page_title="Quant Trade", layout="wide")
    st.title("Quant Trade 量化复盘与研究")
    tabs = st.tabs(["今日复盘", "策略信号", "回测", "数据与任务"])

    with tabs[0]:
        review_files = _latest_review_files(cfg.paths.artifacts_dir / "reviews")
        if daily_generation is not None:
            standalone_date = _review_date(review_files)
            if standalone_date is None or standalone_date <= daily_generation["as_of"]:
                review_files = daily_generation["review_files"]
        summary = review_files.get("summary")
        breadth = review_files.get("csv")
        image = review_files.get("png")
        if summary:
            values = json.loads(summary.read_text(encoding="utf-8"))
            cols = st.columns(5)
            for col, key, label in zip(
                cols,
                ["stocks", "up", "down", "mean_return", "median_return"],
                ["股票数", "上涨", "下跌", "平均涨幅", "中位涨幅"],
            ):
                value = values.get(key, "-")
                if key.endswith("return") and isinstance(value, (float, int)):
                    value = f"{value:.2%}"
                col.metric(label, value)
        if breadth:
            st.dataframe(pd.read_csv(breadth), width="stretch", hide_index=True)
        elif image:
            st.image(str(image), width="stretch")
        if not summary:
            st.info("尚无复盘结果，请运行 qt daily run")

    with tabs[1]:
        daily_signals = daily_generation["signal_files"] if daily_generation is not None else {}
        strategy_names = {
            path.name for path in (cfg.paths.artifacts_dir / "signals").glob("*") if path.is_dir()
        } | set(daily_signals)
        for strategy_name in sorted(strategy_names):
            strategy_dir = cfg.paths.artifacts_dir / "signals" / strategy_name
            standalone = _latest_signal_csv(strategy_dir)
            daily_csv = daily_signals.get(strategy_name, {}).get("csv")
            latest = daily_csv or standalone
            if (
                standalone is not None
                and daily_generation is not None
                and (_signal_date(standalone) or pd.Timestamp.min) > daily_generation["as_of"]
            ):
                latest = standalone
            if latest:
                st.subheader(strategy_name)
                st.dataframe(pd.read_csv(latest), width="stretch")

    with tabs[2]:
        for strategy_dir in sorted((cfg.paths.artifacts_dir / "backtests").glob("*")):
            st.subheader(strategy_dir.name)
            reports = list(strategy_dir.glob("*/report.html"))
            report = _latest(reports) or strategy_dir / "report.html"
            if report.exists():
                components.html(report.read_text(encoding="utf-8"), height=1100, scrolling=True)
            else:
                run_dirs = [path for path in strategy_dir.glob("*") if path.is_dir()]
                latest_run = _latest(run_dirs)
                result_dir = latest_run or strategy_dir
                if (result_dir / "metrics.json").exists():
                    st.json(json.loads((result_dir / "metrics.json").read_text()))
                if (result_dir / "equity.png").exists():
                    st.image(str(result_dir / "equity.png"), width="stretch")

    with tabs[3]:
        if cfg.paths.database.exists():
            try:
                with duckdb.connect(str(cfg.paths.database), read_only=True) as con:
                    st.subheader("最近运行")
                    st.dataframe(
                        con.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 30").df(),
                        width="stretch",
                    )
                    st.subheader("数据源请求")
                    st.dataframe(
                        con.execute(
                            "SELECT * FROM data_fetches ORDER BY fetched_at DESC LIMIT 50"
                        ).df(),
                        width="stretch",
                    )
                    st.subheader("分钟文件导入")
                    st.dataframe(
                        con.execute(
                            """
                            SELECT * FROM minute_archive_imports
                            ORDER BY imported_at DESC LIMIT 50
                            """
                        ).df(),
                        width="stretch",
                    )
                    st.subheader("分钟目录导入")
                    st.dataframe(
                        con.execute(
                            "SELECT * FROM minute_import_runs ORDER BY started_at DESC LIMIT 30"
                        ).df(),
                        width="stretch",
                    )
                    st.subheader("分钟数据覆盖")
                    st.dataframe(
                        con.execute("""
                        SELECT frequency, asset_type, COUNT(DISTINCT symbol) AS symbols,
                               SUM(rows) AS rows, MIN(min_time) AS min_time,
                               MAX(max_time) AS max_time
                        FROM minute_partitions
                        GROUP BY frequency, asset_type ORDER BY frequency, asset_type
                        """).df(),
                        width="stretch",
                    )
            except duckdb.Error as exc:
                st.warning("数据正在更新或数据库暂时不可读，请稍后刷新。")
                st.caption(str(exc))
        else:
            st.info("数据库尚未创建")


if __name__ == "__main__":
    main()
