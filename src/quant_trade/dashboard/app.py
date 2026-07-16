from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from quant_trade.config import load_config


def _latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def main() -> None:
    cfg = load_config()
    st.set_page_config(page_title="Quant Trade", layout="wide")
    st.title("Quant Trade 量化复盘与研究")
    tabs = st.tabs(["今日复盘", "策略信号", "回测", "数据与任务"])

    with tabs[0]:
        summary = _latest(list((cfg.paths.artifacts_dir / "reviews").glob("market_summary_*.json")))
        image = _latest(list((cfg.paths.artifacts_dir / "reviews").glob("market_breadth_*.png")))
        if summary:
            values = json.loads(summary.read_text(encoding="utf-8"))
            cols = st.columns(5)
            for col, key, label in zip(cols, ["stocks", "up", "down", "mean_return", "median_return"], ["股票数", "上涨", "下跌", "平均涨幅", "中位涨幅"]):
                value = values.get(key, "-")
                if key.endswith("return") and isinstance(value, (float, int)):
                    value = f"{value:.2%}"
                col.metric(label, value)
        if image:
            st.image(str(image))
        if not summary:
            st.info("尚无复盘结果，请运行 qt daily run")

    with tabs[1]:
        for strategy_dir in sorted((cfg.paths.artifacts_dir / "signals").glob("*")):
            latest = _latest(list(strategy_dir.glob("signal_*.csv")))
            if latest:
                st.subheader(strategy_dir.name)
                st.dataframe(pd.read_csv(latest), use_container_width=True)

    with tabs[2]:
        for strategy_dir in sorted((cfg.paths.artifacts_dir / "backtests").glob("*")):
            st.subheader(strategy_dir.name)
            report = strategy_dir / "report.html"
            if report.exists():
                components.html(report.read_text(encoding="utf-8"), height=1100, scrolling=True)
            else:
                if (strategy_dir / "metrics.json").exists():
                    st.json(json.loads((strategy_dir / "metrics.json").read_text()))
                if (strategy_dir / "equity.png").exists():
                    st.image(str(strategy_dir / "equity.png"))

    with tabs[3]:
        if cfg.paths.database.exists():
            con = duckdb.connect(str(cfg.paths.database), read_only=True)
            st.subheader("最近运行")
            st.dataframe(con.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 30").df(), use_container_width=True)
            st.subheader("数据源请求")
            st.dataframe(con.execute("SELECT * FROM data_fetches ORDER BY fetched_at DESC LIMIT 50").df(), use_container_width=True)
            st.subheader("分钟文件导入")
            st.dataframe(con.execute("SELECT * FROM minute_imports ORDER BY imported_at DESC LIMIT 50").df(), use_container_width=True)
            st.subheader("分钟目录导入")
            st.dataframe(con.execute("SELECT * FROM minute_import_runs ORDER BY started_at DESC LIMIT 30").df(), use_container_width=True)
            st.subheader("分钟数据覆盖")
            st.dataframe(con.execute("""
                SELECT frequency, asset_type, COUNT(DISTINCT symbol) AS symbols,
                       SUM(rows) AS rows, MIN(min_time) AS min_time,
                       MAX(max_time) AS max_time
                FROM minute_partitions
                GROUP BY frequency, asset_type ORDER BY frequency, asset_type
            """).df(), use_container_width=True)
            con.close()
        else:
            st.info("数据库尚未创建")


if __name__ == "__main__":
    main()
