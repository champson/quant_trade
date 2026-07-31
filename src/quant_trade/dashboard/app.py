from __future__ import annotations

import hashlib
import json
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from quant_trade.config import AppConfig, load_config
from quant_trade.data.storage import DataStore


MARKET_STATES = ["强势上涨", "震荡偏强", "横盘整理", "震荡偏弱", "弱势下跌"]
RETURN_BUCKETS = [
    "涨幅>10%",
    "涨幅>5%到10%",
    "涨幅>0%到5%",
    "涨幅>-5%到0%",
    "涨幅>-10%到-5%",
    "涨幅小于-10%",
]


def _latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


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


def _review_generations(review_dir: Path) -> list[dict[str, Any]]:
    """Return every checksum-valid standalone review generation, newest first."""
    manifests: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for manifest in review_dir.glob("review_manifest_*.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifests.append((pd.Timestamp(payload["as_of"]).normalize(), manifest, payload))
        except (KeyError, OSError, TypeError, ValueError):
            continue

    generations = []
    for as_of, manifest, payload in sorted(manifests, key=lambda item: item[0], reverse=True):
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
                generations.append(
                    {
                        "as_of": as_of,
                        "manifest": manifest,
                        "review_files": resolved,
                        "source": "standalone",
                    }
                )
        except (KeyError, OSError, TypeError, ValueError):
            continue
    if manifests:
        return generations

    # Backward-compatible fallback for reports created before manifests existed.
    for summary in sorted(
        review_dir.glob("market_summary_*.json"), key=lambda path: path.stem, reverse=True
    ):
        stamp = summary.stem.removeprefix("market_summary_")
        try:
            as_of = pd.Timestamp(stamp).normalize()
        except ValueError:
            continue
        candidates = {
            "summary": summary,
            "csv": review_dir / f"market_breadth_{stamp}.csv",
            "png": review_dir / f"market_breadth_{stamp}.png",
        }
        generations.append(
            {
                "as_of": as_of,
                "manifest": None,
                "review_files": {name: path for name, path in candidates.items() if path.is_file()},
                "source": "legacy",
            }
        )
    return generations


def _latest_review_files(review_dir: Path) -> dict[str, Path]:
    generations = _review_generations(review_dir)
    return generations[0]["review_files"] if generations else {}


def _daily_generations(artifact_root: Path) -> list[dict[str, Any]]:
    """Return every checksum-valid daily review/signal generation, newest first."""
    candidates: list[tuple[pd.Timestamp, str, Path, dict[str, Any]]] = []
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

    generations = []
    for as_of, _, manifest, payload in sorted(
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
            generations.append(
                {
                    "as_of": as_of,
                    "manifest": manifest,
                    "review_files": review_files,
                    "signal_files": signal_files,
                    "source": "daily",
                }
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return generations


def _latest_daily_generation(artifact_root: Path) -> dict[str, Any] | None:
    generations = _daily_generations(artifact_root)
    return generations[0] if generations else None


def _review_history(
    artifact_root: Path,
    daily_generations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge daily and standalone generations into one generation per review date."""
    by_date = {
        generation["as_of"].date(): generation
        for generation in _review_generations(artifact_root / "reviews")
    }
    # A daily manifest is the atomic review+signals publication and wins for the
    # same date. Iterating oldest to newest keeps the newest generated duplicate.
    daily = _daily_generations(artifact_root) if daily_generations is None else daily_generations
    for generation in reversed(daily):
        by_date[generation["as_of"].date()] = generation
    return sorted(by_date.values(), key=lambda item: item["as_of"], reverse=True)


def _latest_signal_csv(strategy_dir: Path) -> Path | None:
    candidates = sorted(strategy_dir.glob("signal_*.csv"), key=lambda path: path.stem)
    return candidates[-1] if candidates else None


def _signal_date(path: Path | None) -> pd.Timestamp | None:
    if path is None:
        return None
    try:
        return pd.to_datetime(path.stem.removeprefix("signal_"), format="%Y%m%d").normalize()
    except ValueError:
        return None


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --qt-primary: #2563eb;
            --qt-primary-soft: #dbeafe;
            --qt-accent: #c2410c;
            --qt-bg: #f6f8fc;
            --qt-surface: #ffffff;
            --qt-text: #172033;
            --qt-muted: #526077;
            --qt-border: #dbe2ec;
            --qt-positive: #b42318;
            --qt-negative: #087a55;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 86% 2%, rgba(37, 99, 235, .10), transparent 25rem),
                var(--qt-bg);
            color: var(--qt-text);
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .qt-hero {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
            padding: 1.75rem 2rem;
            margin-bottom: 1.25rem;
            color: #ffffff;
            background: linear-gradient(120deg, #172554 0%, #1d4ed8 70%, #2563eb 100%);
            border-radius: 20px;
            box-shadow: 0 16px 40px rgba(30, 64, 175, .18);
        }
        .qt-eyebrow {
            margin-bottom: .45rem;
            color: #bfdbfe;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .12em;
        }
        .qt-hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.08;
            letter-spacing: -.04em;
        }
        .qt-hero p {
            max-width: 48rem;
            margin: .7rem 0 0;
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.65;
        }
        .qt-status {
            flex: 0 0 auto;
            padding: .7rem 1rem;
            color: #eff6ff;
            background: rgba(255, 255, 255, .12);
            border: 1px solid rgba(255, 255, 255, .22);
            border-radius: 999px;
            font-size: .86rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .qt-section {
            margin: 1.4rem 0 .7rem;
        }
        .qt-section h2 {
            margin: 0;
            color: var(--qt-text);
            font-size: 1.35rem;
            letter-spacing: -.02em;
        }
        .qt-section p {
            margin: .3rem 0 0;
            color: var(--qt-muted);
            font-size: .92rem;
        }
        div[data-testid="stMetric"] {
            min-height: 126px;
            padding: 1.05rem 1.1rem;
            background: var(--qt-surface);
            border: 1px solid var(--qt-border);
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, .04);
        }
        div[data-testid="stMetricLabel"] { color: var(--qt-muted); }
        div[data-testid="stMetricValue"] {
            color: var(--qt-text);
            font-variant-numeric: tabular-nums;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
            padding: .3rem;
            background: #e9eef7;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            min-height: 44px;
            padding: .55rem 1rem;
            border-radius: 9px;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(15, 23, 42, .08);
        }
        .stButton > button, .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            min-height: 44px;
            border-radius: 10px;
            font-weight: 650;
            transition: box-shadow 180ms ease, background 180ms ease, border-color 180ms ease;
        }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
        div[data-testid="stFormSubmitButton"] button:focus-visible {
            outline: 3px solid rgba(37, 99, 235, .30);
            outline-offset: 2px;
        }
        [data-testid="stForm"] {
            padding: 1.35rem;
            background: var(--qt-surface);
            border: 1px solid var(--qt-border);
            border-radius: 16px;
        }
        [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] {
            overflow: hidden;
            background: var(--qt-surface);
            border: 1px solid var(--qt-border);
            border-radius: 14px;
        }
        .qt-note-preview {
            min-height: 120px;
            padding: 1.1rem 1.2rem;
            background: #ffffff;
            border: 1px solid var(--qt-border);
            border-left: 4px solid var(--qt-primary);
            border-radius: 12px;
        }
        .qt-note-preview strong { color: var(--qt-text); }
        .qt-note-preview p { margin: .45rem 0 0; color: var(--qt-muted); line-height: 1.6; }
        @media (max-width: 720px) {
            .block-container { padding: 1rem .85rem 3rem; }
            .qt-hero { align-items: flex-start; flex-direction: column; padding: 1.35rem; }
            .qt-status { white-space: normal; }
            .stTabs [data-baseweb="tab"] { padding-inline: .65rem; }
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --qt-bg: #0b1220;
                --qt-surface: #111b2e;
                --qt-text: #f1f5f9;
                --qt-muted: #a9b6ca;
                --qt-border: #334155;
            }
            .stTabs [data-baseweb="tab-list"] { background: #172033; }
            .stTabs [aria-selected="true"] {
                background: #26344d;
                box-shadow: none;
            }
            .qt-note-preview { background: var(--qt-surface); }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                transition-duration: .01ms !important;
                animation-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="qt-section">
            <h2>{escape(title)}</h2>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _percentage(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "-"


def _read_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def _render_review(files: dict[str, Path], as_of: date) -> None:
    summary = _read_summary(files.get("summary"))
    matrix_path = files.get("daily_review") or files.get("csv")
    matrix = _read_csv(matrix_path)

    _section(
        f"{as_of:%Y年%m月%d日} 市场概览",
        "数据来自已发布且通过校验的复盘产物；红色代表上涨，绿色代表下跌。",
    )
    metrics = st.columns(5)
    values = [
        ("A股数量", summary.get("stocks", "-")),
        ("上涨家数", summary.get("up", "-")),
        ("下跌家数", summary.get("down", "-")),
        ("平均涨幅", _percentage(summary.get("mean_return"))),
        ("中位涨幅", _percentage(summary.get("median_return"))),
    ]
    for column, (label, value) in zip(metrics, values, strict=True):
        column.metric(label, value)

    if matrix.empty:
        image = files.get("png")
        if image:
            st.image(str(image), width="stretch")
        else:
            st.info("本期复盘只有摘要，没有可展示的明细表。")
        return

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("#### 当天涨跌分布")
        if {"名称", "当天"} <= set(matrix.columns):
            buckets = matrix[matrix["名称"].isin(RETURN_BUCKETS)][["名称", "当天"]].copy()
            buckets["比例"] = pd.to_numeric(
                buckets["当天"].astype(str).str.rstrip("%"), errors="coerce"
            )
            buckets = buckets.dropna(subset=["比例"]).rename(columns={"名称": "涨跌区间"})
            if not buckets.empty:
                st.bar_chart(
                    buckets,
                    x="比例",
                    y="涨跌区间",
                    horizontal=True,
                    color="#2563EB",
                    height=310,
                )
            else:
                st.caption("本期没有涨跌分布数据。")
        else:
            st.caption("旧版复盘表不包含当天分布字段。")

    with right:
        st.markdown("#### 当天关键读数")
        if {"名称", "当天"} <= set(matrix.columns):
            lookup = matrix.set_index("名称")["当天"]
            insight_rows = [
                ("上涨比例", lookup.get("A股上涨比例", "-")),
                ("下跌比例", lookup.get("A股下跌比例", "-")),
                ("涨幅中位数", lookup.get("A股涨幅中位数", "-")),
                ("可转债平均", lookup.get("可转债算术平均涨幅", "-")),
                ("实盘收益", lookup.get("我的实盘合计", "-")),
            ]
            insight = pd.DataFrame(insight_rows, columns=["指标", "数值"])
            st.dataframe(insight, width="stretch", hide_index=True, height=248)
        st.download_button(
            "下载本期复盘 CSV",
            data=matrix.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"daily_review_{as_of}.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("查看完整复盘矩阵", expanded=False):
        st.dataframe(matrix, width="stretch", hide_index=True, height=720)


def _note_defaults(note: dict[str, Any] | None) -> dict[str, Any]:
    market_state = (note or {}).get("market_state")
    return {
        "headline": (note or {}).get("headline", ""),
        "market_state": market_state if market_state in MARKET_STATES else "横盘整理",
        "sentiment_score": int((note or {}).get("sentiment_score", 3)),
        "discipline_score": int((note or {}).get("discipline_score", 3)),
        "position_pct": float((note or {}).get("position_pct", 0)) * 100,
        "portfolio_return": float((note or {}).get("portfolio_return", 0)) * 100,
        "market_observation": (note or {}).get("market_observation", ""),
        "trade_review": (note or {}).get("trade_review", ""),
        "lessons": (note or {}).get("lessons", ""),
        "next_plan": (note or {}).get("next_plan", ""),
        "tags": ", ".join((note or {}).get("tags", [])),
    }


def _render_note_editor(store: DataStore | None, default_date: date) -> None:
    if store is None:
        st.warning("数据库正在被其他任务占用，暂时无法读取或保存手动记录。请稍后重试。")
        return
    record_date = st.date_input(
        "记录日期",
        value=default_date,
        max_value=date.today(),
        help="默认使用当前选择的复盘日期，也可以补写任意历史日期。",
        key=f"review_note_date_{default_date}",
    )
    try:
        existing = store.review_note(record_date)
    except duckdb.Error as exc:
        st.error(f"读取复盘记录失败：{exc}")
        return
    defaults = _note_defaults(existing)
    if existing:
        st.caption(f"上次保存：{pd.Timestamp(existing['updated_at']):%Y-%m-%d %H:%M}")
    else:
        st.caption("该日期还没有手动记录，保存后可在记录档案中检索。")

    with st.form(f"review_note_form_{record_date}", clear_on_submit=False):
        st.markdown("#### 结论与状态")
        headline = st.text_input(
            "一句话结论",
            value=defaults["headline"],
            placeholder="例如：缩量震荡，防守优先，等待方向确认",
            help="用一句话记录今天最重要的市场判断。",
        )
        state_col, sentiment_col, discipline_col = st.columns(3)
        market_state = state_col.selectbox(
            "市场状态",
            MARKET_STATES,
            index=MARKET_STATES.index(defaults["market_state"]),
        )
        sentiment_score = sentiment_col.slider(
            "市场体感",
            min_value=1,
            max_value=5,
            value=defaults["sentiment_score"],
            help="1 表示极弱，5 表示极强。",
        )
        discipline_score = discipline_col.slider(
            "执行纪律",
            min_value=1,
            max_value=5,
            value=defaults["discipline_score"],
            help="评价自己是否按计划执行，而不是评价盈亏。",
        )
        position_col, return_col = st.columns(2)
        position_pct = position_col.number_input(
            "收盘仓位（%）",
            min_value=0.0,
            max_value=100.0,
            value=defaults["position_pct"],
            step=5.0,
        )
        portfolio_return = return_col.number_input(
            "实盘当天收益（%）",
            value=defaults["portfolio_return"],
            step=0.1,
            format="%.2f",
        )

        st.markdown("#### 观察与复盘")
        market_observation = st.text_area(
            "市场观察",
            value=defaults["market_observation"],
            placeholder="指数、量能、风格、涨跌结构和重要事件发生了什么？",
            height=130,
        )
        trade_review = st.text_area(
            "交易回顾",
            value=defaults["trade_review"],
            placeholder="今天做了什么，哪些交易符合计划，哪些偏离计划？",
            height=130,
        )
        lesson_col, plan_col = st.columns(2)
        lessons = lesson_col.text_area(
            "经验与错误",
            value=defaults["lessons"],
            placeholder="今天最值得保留或纠正的一点。",
            height=150,
        )
        next_plan = plan_col.text_area(
            "下一交易日计划",
            value=defaults["next_plan"],
            placeholder="触发条件、观察重点、仓位边界和风险预案。",
            height=150,
        )
        tags = st.text_input(
            "标签",
            value=defaults["tags"],
            placeholder="缩量, 防守, 轮动",
            help="使用逗号分隔，便于后续检索。",
        )
        submitted = st.form_submit_button("保存复盘记录", type="primary", width="stretch")

    if submitted:
        try:
            store.save_review_note(
                record_date,
                {
                    "headline": headline,
                    "market_state": market_state,
                    "sentiment_score": sentiment_score,
                    "discipline_score": discipline_score,
                    "position_pct": position_pct / 100,
                    "portfolio_return": portfolio_return / 100,
                    "market_observation": market_observation,
                    "trade_review": trade_review,
                    "lessons": lessons,
                    "next_plan": next_plan,
                    "tags": tags,
                },
            )
            st.session_state["review_note_saved"] = str(record_date)
            st.rerun()
        except (ValueError, duckdb.Error) as exc:
            st.error(f"保存失败：{exc}")


def _render_note_archive(store: DataStore | None) -> None:
    if store is None:
        st.warning("数据库正在被其他任务占用，暂时无法读取记录档案。")
        return
    try:
        notes = store.list_review_notes()
    except duckdb.Error as exc:
        st.error(f"读取记录档案失败：{exc}")
        return
    if notes.empty:
        st.info("还没有手动复盘记录。先在“手动记录”中完成第一篇复盘。")
        return

    display = notes.copy()
    display["仓位"] = display["position_pct"].map(lambda value: f"{value:.0%}")
    display["实盘当天"] = display["portfolio_return"].map(lambda value: f"{value:.2%}")

    def display_tags(value: Any) -> str:
        try:
            return " · ".join(json.loads(value or "[]"))
        except (TypeError, ValueError):
            return ""

    display["标签"] = display["tags"].map(display_tags)
    display = display.rename(
        columns={
            "trade_date": "日期",
            "headline": "一句话结论",
            "market_state": "市场状态",
            "sentiment_score": "市场体感",
            "discipline_score": "执行纪律",
            "updated_at": "更新时间",
        }
    )
    columns = [
        "日期",
        "一句话结论",
        "市场状态",
        "市场体感",
        "执行纪律",
        "仓位",
        "实盘当天",
        "标签",
        "更新时间",
    ]
    st.dataframe(display[columns], width="stretch", hide_index=True, height=360)
    st.download_button(
        "导出全部记录 CSV",
        data=display[columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="review_notes.csv",
        mime="text/csv",
    )

    selected_date = st.selectbox(
        "查看记录详情",
        options=notes["trade_date"].dt.date.tolist(),
        format_func=lambda value: value.strftime("%Y年%m月%d日"),
    )
    note = store.review_note(selected_date)
    if note is None:
        return
    st.markdown(
        f"""
        <div class="qt-note-preview">
            <strong>{escape(note["headline"] or "未填写一句话结论")}</strong>
            <p>{escape(note["market_state"])} · 市场体感 {note["sentiment_score"]}/5 ·
            执行纪律 {note["discipline_score"]}/5 · 仓位 {note["position_pct"]:.0%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    detail_columns = st.columns(2)
    sections = [
        ("市场观察", note["market_observation"]),
        ("交易回顾", note["trade_review"]),
        ("经验与错误", note["lessons"]),
        ("下一交易日计划", note["next_plan"]),
    ]
    for index, (label, content) in enumerate(sections):
        with detail_columns[index % 2]:
            st.markdown(f"#### {label}")
            st.write(content or "未填写")


def _render_review_center(
    store: DataStore | None,
    history: list[dict[str, Any]],
) -> None:
    selected_generation = None
    selected_date = date.today()
    if history:
        selected_date = st.selectbox(
            "历史复盘日期",
            options=[generation["as_of"].date() for generation in history],
            format_func=lambda value: value.strftime("%Y年%m月%d日"),
            help="只展示已完整发布且校验通过的历史复盘。",
        )
        selected_generation = next(
            generation for generation in history if generation["as_of"].date() == selected_date
        )

    view = st.segmented_control(
        "复盘视图",
        ["市场数据", "手动记录", "记录档案"],
        default="市场数据" if history else "手动记录",
        width="stretch",
        label_visibility="collapsed",
    )
    if view == "市场数据":
        if selected_generation is None:
            st.info("尚无复盘结果。先运行 `qt daily run`，或切换到“手动记录”。")
        else:
            _render_review(selected_generation["review_files"], selected_date)
    elif view == "手动记录":
        _section(
            "手动复盘记录",
            "把行情事实、主观判断和次日计划分开记录，避免只用盈亏评价交易质量。",
        )
        _render_note_editor(store, selected_date)
    else:
        _section("记录档案", "按日期回看判断与执行，让复盘形成可检索的决策日志。")
        _render_note_archive(store)


def _render_signals(cfg: AppConfig, daily_generation: dict[str, Any] | None) -> None:
    daily_signals = daily_generation["signal_files"] if daily_generation is not None else {}
    strategy_names = {
        path.name for path in (cfg.paths.artifacts_dir / "signals").glob("*") if path.is_dir()
    } | set(daily_signals)
    if not strategy_names:
        st.info("尚无策略信号。运行启用策略的 `qt daily run` 后可在这里查看。")
        return
    strategy_name = st.selectbox("策略", sorted(strategy_names))
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
    _section(strategy_name, "显示该策略最近一次完整生成的目标权重与信号。")
    if latest:
        signal = _read_csv(latest)
        st.dataframe(signal, width="stretch", hide_index=True)
        st.download_button(
            "下载策略信号 CSV",
            data=signal.to_csv(index=False).encode("utf-8-sig"),
            file_name=latest.name,
            mime="text/csv",
        )


def _render_backtests(cfg: AppConfig) -> None:
    strategy_dirs = sorted(
        path for path in (cfg.paths.artifacts_dir / "backtests").glob("*") if path.is_dir()
    )
    if not strategy_dirs:
        st.info("尚无回测结果。")
        return
    strategy_dir = st.selectbox(
        "回测策略",
        strategy_dirs,
        format_func=lambda path: path.name,
    )
    _section(strategy_dir.name, "查看最近一次回测的交互报告、净值与关键指标。")
    reports = list(strategy_dir.glob("*/report.html"))
    report = _latest(reports) or strategy_dir / "report.html"
    if report.exists():
        components.html(report.read_text(encoding="utf-8"), height=1100, scrolling=True)
        return
    run_dirs = [path for path in strategy_dir.glob("*") if path.is_dir()]
    result_dir = _latest(run_dirs) or strategy_dir
    if (result_dir / "metrics.json").exists():
        st.json(json.loads((result_dir / "metrics.json").read_text()))
    if (result_dir / "equity.png").exists():
        st.image(str(result_dir / "equity.png"), width="stretch")


def _render_data_status(cfg: AppConfig) -> None:
    if not cfg.paths.database.exists():
        st.info("数据库尚未创建。")
        return
    try:
        with duckdb.connect(str(cfg.paths.database), read_only=True) as con:
            latest_run = con.execute(
                "SELECT status, finished_at FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            fetch_count = con.execute(
                "SELECT COUNT(*) FROM data_fetches WHERE status = 'failed'"
            ).fetchone()[0]
            note_count = con.execute("SELECT COUNT(*) FROM review_notes").fetchone()[0]
            status_columns = st.columns(3)
            status_columns[0].metric("最近任务状态", latest_run[0] if latest_run else "暂无")
            status_columns[1].metric("失败请求累计", int(fetch_count))
            status_columns[2].metric("手动复盘记录", int(note_count))

            tables = [
                (
                    "最近运行",
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT 30",
                ),
                (
                    "数据源请求",
                    "SELECT * FROM data_fetches ORDER BY fetched_at DESC LIMIT 50",
                ),
                (
                    "分钟文件导入",
                    """
                    SELECT * FROM minute_archive_imports
                    ORDER BY imported_at DESC LIMIT 50
                    """,
                ),
                (
                    "分钟目录导入",
                    "SELECT * FROM minute_import_runs ORDER BY started_at DESC LIMIT 30",
                ),
                (
                    "分钟数据覆盖",
                    """
                    SELECT frequency, asset_type, COUNT(DISTINCT symbol) AS symbols,
                           SUM(rows) AS rows, MIN(min_time) AS min_time,
                           MAX(max_time) AS max_time
                    FROM minute_partitions
                    GROUP BY frequency, asset_type ORDER BY frequency, asset_type
                    """,
                ),
            ]
            for title, query in tables:
                with st.expander(title, expanded=title == "最近运行"):
                    st.dataframe(con.execute(query).df(), width="stretch", hide_index=True)
    except duckdb.Error as exc:
        st.warning("数据正在更新或数据库暂时不可读，请稍后刷新。")
        st.caption(str(exc))


def main() -> None:
    st.set_page_config(
        page_title="Quant Trade 复盘中心",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    cfg = load_config()
    _inject_styles()
    try:
        store: DataStore | None = DataStore(cfg)
    except duckdb.Error:
        store = None
    daily_generations = _daily_generations(cfg.paths.artifacts_dir)
    history = _review_history(cfg.paths.artifacts_dir, daily_generations)
    daily_generation = daily_generations[0] if daily_generations else None
    latest_date = history[0]["as_of"].date() if history else None
    latest_label = latest_date.strftime("%Y-%m-%d") if latest_date else "等待首次复盘"

    st.markdown(
        f"""
        <div class="qt-hero">
            <div>
                <div class="qt-eyebrow">MARKET REVIEW WORKSPACE</div>
                <h1>Quant Trade 复盘中心</h1>
                <p>把市场数据、策略信号与主观交易记录放在同一条时间线上，
                支持回看历史，也支持持续修正自己的决策过程。</p>
            </div>
            <div class="qt-status">最新复盘&nbsp;&nbsp;{latest_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    saved_date = st.session_state.pop("review_note_saved", None)
    if saved_date:
        st.toast(f"{saved_date} 的复盘记录已保存")

    tabs = st.tabs(["复盘中心", "策略信号", "回测研究", "数据状态"])
    with tabs[0]:
        _render_review_center(store, history)
    with tabs[1]:
        _render_signals(cfg, daily_generation)
    with tabs[2]:
        _render_backtests(cfg)
    with tabs[3]:
        _render_data_status(cfg)


if __name__ == "__main__":
    main()
