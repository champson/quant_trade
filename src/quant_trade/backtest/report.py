from __future__ import annotations

import base64
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_trade.backtest.engine import BacktestResult, ExecutionConfig
from quant_trade.backtest.metrics import performance_metrics


@dataclass(frozen=True)
class BacktestReportPaths:
    markdown: Path
    html: Path
    equity_chart: Path
    monthly_chart: Path
    equity: Path
    positions: Path
    trades: Path
    metrics: Path

    def as_dict(self) -> dict[str, Path]:
        return asdict(self)


METRIC_LABELS = {
    "total_return": "累计收益率",
    "cagr": "年化收益率",
    "annual_volatility": "年化波动率",
    "sharpe": "Sharpe 比率",
    "max_drawdown": "最大回撤",
    "calmar": "Calmar 比率",
}


def _returns_by_period(equity: pd.Series, rule: str) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    sampled = equity.sort_index().resample(rule).last()
    first = pd.Series([equity.iloc[0]], index=[equity.index.min() - pd.Timedelta(days=1)])
    return pd.concat([first, sampled]).pct_change().iloc[1:]


def _percent(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2%}"


def _number(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _metric_value(key: str, value: float | None) -> str:
    return (
        _percent(value)
        if key in {"total_return", "cagr", "annual_volatility", "max_drawdown"}
        else _number(value)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _trade_summary(result: BacktestResult) -> dict[str, float | int]:
    trades = result.trades
    if trades.empty:
        return {
            "orders": 0,
            "buys": 0,
            "sells": 0,
            "notional": 0.0,
            "commission": 0.0,
            "tax": 0.0,
            "cost": 0.0,
            "turnover": 0.0,
        }
    commission = float(trades.get("commission", pd.Series(dtype=float)).sum())
    tax = float(trades.get("tax", pd.Series(dtype=float)).sum())
    notional = float(trades.get("notional", pd.Series(dtype=float)).sum())
    average_equity = float(result.equity.mean()) if not result.equity.empty else 0.0
    return {
        "orders": len(trades),
        "buys": int((trades["side"] == "BUY").sum()),
        "sells": int((trades["side"] == "SELL").sum()),
        "notional": notional,
        "commission": commission,
        "tax": tax,
        "cost": commission + tax,
        "turnover": notional / average_equity if average_equity else 0.0,
    }


def _make_charts(
    result: BacktestResult,
    out_dir: Path,
    benchmark_equity: pd.Series | None,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity_path = out_dir / "equity.png"
    monthly_path = out_dir / "monthly_returns.png"
    nav = result.equity / result.equity.iloc[0]
    drawdown = nav / nav.cummax() - 1

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    top.plot(nav.index, nav, label="Strategy", linewidth=1.8)
    if benchmark_equity is not None and not benchmark_equity.empty:
        benchmark_nav = benchmark_equity / benchmark_equity.iloc[0]
        top.plot(benchmark_nav.index, benchmark_nav, label="Benchmark", linewidth=1.3)
    top.set_ylabel("NAV")
    top.grid(alpha=0.25)
    top.legend()
    bottom.fill_between(drawdown.index, drawdown.values, 0, color="#d95f59", alpha=0.55)
    bottom.set_ylabel("Drawdown")
    bottom.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(equity_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    monthly = _returns_by_period(result.equity, "ME")
    colors = ["#c94c4c" if value < 0 else "#2f8f68" for value in monthly]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(monthly.index.strftime("%Y-%m"), monthly.values, color=colors)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_ylabel("Monthly return")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(monthly_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return equity_path, monthly_path


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _image_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_backtest_report(
    *,
    name: str,
    result: BacktestResult,
    out_dir: Path,
    execution: ExecutionConfig,
    strategy_config: dict[str, Any] | None = None,
    benchmark_equity: pd.Series | None = None,
    benchmark_name: str | None = None,
) -> BacktestReportPaths:
    """Write raw results plus human-readable Markdown and self-contained HTML."""
    if result.equity.empty:
        raise ValueError("回测净值为空，无法生成报告")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.equity.index = pd.to_datetime(result.equity.index)
    if benchmark_equity is not None:
        benchmark_equity = benchmark_equity.dropna().sort_index()
        benchmark_equity.index = pd.to_datetime(benchmark_equity.index)

    equity_path = out_dir / "equity.csv"
    positions_path = out_dir / "positions.csv"
    trades_path = out_dir / "trades.csv"
    metrics_path = out_dir / "metrics.json"
    markdown_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    result.equity.to_csv(equity_path, header=True)
    result.positions.to_csv(positions_path)
    result.trades.to_csv(trades_path, index=False)

    benchmark_metrics = (
        performance_metrics(benchmark_equity, risk_free=execution.risk_free_annual)
        if benchmark_equity is not None and len(benchmark_equity) > 1
        else {}
    )
    metrics_payload = {
        "strategy": result.metrics,
        "benchmark": benchmark_metrics,
        "benchmark_name": benchmark_name,
    }
    metrics_path.write_text(
        json.dumps(_json_safe(metrics_payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    equity_chart, monthly_chart = _make_charts(result, out_dir, benchmark_equity)

    metric_rows = [
        [
            METRIC_LABELS[key],
            _metric_value(key, result.metrics.get(key)),
            _metric_value(key, benchmark_metrics.get(key)),
        ]
        for key in METRIC_LABELS
    ]
    annual = _returns_by_period(result.equity, "YE")
    benchmark_annual = (
        _returns_by_period(benchmark_equity, "YE")
        if benchmark_equity is not None
        else pd.Series(dtype=float)
    )
    annual_rows = []
    for index, value in annual.items():
        benchmark_value = benchmark_annual.get(index)
        excess = value - benchmark_value if benchmark_value is not None else None
        annual_rows.append(
            [str(index.year), _percent(value), _percent(benchmark_value), _percent(excess)]
        )
    monthly = _returns_by_period(result.equity, "ME")
    positive_months = int((monthly > 0).sum())
    negative_months = int((monthly < 0).sum())
    monthly_rows = [
        ["盈利月份", str(positive_months)],
        ["亏损月份", str(negative_months)],
        ["月度胜率", _percent(positive_months / len(monthly) if len(monthly) else None)],
        ["平均月收益", _percent(float(monthly.mean()) if len(monthly) else None)],
        ["最好月份", _percent(float(monthly.max()) if len(monthly) else None)],
        ["最差月份", _percent(float(monthly.min()) if len(monthly) else None)],
    ]
    trade = _trade_summary(result)
    trade_rows = [
        ["成交订单数", f"{trade['orders']:,}"],
        ["买入 / 卖出", f"{trade['buys']:,} / {trade['sells']:,}"],
        ["成交金额", f"{trade['notional']:,.2f}"],
        ["佣金", f"{trade['commission']:,.2f}"],
        ["印花税", f"{trade['tax']:,.2f}"],
        ["总交易成本", f"{trade['cost']:,.2f}"],
        ["区间双边换手", _percent(float(trade["turnover"]))],
    ]
    credibility_rows = [
        ["未来函数", "已控制：收盘信号在下一可用开盘价执行"],
        ["手续费、印花税、滑点", "已建模"],
        ["涨跌停、停牌", "未建模"],
        ["T+1", "未建模"],
        ["整数手", f"已建模：每笔成交按 lot_size={execution.lot_size} 向下取整"],
        ["财务公告时点", "由策略输入数据负责；引擎未独立校验"],
        ["退市与生存者偏差", "取决于股票池数据；引擎未独立校验"],
    ]
    start, end = result.equity.index.min(), result.equity.index.max()
    benchmark_label = benchmark_name or "未提供"
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    config_text = json.dumps(strategy_config or {}, ensure_ascii=False, indent=2, default=str)

    markdown = f"""# {name} 回测报告

> 自动生成时间：{generated_at}  
> 本报告用于研究记录，不构成投资建议。

## 一、策略与回测设置

| 项目 | 内容 |
| --- | --- |
| 回测区间 | {start.date()} 至 {end.date()} |
| 初始资金 | {execution.initial_cash:,.2f} |
| 期末权益 | {result.equity.iloc[-1]:,.2f} |
| 比较基准 | {benchmark_label} |
| 佣金率 | {execution.commission_rate:.4%} |
| 卖出印花税率 | {execution.stamp_duty_rate:.4%} |
| 滑点率 | {execution.slippage_rate:.4%} |
| 每手数量 | {execution.lot_size} |

策略配置：

```json
{config_text}
```

## 二、核心绩效

{_markdown_table(["指标", "策略", "基准"], metric_rows)}

![净值与回撤](equity.png)

## 三、年度表现

{_markdown_table(["年份", "策略收益", "基准收益", "超额收益"], annual_rows) if annual_rows else "数据不足。"}

## 四、月度表现

{_markdown_table(["指标", "数值"], monthly_rows)}

![月度收益](monthly_returns.png)

## 五、交易与成本

{_markdown_table(["指标", "数值"], trade_rows)}

当前成交表记录的是订单成交，不是配对后的完整开平仓，因此暂不报告单笔胜率、盈亏比和平均持有期。

## 六、回测可信度

{_markdown_table(["检查项", "处理状态"], credibility_rows)}

## 七、结论与后续验证

- 当前自动报告覆盖收益、风险、年度/月度稳定性、订单与成本。
- 在涨跌停、停牌、T+1和股票池历史完整性建模前，不应直接将结果解释为可实盘收益。
- 鲁棒性、样本外、参数扰动和盈利集中度需要作为独立实验运行，不能由单次回测自动推断。

## 八、明细文件

- [净值序列](equity.csv)
- [每日持仓](positions.csv)
- [成交记录](trades.csv)
- [机器可读指标](metrics.json)
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    summary_cards = "".join(
        f'<div class="card"><span>{html.escape(METRIC_LABELS[key])}</span><strong>{html.escape(_metric_value(key, result.metrics.get(key)))}</strong></div>'
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "calmar")
    )
    html_document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(name)} 回测报告</title>
<style>
:root{{--ink:#17202a;--muted:#65727e;--line:#dfe5ea;--paper:#fff;--bg:#f4f6f8;--accent:#245b85}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:28px auto;padding:32px 42px;background:var(--paper);box-shadow:0 8px 30px #16202a12}}
h1{{margin:0 0 4px;font-size:30px}} h2{{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:8px;font-size:20px}}
.muted{{color:var(--muted)}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.card{{border:1px solid var(--line);border-radius:8px;padding:14px;background:#fafcfd}} .card span{{display:block;color:var(--muted);font-size:13px}} .card strong{{display:block;font-size:23px;margin-top:5px}}
table{{width:100%;border-collapse:collapse;margin:14px 0 22px}} th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}} th{{background:#f2f5f7}}
img{{display:block;width:100%;height:auto;margin:18px 0}} pre{{background:#f5f7f8;padding:14px;overflow:auto;border-radius:6px}}
.warning{{border-left:4px solid #d29b35;background:#fff8e8;padding:12px 15px}} a{{color:var(--accent)}}
@media(max-width:700px){{main{{margin:0;padding:22px 16px}}}}
</style></head><body><main>
<h1>{html.escape(name)} 回测报告</h1><div class="muted">{start.date()} 至 {end.date()} · 自动生成于 {html.escape(generated_at)}</div>
<div class="cards">{summary_cards}</div>
<h2>策略与回测设置</h2>
{_html_table(["项目", "内容"], [["初始资金", f"{execution.initial_cash:,.2f}"], ["期末权益", f"{result.equity.iloc[-1]:,.2f}"], ["比较基准", benchmark_label], ["佣金 / 印花税 / 滑点", f"{execution.commission_rate:.4%} / {execution.stamp_duty_rate:.4%} / {execution.slippage_rate:.4%}"], ["每手数量", str(execution.lot_size)]])}
<details><summary>策略配置</summary><pre>{html.escape(config_text)}</pre></details>
<h2>净值与回撤</h2><img alt="净值与回撤" src="{_image_data(equity_chart)}">
<h2>核心绩效</h2>{_html_table(["指标", "策略", "基准"], metric_rows)}
<h2>年度表现</h2>{_html_table(["年份", "策略收益", "基准收益", "超额收益"], annual_rows) if annual_rows else "<p>数据不足。</p>"}
<h2>月度表现</h2>{_html_table(["指标", "数值"], monthly_rows)}<img alt="月度收益" src="{_image_data(monthly_chart)}">
<h2>交易与成本</h2>{_html_table(["指标", "数值"], trade_rows)}<p class="muted">成交表尚未配对为完整开平仓，因此不展示单笔胜率、盈亏比和平均持有期。</p>
<h2>回测可信度</h2>{_html_table(["检查项", "处理状态"], credibility_rows)}
<div class="warning">涨跌停、停牌、T+1及历史股票池完整性尚未由引擎完整建模，本报告不能直接视为可实盘收益证明。</div>
<h2>原始结果</h2><ul><li><a href="equity.csv">净值序列</a></li><li><a href="positions.csv">每日持仓</a></li><li><a href="trades.csv">成交记录</a></li><li><a href="metrics.json">机器可读指标</a></li><li><a href="report.md">Markdown 报告</a></li></ul>
</main></body></html>"""
    html_path.write_text(html_document, encoding="utf-8")
    return BacktestReportPaths(
        markdown=markdown_path,
        html=html_path,
        equity_chart=equity_chart,
        monthly_chart=monthly_chart,
        equity=equity_path,
        positions=positions_path,
        trades=trades_path,
        metrics=metrics_path,
    )
