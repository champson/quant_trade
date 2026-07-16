# Quant Trade

统一的 A 股收盘复盘、数据管理、策略研究和回测平台。所有功能通过 `qt` 命令运行，
架构和产物规范见 [`docs/design.md`](docs/design.md)。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

将 Tushare token 写入 `.env`，不要写入源码。当前旧脚本曾包含明文 token，使用前应在
Tushare 后台更换该 token。

## 常用命令

```bash
qt data update --symbols 000001.SZ,600000.SH --start 2025-01-01
qt data minute import-inbox
qt data minute inspect-directory ~/Documents/分钟K线/tushare_5min
qt data minute import-directory ~/Documents/分钟K线/tushare_5min --frequency 5min
qt data minute verify --frequency 5min
qt review close
qt strategy signal etf_rotation
qt strategy backtest etf_rotation --start 2020-01-01
qt daily run
qt dashboard
```

所有产物写入 `artifacts/`，运行状态写入 `runs/` 和 DuckDB。分钟 ZIP 放入
`data/inbox/minute/` 后运行导入命令；成功文件归档，失败文件进入隔离目录。

回测会在 `artifacts/backtests/<策略名>/` 同时生成自包含的 `report.html`、可归档的
`report.md`、净值与月度图表，以及权益、持仓、成交和指标明细。`qt dashboard` 会把完整
HTML 报告直接集成到回测页面。

## 5分钟目录导入

目录导入只读取源文件，不移动或修改原始CSV。存在 `manifest.csv` 时使用其中的
`category` 保留股票、ETF和指数类型；输出目录不按资产类型分类：

```text
data/processed/minute/
  frequency=5min/
    symbol=000001_SZ/
      year=2025.parquet
```

输出使用ZSTD压缩。导入状态、文件哈希、过滤数量和分区路径记录在DuckDB；再次运行时，
文件大小和修改时间未变化的证券会直接跳过。`--force` 可强制重建。

导入会过滤ETF的 `09:31 + OHLC全为1 + 零成交` 占位行，保留股票09:30集合竞价并设置
`is_auction=true`。5分钟数据可以向上聚合，不能用来生成真实1分钟行情。
