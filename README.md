# Quant Trade

统一的 A 股收盘复盘、数据管理、策略研究和回测平台。旧脚本暂时保留在
`market_breadth/` 与 `etf/`，新功能通过 `qt` 命令运行。

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
qt review close
qt strategy signal etf_rotation
qt strategy backtest etf_rotation --start 2020-01-01
qt daily run
qt dashboard
```

所有产物写入 `artifacts/`，运行状态写入 `runs/` 和 DuckDB。分钟 ZIP 放入
`data/inbox/minute/` 后运行导入命令；成功文件归档，失败文件进入隔离目录。

