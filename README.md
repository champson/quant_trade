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
qt data verify --asset-type stock --start 2025-01-01
qt data minute import-inbox --frequency 5min
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
`data/inbox/minute/` 后按真实频率运行导入命令；成功文件归档，失败文件进入隔离目录。
ZIP 会拒绝坏行和与声明频率不符的时间点，先完整写入 staging，再与同年已有数据做无冲突
增量合并，最后在跨进程写锁内原子提交到统一的 symbol/year 分区；进程中断后会按提交日志
自动恢复。若启动报告分钟提交缺少 journal/恢复目录，请先停止所有 `qt` 进程，检查
`data/processed/.staging/minute-commit/<commit_id>/` 与 DuckDB `minute_commit_log`，从备份恢复
或确认正式分区后再手工清理；不要直接删除仍处于 `preparing` 的记录。
`qt daily run` 自动处理 inbox 时使用 `minute.inbox_frequency` 和
`minute.inbox_asset_type`；请按下载包实际口径配置，命令行参数可临时覆盖。任一 ZIP 导入
失败时命令返回非零状态，每日流水线默认失败退出；仅在明确接受降级运行时将
`minute.fail_daily_on_import_error` 设为 `false`。

日线缓存按 `资产类型/复权方式` 隔离。旧版 `data/processed/daily/<资产类型>/*.parquet`
不会自动迁移，因为其中可能混有不同复权价格；升级后首次执行数据更新会重新下载到
`adjustment=none|qfq|hfq/` 子目录，确认新缓存完整后可自行归档旧文件。
前复权的历史尺度会随锚定日变化，因此 `qfq` 更新会扩展到该证券全部已有缓存日期，完整
重取并原子替换；若响应漏掉任一原缓存日期则保留旧文件并报错。默认后复权 `hfq` 仍按
缺口增量更新。
交易日历和明确的空行情日期缓存在 DuckDB，完整缓存更新可离线运行；部分成功响应遗漏的
日期仍会继续补取。今天及未来的交易日历默认每24小时刷新；行情命令拒绝未来 `--end`。
全市场 `data update` 只支持单日且仅支持 `adjustment=none`，历史回填使用
`qt data market-history`。快照行数、证券成员、参考阈值和完整性状态写入 DuckDB；成员
Parquet 丢失、缺少目标交易日或响应未达到阈值时不会命中断点缓存。写入完成标记时对所有
成员保存文件指纹，并按 `market_snapshot_validation_sample_size` 确定性抽检内容；之后的
断点检查只比较指纹，并按 `market_snapshot_cache_ttl_seconds` 短暂记忆结果，不会反复打开
约 4500 个 Parquet，也不会永久掩盖其他进程改写。周期性运行
`qt data verify` 可全量扫描全部成员的键、OHLCV，并同时审计股票 `daily_basic` 的日期、
重复项和正市值；失败快照默认撤销完整标记。`market-history` 按
`providers.market_history_batch_days` 合并多日后再写盘，避免逐日重写每个证券的全部历史。
强制刷新返回残缺数据时，
既有完整标记和成员清单会保留。早期历史只使用经 `stock_basic` 独立规模参考验证过的
当日 `daily_basic`；`--no-basic` 时全市场响应本身也携带该独立参考，因此不会套用当前
4500 只股票的下限。两个同日响应不能互相证明完整。微盘股信号和回测要求请求区间的
交易日历、每个交易日的全市场快照和 daily_basic 全部完整，不会因某天完全没有行情文件
而漏检；复盘目标日及日/周/月/年收益锚点也必须是完整快照，否则不会生成报告。

每次回测会在 `artifacts/backtests/<策略名>/<运行编号>/` 同时生成自包含的
`report.html`、可归档的 `report.md`、净值与月度图表，以及权益、持仓、成交和指标明细，
不会覆盖此前结果。`qt dashboard` 自动选择最新运行，并兼容旧版直接位于策略目录的报告。

## 5分钟目录导入

目录导入只读取源文件，不移动或修改原始CSV。存在 `manifest.csv` 时使用其中的
`category` 保留股票、ETF和指数类型；没有 manifest、也没有分类子目录时，会按证券代码
推断资产类型（例如 `16xxxx` 为 ETF、`93xxxx` 为指数）。输出目录不按资产类型分类：

```text
data/processed/minute/
  frequency=5min/
    symbol=000001_SZ/
      year=2025.parquet
```

输出使用ZSTD压缩。导入状态、文件哈希、过滤数量、年份、行数、时间覆盖和分区路径记录在
DuckDB；只有源文件未变化且登记的数据仍完整存在时才会跳过。`--force` 可强制重建。
空文件、仅表头文件和过滤后为空的源文件不会清除已经导入的证券分区。
同一目录中一个证券只能对应一个源文件；若历史按年份拆成多个 CSV，应先合并，避免
不同文件对同一证券的“完整镜像”语义冲突。
分钟分区目录、目录表和成功来源记录在同一个 DuckDB/文件提交事务中完成；任一元数据写入
失败都会恢复原分区，不会留下可被断点续传误认的成功记录。
ZIP 预检失败也会记录失败状态并移动到隔离目录；成功归档的每个 symbol/year 成员都绑定
统一目录记录和目标 Parquet 的 size/mtime 指纹。后续 ZIP 增量合并会原子刷新所有相关
归档的指纹，因此历史包仍可断点跳过；目录项丢失或文件被外部改写时则会重新导入。

导入会过滤ETF的 `09:31 + OHLC全为1 + 零成交` 占位行，保留股票09:30集合竞价并设置
`is_auction=true`。`minute.timestamp_convention` 支持 `source`、`bar_end` 和 `bar_start`；
`bar_start` 会按频率平移为统一的结束时间。`qt data minute verify` 会深检目录元数据、键、
时间范围和 OHLCV，而不只检查文件存在。5分钟数据可以向上聚合，不能用来生成真实1分钟行情。
