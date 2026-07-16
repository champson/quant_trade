# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A-share (Chinese stock market) close-of-day review, data management, strategy research, and backtesting platform. Python 3.11+, `src/` layout, single Typer CLI entry point `qt`. The authoritative architecture document is `docs/design.md` (Chinese); contributor conventions are in `AGENTS.md`; the backtest-report field specification is `report.md` at the repo root.

## Commands

```bash
pip install -e '.[dev]'                    # setup; then cp .env.example .env and set TUSHARE_TOKEN
pytest -q                                  # full test suite (offline — no tokens or network)
pytest tests/test_backtest_report.py -q    # single test file
ruff check src tests                       # lint
ruff format src tests                      # format (line length 100)
```

Common CLI: `qt data update`, `qt data market-history`, `qt data minute import-directory|import-inbox|verify`, `qt review close`, `qt strategy signal|backtest <name>`, `qt daily run`, `qt dashboard`.

## Architecture

Layering (enforced — see docs/design.md §2): CLI/Dashboard → `pipelines/` (orchestration) → `services.py` (use cases) → `strategies/`, `backtest/`, `reports/`, `data/`.

- **data/**: `router.py` routes requests across `providers/` in configured priority (tushare → baostock → akshare) with retry, per-provider circuit breaker, and fallback; every fetch is logged to the DuckDB `data_fetches` table. `storage.py` (`DataStore`) is Parquet files plus a DuckDB catalog (`data/quant_trade.duckdb`):
  - Daily bars: `data/processed/daily/{asset_type}/{symbol}.parquet` (dots in symbols become underscores on disk).
  - Minute bars: `data/processed/minute/frequency=5min/symbol=000001_SZ/year=2025.parquet` (ZSTD). The DuckDB `minute_partitions` table is the catalog `read_minute` queries — Parquet files and catalog must stay in sync (`qt data minute verify` checks this).
  - Minute data comes only from file imports (`minute_archive.py` for ZIP inbox, `minute_directory.py` for read-only directory import with hash/mtime resume and atomic per-symbol commit); providers only serve daily frequency.
- **strategies/**: subclass `Strategy` (`base.py`), implement `generate_targets(bars) → DataFrame` of target weights indexed by signal date. Register by editing the hardcoded `REGISTRY` dict in `registry.py` — there is no auto-discovery. Strategies must not fetch data, manage caches, or render output.
- **backtest/**: `engine.py` `run_weight_backtest` — long-only, weights sum ≤ 1, models commission, sell-side stamp duty, and slippage. `report.py` writes `artifacts/backtests/<name>/` (self-contained `report.html`, archivable `report.md`, charts, CSVs, `metrics.json`); HTML and Markdown must come from the same computed results, never calculated separately.
- **pipelines/daily.py**: `qt daily run` = trade-calendar check → market snapshots at anchor dates (prev day / week / month / year ends) → index and strategy-symbol updates → minute inbox import → market review → per-strategy signals → run record (`runs/*.json` + DuckDB `runs` table).
- **Config**: pydantic models in `config.py`; YAML resolved from `--config` → `QT_CONFIG` env → `configs/default.yaml`. Secrets only via `.env` (`Secrets` BaseSettings): `TUSHARE_TOKEN`, `TUSHARE_HTTP_URL`.

## Critical invariants

- **No look-ahead**: close-generated targets execute at the *next* available bar's open, never same-day. Tested in `tests/test_backtest_strategies.py`; do not weaken.
- **Report credibility** (docs/design.md §5): reports state only what the engine actually models. Unmodeled effects (limit-up/down and halts, T+1, 100-share lots, announcement timing, survivorship bias) must stay explicitly labeled as such — never fake them. Trades are unpaired order flow, so per-trade win rate / profit-loss ratio / holding period must not be reported as automated metrics.
- **Adjustment (复权) contract**: daily bars are stored and read per adjustment mode (`adjustment` column; `read_daily(..., adjustment=)`). Tushare does not serve adjusted bars — the router falls back to a provider that does. Never mix adjustment modes in one series.
- Microcap strategy requires point-in-time `total_mv` (previous day via `.shift(1)`).
- Minute import: source files are read-only; 5-min data may aggregate upward but must never fabricate 1-min bars; ETF `09:31` placeholder rows are filtered; stock 09:30 auction bars kept with `is_auction=true`.

## Conventions

- Symbols: `{6 digits}.{SH|SZ|BJ}` (e.g. `600000.SH`). Providers convert to their own dialects internally.
- Bilingual code: identifiers, comments, and docstrings in English; user-facing strings (CLI output, report labels, exception messages) in Chinese. Match this split.
- Tests are offline: fake `DataProvider` subclasses plus the `tmp_path`-rooted `app_config` fixture (`tests/conftest.py`). No tokens, network, or user data directories.
- `data/`, `artifacts/`, `runs/` are runtime output — never commit them.
- Timezone is implicitly Asia/Shanghai; datetimes are naive throughout.
