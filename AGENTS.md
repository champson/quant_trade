# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/quant_trade/`. Keep commands in `cli.py`, orchestration in `services.py` and `pipelines/`, and domain logic in focused packages:

- `data/`: providers, retry/routing, imports, quality checks, and Parquet/DuckDB storage.
- `strategies/`: signal and target-weight generation; strategies must not download data or render reports.
- `backtest/`: execution simulation, metrics, and backtest reports.
- `reports/`: market-review calculations and rendering.
- `dashboard/`: the Streamlit user interface.

Tests are in `tests/`. Configuration belongs in `configs/default.yaml`; architecture decisions are in `docs/design.md`. Runtime data, reports, and task state belong in `data/`, `artifacts/`, and `runs/` and must not be committed.

## Build, Test, and Development Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'       # editable install with test/lint tools
ruff check src tests          # lint Python code
ruff format --check src tests # verify formatting
pytest -q                     # run the full test suite
pytest --cov=quant_trade      # run tests with coverage
qt --help                     # inspect CLI commands
qt dashboard                  # start the local Streamlit UI
```

Run targeted tests while iterating: `pytest tests/test_backtest_report.py -q`.

## Coding Style & Naming Conventions

Use Python 3.11+ with four-space indentation, type hints, and a 100-character line limit. Ruff is the formatter and linter. Use `snake_case` for modules, functions, variables, and test names; `PascalCase` for classes; and uppercase names for constants. Prefer small, pure calculation functions and keep provider-specific schemas behind the data layer. Preserve the rule that close-generated signals execute no earlier than the next available bar.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and functions named `test_*`. Add regression tests for every bug fix and unit tests for new strategy, importer, storage, or report behavior. Use temporary paths and offline fixtures; tests must not require API tokens, network access, or the user’s market-data directories. Before submitting, run linting and the full suite.

## Commit & Pull Request Guidelines

The repository has minimal history, so no detailed convention is established. Use short imperative subjects such as `Add self-contained backtest report`; keep unrelated changes separate. Pull requests should explain purpose, behavior changes, verification commands, configuration or schema impacts, and linked issues. Include screenshots for dashboard/report UI changes and call out migrations or large-data implications.

## Security & Configuration

Copy `.env.example` to `.env` and keep tokens out of source, logs, fixtures, and screenshots. Never commit DuckDB files, downloaded CSV/ZIP data, Parquet partitions, or generated artifacts.
