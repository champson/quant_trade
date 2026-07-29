from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from quant_trade.reports.market_review import MarketReview, build_daily_review_table


CHINESE_FONT_CANDIDATES = (
    "Hiragino Sans GB",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
)


def _chinese_font_properties():
    from matplotlib import font_manager

    installed = {font.name: font.fname for font in font_manager.fontManager.ttflist}
    for name in CHINESE_FONT_CANDIDATES:
        if path := installed.get(name):
            return font_manager.FontProperties(fname=path)
    return font_manager.FontProperties()


def save_market_review(
    review: MarketReview,
    out_dir: Path,
    *,
    index_returns: pd.DataFrame | None = None,
    index_bias: pd.Series | None = None,
    portfolio: pd.Series | None = None,
    convertible_summary: pd.DataFrame | None = None,
    underlying_summary: pd.DataFrame | None = None,
    microcap_returns: pd.Series | None = None,
    bias: pd.DataFrame | None = None,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chinese_font = _chinese_font_properties()
    matplotlib.rcParams["axes.unicode_minus"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = review.as_of.strftime("%Y-%m-%d")
    names = {
        "csv": f"market_breadth_{stamp}.csv",
        "png": f"market_breadth_{stamp}.png",
        "summary": f"market_summary_{stamp}.json",
        "daily_review": f"daily_review_{stamp}.csv",
    }
    extras = {
        "indices": index_returns,
        "portfolio": portfolio,
        "convertible_bonds": convertible_summary,
        "logbias": bias,
    }
    for name, value in extras.items():
        if value is not None and not value.empty:
            names[name] = f"{name}_{stamp}.csv"

    staging = out_dir / ".staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    manifest_name = f"review_manifest_{stamp}.json"
    try:
        daily_review = build_daily_review_table(
            review,
            index_returns=index_returns,
            index_bias=index_bias,
            convertible_summary=convertible_summary,
            underlying_summary=underlying_summary,
            microcap_returns=microcap_returns,
            portfolio_returns=portfolio,
        )
        review.breadth.to_csv(staging / names["csv"], index=False, encoding="utf-8-sig")
        daily_review.to_csv(staging / names["daily_review"], index=False, encoding="utf-8-sig")
        pd.Series(review.summary).to_json(staging / names["summary"], force_ascii=False, indent=2)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        try:
            ax.axis("off")
            table = ax.table(
                cellText=review.breadth.values,
                colLabels=review.breadth.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            for cell in table.get_celld().values():
                cell.get_text().set_fontproperties(chinese_font)
            ax.set_title(f"A股市场宽度 {stamp}", fontproperties=chinese_font)
            fig.tight_layout()
            fig.savefig(staging / names["png"], dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
        for name, value in extras.items():
            if name in names:
                value.to_csv(staging / names[name], encoding="utf-8-sig")

        checksums = {
            name: hashlib.sha256((staging / filename).read_bytes()).hexdigest()
            for name, filename in names.items()
        }
        manifest = {
            "as_of": stamp,
            "files": names,
            "sha256": checksums,
        }
        (staging / manifest_name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # A manifest is published last, so readers never select a partially
        # rendered generation. Individual replacements are atomic on one volume.
        for name in names.values():
            (staging / name).replace(out_dir / name)
        for name in extras:
            if name not in names:
                (out_dir / f"{name}_{stamp}.csv").unlink(missing_ok=True)
        (staging / manifest_name).replace(out_dir / manifest_name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {name: out_dir / filename for name, filename in names.items()}
