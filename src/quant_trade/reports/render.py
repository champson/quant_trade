from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from quant_trade.reports.market_review import MarketReview


def save_market_review(
    review: MarketReview,
    out_dir: Path,
    *,
    index_returns: pd.DataFrame | None = None,
    portfolio: pd.Series | None = None,
    convertible_summary: pd.DataFrame | None = None,
    bias: pd.DataFrame | None = None,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = review.as_of.strftime("%Y-%m-%d")
    csv_path = out_dir / f"market_breadth_{stamp}.csv"
    png_path = out_dir / f"market_breadth_{stamp}.png"
    summary_path = out_dir / f"market_summary_{stamp}.json"
    review.breadth.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.Series(review.summary).to_json(summary_path, force_ascii=False, indent=2)
    fig, ax = plt.subplots(figsize=(10, 4.8))
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
    ax.set_title(f"A股市场宽度 {stamp}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Glyph .* missing from font")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outputs = {"csv": csv_path, "png": png_path, "summary": summary_path}
    extras = {
        "indices": index_returns,
        "portfolio": portfolio,
        "convertible_bonds": convertible_summary,
        "logbias": bias,
    }
    for name, value in extras.items():
        if value is not None and not value.empty:
            path = out_dir / f"{name}_{stamp}.csv"
            value.to_csv(path, encoding="utf-8-sig")
            outputs[name] = path
    return outputs
