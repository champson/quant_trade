from __future__ import annotations

from typing import Any

from quant_trade.strategies.base import Strategy
from quant_trade.strategies.etf_rotation import EtfRotationStrategy
from quant_trade.strategies.logbias import LogBiasStrategy
from quant_trade.strategies.microcap import MicrocapStrategy


REGISTRY: dict[str, type[Strategy]] = {
    "etf_rotation": EtfRotationStrategy,
    "logbias": LogBiasStrategy,
    "microcap": MicrocapStrategy,
}


def get_strategy(name: str, config: dict[str, Any] | None = None) -> Strategy:
    try:
        return REGISTRY[name](config)
    except KeyError as exc:
        raise KeyError(f"未知策略 {name}；可用: {', '.join(REGISTRY)}") from exc


def strategy_names() -> list[str]:
    return sorted(REGISTRY)
