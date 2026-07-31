from __future__ import annotations


def is_b_share_symbol(symbol: object) -> bool:
    """Return whether a Tushare-style symbol is a Shanghai/Shenzhen B share."""
    value = str(symbol).strip().upper()
    code, separator, exchange = value.partition(".")
    if not separator:
        return False
    return (exchange == "SZ" and code.startswith("2")) or (
        exchange == "SH" and code.startswith("9")
    )
