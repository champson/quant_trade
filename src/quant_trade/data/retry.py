from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import TypeVar

from quant_trade.config import RetryConfig
from quant_trade.data.base import PermanentProviderError, TransientProviderError

T = TypeVar("T")


TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "reset",
    "temporarily",
    "频率",
    "每分钟",
    "limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "拒绝连接",
)
PERMANENT_MARKERS = (
    "token",
    "权限",
    "积分",
    "invalid parameter",
    "非法参数",
    "ip数量超限",
)


def classify_provider_exception(exc: Exception) -> Exception:
    message = str(exc).lower()
    if any(marker in message for marker in PERMANENT_MARKERS):
        return PermanentProviderError(str(exc))
    if any(marker in message for marker in TRANSIENT_MARKERS):
        return TransientProviderError(str(exc))
    return TransientProviderError(str(exc))


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_seconds: int):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._states: dict[str, CircuitState] = {}
        self._lock = Lock()

    def allow(self, provider: str) -> bool:
        with self._lock:
            state = self._states.setdefault(provider, CircuitState())
            if state.opened_at is None:
                return True
            if time.monotonic() - state.opened_at >= self.cooldown_seconds:
                state.failures = 0
                state.opened_at = None
                return True
            return False

    def success(self, provider: str) -> None:
        with self._lock:
            self._states[provider] = CircuitState()

    def failure(self, provider: str) -> None:
        with self._lock:
            state = self._states.setdefault(provider, CircuitState())
            state.failures += 1
            if state.failures >= self.threshold:
                state.opened_at = time.monotonic()


def retry_call(
    operation: Callable[[], T],
    config: RetryConfig,
    on_retry: Callable[[int, float, Exception], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last: Exception | None = None
    for attempt in range(1, config.attempts + 1):
        try:
            return operation()
        except PermanentProviderError:
            raise
        except Exception as raw:
            exc = raw if isinstance(raw, TransientProviderError) else classify_provider_exception(raw)
            if isinstance(exc, PermanentProviderError):
                raise exc from raw
            last = exc
            if attempt >= config.attempts:
                break
            base = config.delays[min(attempt - 1, len(config.delays) - 1)]
            delay = base * random.uniform(0.8, 1.2)
            if on_retry:
                on_retry(attempt, delay, exc)
            sleep(delay)
    assert last is not None
    raise last

