from __future__ import annotations

from dataclasses import replace

from quant_trade.config import AppConfig
from quant_trade.data.base import DataProvider, PermanentProviderError, ProviderError
from quant_trade.data.quality import validate_bars
from quant_trade.data.retry import CircuitBreaker, retry_call
from quant_trade.data.storage import DataStore
from quant_trade.models import DataBatch, DataRequest, Dataset


class DataRouter:
    def __init__(
        self,
        config: AppConfig,
        providers: dict[str, DataProvider],
        store: DataStore | None = None,
    ):
        self.config = config
        self.providers = providers
        self.store = store
        retry = config.providers.retry
        self.circuit = CircuitBreaker(retry.circuit_failures, retry.circuit_cooldown_seconds)

    def _candidates(self, request: DataRequest) -> list[str]:
        if request.provider != "auto":
            return [request.provider]
        if not self.config.providers.allow_fallback:
            return self.config.providers.priority[:1]
        return self.config.providers.priority

    def fetch(self, request: DataRequest) -> DataBatch:
        errors: list[str] = []
        for name in self._candidates(request):
            provider = self.providers.get(name)
            if provider is None or not provider.supports(request):
                errors.append(f"{name}: 不支持该数据集")
                continue
            if not self.circuit.allow(name):
                errors.append(f"{name}: 熔断中")
                continue
            try:
                batch = retry_call(
                    lambda: provider.fetch(replace(request, provider=name)),
                    self.config.providers.retry,
                )
                if request.dataset == Dataset.BARS:
                    batch.warnings.extend(validate_bars(batch.data))
                self.circuit.success(name)
                if errors:
                    batch.warnings.append("数据源回退: " + " | ".join(errors))
                if self.store:
                    self.store.record_fetch(
                        dataset=request.dataset.value,
                        provider=name,
                        symbols=",".join(request.symbols),
                        start=str(request.start), end=str(request.end), rows=len(batch.data),
                        status="success", warnings=batch.warnings,
                    )
                return batch
            except PermanentProviderError as exc:
                errors.append(f"{name}: {exc}")
            except Exception as exc:
                self.circuit.failure(name)
                errors.append(f"{name}: {exc}")
        if self.store:
            self.store.record_fetch(
                dataset=request.dataset.value, provider="none",
                symbols=",".join(request.symbols), start=str(request.start), end=str(request.end),
                rows=0, status="failed", warnings=errors,
            )
        raise ProviderError("所有数据源均失败: " + " | ".join(errors))

    def close(self) -> None:
        for provider in self.providers.values():
            provider.close()


def build_router(config: AppConfig, store: DataStore | None = None) -> DataRouter:
    from quant_trade.config import Secrets
    from quant_trade.data.providers import AkShareProvider, BaoStockProvider, TushareProvider

    secrets = Secrets()
    providers = {
        "tushare": TushareProvider(
            float(config.providers.tushare.get("request_interval_seconds", 0.5)), secrets
        ),
        "baostock": BaoStockProvider(
            float(config.providers.baostock.get("request_interval_seconds", 0.25))
        ),
        "akshare": AkShareProvider(
            float(config.providers.akshare.get("request_interval_seconds", 1.0))
        ),
    }
    return DataRouter(config, providers, store)

