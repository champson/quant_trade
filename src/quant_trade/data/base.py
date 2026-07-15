from __future__ import annotations

from abc import ABC, abstractmethod

from quant_trade.models import DataBatch, DataRequest, Dataset


class ProviderError(RuntimeError):
    """Base provider error."""


class TransientProviderError(ProviderError):
    """An operation that can be retried."""


class PermanentProviderError(ProviderError):
    """Authentication, permission, validation, or unsupported request failure."""


class EmptyDataError(ProviderError):
    """The provider returned no rows for an otherwise valid request."""


class DataProvider(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> set[Dataset]:
        raise NotImplementedError

    def supports(self, request: DataRequest) -> bool:
        return request.dataset in self.capabilities()

    @abstractmethod
    def fetch(self, request: DataRequest) -> DataBatch:
        raise NotImplementedError

    def close(self) -> None:
        return None

