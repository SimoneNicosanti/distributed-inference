from abc import ABC, abstractmethod

from worker.domain.model.model_pass import ModelPassError


class FailurePropagator(ABC):
    @abstractmethod
    async def propagate_failure(self, error: ModelPassError) -> None: ...
