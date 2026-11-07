from abc import ABC, abstractmethod

from worker.domain.model.model_pass import ModelPassError


class FailureReceiver(ABC):
    @abstractmethod
    async def notify_failure(self, error: ModelPassError) -> None: ...
