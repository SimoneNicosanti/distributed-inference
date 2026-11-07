from abc import ABC, abstractmethod

from shared.service.service import WorkerId
from worker.domain.model.model_pass import ModelPassError


class FailureSendError(Exception):
    pass


class FailureSender(ABC):
    @abstractmethod
    async def send_failure(
        self, error: ModelPassError, worker_id: WorkerId
    ) -> None: ...
