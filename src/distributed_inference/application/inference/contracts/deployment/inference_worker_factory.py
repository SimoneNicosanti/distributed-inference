from abc import ABC, abstractmethod
from typing import Any

from distributed_inference.application.inference.contracts.execution.worker.inference_worker import (
    InferenceWorker,
)


class InferenceWorkerFactory(ABC):
    @abstractmethod
    async def create(
        self,
        deployment: Any,
    ) -> InferenceWorker: ...
