from abc import ABC, abstractmethod

from distributed_inference.worker.application.ports.outbound.inference_worker import (
    InferenceWorker,
)


class InferenceWorkerRegistry(ABC):
    @abstractmethod
    async def register_inference_worker(
        self, inference_worker: InferenceWorker
    ) -> None: ...

    @abstractmethod
    async def get_inference_worker(
        self, inference_worker_id: str
    ) -> InferenceWorker: ...
