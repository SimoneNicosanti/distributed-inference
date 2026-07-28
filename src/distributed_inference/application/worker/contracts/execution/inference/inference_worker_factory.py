from abc import ABC, abstractmethod

from distributed_inference.application.worker.contracts.execution.inference.inference_worker import (
    InferenceWorker,
)


class InferenceWorkerDeploymentOptions:
    pass


class InferenceWorkerFactory(ABC):
    @abstractmethod
    async def create_inference_worker(
        self,
        deployment: InferenceWorkerDeploymentOptions,
    ) -> InferenceWorker: ...
