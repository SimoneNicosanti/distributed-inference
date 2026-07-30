from abc import ABC, abstractmethod

from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)


class SubModelExecutorDeploymentOptions:
    pass


class SubModelExecutorFactory(ABC):
    @abstractmethod
    async def create_sub_model_executor(
        self,
        deployment: SubModelExecutorDeploymentOptions,
    ) -> SubModelExecutor: ...
