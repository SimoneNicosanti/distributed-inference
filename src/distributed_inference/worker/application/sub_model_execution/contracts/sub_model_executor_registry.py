from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from distributed_inference.domain.plan import SubModelDeployment
from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)


class SubModelExecutorRegistry(ABC):
    @abstractmethod
    async def register_sub_model_executor(
        self,
        sub_model_deployment: SubModelDeployment,
        sub_model_executor: SubModelExecutor,
    ) -> None: ...

    @abstractmethod
    async def unregister_sub_model_executor(
        self,
        sub_model_deployment: SubModelDeployment,
    ) -> SubModelExecutor: ...

    @abstractmethod
    def acquire_sub_model_executor(
        self, sub_model_deployment: SubModelDeployment
    ) -> AbstractAsyncContextManager[SubModelExecutor]: ...

    @abstractmethod
    async def check_sub_model_executor(
        self, sub_model_deployment: SubModelDeployment
    ) -> bool: ...
