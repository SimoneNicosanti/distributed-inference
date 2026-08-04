from abc import ABC, abstractmethod

from distributed_inference.domain.identifiers import SubModelDeploymentId
from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)


class SubModelExecutorRegistry(ABC):
    @abstractmethod
    async def register_sub_model_executor(
        self,
        sub_model_deployment_id: SubModelDeploymentId,
        sub_model_executor: SubModelExecutor,
    ) -> None: ...

    @abstractmethod
    async def unregister_sub_model_executor(
        self, sub_model_deployment_id: SubModelDeploymentId
    ) -> None: ...

    ## TODO: Here we need to discriminate the multiple possible deployments of the same sub-model
    ## TODO: WE NEED TO USE A SUB-MODEL-DEPLOYMENT-ID IN ORDER TO MANAGER MULTIPLE REPLICAS OF THE SAME SUB-MODEL
    @abstractmethod
    async def get_sub_model_executor(
        self, sub_model_deployment_id: SubModelDeploymentId
    ) -> SubModelExecutor: ...
