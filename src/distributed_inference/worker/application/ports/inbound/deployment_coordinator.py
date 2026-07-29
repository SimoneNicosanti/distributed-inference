from abc import ABC, abstractmethod

from distributed_inference.domain.plan import ServiceInferencePlan


class DeploymentCoordinator(ABC):
    @abstractmethod
    async def deploy_plan(
        self,
        service_plan: ServiceInferencePlan,
    ) -> None: ...
