from abc import ABC, abstractmethod

from distributed_inference.domain.plan import ServiceInferencePlan


class ServiceInferencePlanDeployer(ABC):
    @abstractmethod
    async def deploy_service_inference_plan(
        self,
        service_inference_plan: ServiceInferencePlan,
    ) -> None: ...
