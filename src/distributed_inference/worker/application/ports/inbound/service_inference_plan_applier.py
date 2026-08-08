from abc import ABC, abstractmethod

from distributed_inference.domain.plan import ServiceInferencePlan


class ServiceInferencePlanApplier(ABC):
    @abstractmethod
    async def apply_service_inference_plan(
        self,
        service_inference_plan: ServiceInferencePlan,
    ) -> None: ...
