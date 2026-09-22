from abc import ABC, abstractmethod

from integration.plan.plan import ServiceInferencePlan


class ServiceInferencePlanApplier(ABC):
    @abstractmethod
    async def apply_service_inference_plan(
        self,
        service_inference_plan: ServiceInferencePlan,
    ) -> None: ...
