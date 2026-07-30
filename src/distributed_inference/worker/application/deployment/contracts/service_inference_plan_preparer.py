from abc import ABC, abstractmethod

from distributed_inference.domain.plan import ServiceInferencePlan


class ServiceInferencePlanPreparer(ABC):
    @abstractmethod
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None: ...
