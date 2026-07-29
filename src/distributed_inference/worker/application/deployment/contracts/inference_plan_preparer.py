from abc import ABC, abstractmethod

from distributed_inference.domain.plan import ServiceInferencePlan


class InferencePlanPreparer(ABC):
    @abstractmethod
    async def prepare_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None: ...
