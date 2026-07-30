from abc import ABC, abstractmethod

from distributed_inference.domain.plan import InferencePlanVersion, ServiceInferencePlan


class ServiceInferencePlanStore(ABC):
    @abstractmethod
    async def put_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None: ...

    @abstractmethod
    async def get_service_inference_plan_by_version(
        self, version: InferencePlanVersion
    ) -> ServiceInferencePlan | None: ...

    @abstractmethod
    async def get_latest_service_inference_plan(
        self,
    ) -> ServiceInferencePlan | None: ...

    @abstractmethod
    async def get_active_service_inference_plan(
        self,
    ) -> ServiceInferencePlan | None: ...

    @abstractmethod
    async def activate_service_inference_plan(
        self, version: InferencePlanVersion
    ) -> None: ...
