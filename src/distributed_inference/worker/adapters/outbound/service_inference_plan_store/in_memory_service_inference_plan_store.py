from typing import override

from distributed_inference.domain.plan import InferencePlanVersion, ServiceInferencePlan
from distributed_inference.worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)


class InMemoryServiceInferencePlanStore(ServiceInferencePlanStore):
    def __init__(self) -> None:
        self._latest_inference_plan: ServiceInferencePlan | None = None
        self._active_inference_plan: ServiceInferencePlan | None = None
        self._inference_plan_by_version: dict[
            InferencePlanVersion, ServiceInferencePlan
        ] = {}

    @override
    async def put_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:

        if service_inference_plan.plan_version in self._inference_plan_by_version:
            return

        if self._latest_inference_plan is None:
            self._latest_inference_plan = service_inference_plan
        else:
            if (
                service_inference_plan.plan_version
                > self._latest_inference_plan.plan_version
            ):
                self._latest_inference_plan = service_inference_plan

        self._inference_plan_by_version[service_inference_plan.plan_version] = (
            service_inference_plan
        )

    @override
    async def get_service_inference_plan_by_version(
        self, version: InferencePlanVersion
    ) -> ServiceInferencePlan | None:
        return self._inference_plan_by_version.get(version, None)

    @override
    async def get_latest_service_inference_plan(self) -> ServiceInferencePlan | None:
        return self._latest_inference_plan

    @override
    async def get_active_service_inference_plan(self) -> ServiceInferencePlan | None:
        return self._active_inference_plan

    @override
    async def activate_service_inference_plan(
        self, version: InferencePlanVersion
    ) -> None:
        if version not in self._inference_plan_by_version:
            raise ValueError("Inference plan does not exist")
        self._active_inference_plan = self._inference_plan_by_version[version]
