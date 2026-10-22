from typing import override

import httpx

from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from controller.application.ports.outbound.plan_share_gateway import PlanShareGateway
from controller.domain.plan.service_plan import ServicePlan
from integration.service_instance.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from integration.worker.deployment.dto.service_plan_dto import ServicePlanDto
from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareServicePlanRequest,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class HttpPlanShareGateway(PlanShareGateway):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        worker_directory_gateway: WorkerDirectoryGateway,
    ) -> None:
        self._http_client = http_client
        self._worker_directory_gateway = worker_directory_gateway

    @override
    async def send_prepare_plan_message(
        self,
        worker_id: WorkerId,
        plan: ServicePlan,
    ) -> None:
        base_url = await self._get_plan_actuation_base_url(worker_id)
        plan_version = plan.worker_plan.plan_version
        plan_dto = ServicePlanDto.model_validate(plan.model_dump())
        request = PrepareServicePlanRequest(service_plan=plan_dto)

        response = await self._http_client.put(
            self._build_url(
                base_url,
                PREPARE_SERVICE_PLAN_PATH,
                plan_version,
            ),
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

    @override
    async def send_apply_plan_message(
        self,
        worker_id: WorkerId,
        plan_version: PlanVersion,
    ) -> None:
        base_url = await self._get_plan_actuation_base_url(worker_id)
        response = await self._http_client.put(
            self._build_url(
                base_url,
                APPLY_SERVICE_PLAN_PATH,
                plan_version,
            )
        )
        response.raise_for_status()

    async def _get_plan_actuation_base_url(self, worker_id: WorkerId) -> str:
        worker_instance = (
            await self._worker_directory_gateway.get_worker_instance_by_id(worker_id)
        )
        interface = self._get_plan_actuation_http_interface(worker_instance)
        return str(interface.endpoint).rstrip("/")

    @staticmethod
    def _get_plan_actuation_http_interface(
        worker: ServiceInstance,
    ) -> CapabilityInterface:
        interfaces = worker.get_capability_interfaces_by_type_and_protocol(
            CapabilityType.PLAN_ACTUATION,
            CapabilityProtocol.HTTP,
        )
        if not interfaces:
            raise ValueError(
                f"Worker {worker.service_id} does not expose an HTTP plan interface"
            )
        if len(interfaces) > 1:
            raise ValueError(
                f"Worker {worker.service_id} exposes multiple HTTP plan interfaces"
            )
        return interfaces[0]

    @staticmethod
    def _build_url(
        base_url: str,
        path: str,
        plan_version: PlanVersion,
    ) -> str:
        return f"{base_url}{path.format(version_number=plan_version.version_number)}"
