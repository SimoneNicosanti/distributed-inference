from typing import override

import httpx
import pytest
from pydantic import HttpUrl

from controller.adapters.outbound.plan_share.http_plan_share_gateway import (
    HttpPlanShareGateway,
)
from controller.domain.plan.deployment_plan import DeploymentPlan
from controller.domain.plan.partition_execution_plan import PartitionExecutionPlan
from controller.domain.plan.routing_plan import RoutingPlan
from controller.domain.plan.scheduling_plan import SchedulingPlan
from controller.domain.plan.service_plan import ServicePlan
from controller.domain.plan.worker_plan import WorkerPlan
from directory.api.directory import ServiceDirectory
from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareServicePlanRequest,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


class StubServiceDirectory(ServiceDirectory):
    def __init__(self, service_instance: ServiceInstance) -> None:
        self._service_instance = service_instance

    @override
    async def get_service_instance_by_service_id(
        self,
        service_id: ServiceId,
    ) -> ServiceInstance:
        if service_id != self._service_instance.service_id:
            raise KeyError(service_id)
        return self._service_instance

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        return [self._service_instance]

    @override
    async def put_service_instance(self, service_instance: ServiceInstance) -> None:
        self._service_instance = service_instance


def _service_plan(plan_version: PlanVersion) -> ServicePlan:
    return ServicePlan(
        worker_plan=WorkerPlan(
            plan_version=plan_version,
            deployment_plan=DeploymentPlan(replica_ids=(), allocations=()),
            scheduling_plan=SchedulingPlan(priority_keys=(), priorities=()),
            partition_execution_plan=PartitionExecutionPlan(
                partition_ids=(),
                schemes=(),
            ),
            routing_plan=RoutingPlan(replica_ids=(), routings=()),
        ),
        ingress_plan=None,
    )


def _worker_instance(
    worker_id: ServiceId,
    interfaces: list[CapabilityInterface],
) -> ServiceInstance:
    return ServiceInstance(
        service_id=worker_id,
        capabilities=(
            [
                Capability(
                    type=CapabilityType.PLAN_ACTUATION,
                    interfaces=interfaces,
                )
            ]
            if interfaces
            else []
        ),
    )


@pytest.mark.asyncio
async def test_sends_prepare_and_apply_to_discovered_worker_endpoint() -> None:
    worker_id = ServiceId()
    directory = StubServiceDirectory(
        _worker_instance(
            worker_id,
            [
                CapabilityInterface(
                    protocol=CapabilityProtocol.HTTP,
                    endpoint=HttpUrl("http://worker.local:9000/"),
                )
            ],
        )
    )
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=204)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as http_client:
        gateway = HttpPlanShareGateway(http_client, directory)
        plan_version = PlanVersion(version_number=3)
        plan = _service_plan(plan_version)

        await gateway.send_prepare_plan_message(worker_id, plan)
        await gateway.send_apply_plan_message(worker_id, plan_version)

    assert [request.method for request in requests] == ["PUT", "PUT"]
    assert [request.url.path for request in requests] == [
        "/worker/deployment/plans/3",
        "/worker/deployment/active-plan/3",
    ]
    request = PrepareServicePlanRequest.model_validate_json(requests[0].content)
    assert request.service_plan.worker_plan.plan_version == plan_version
    assert request.service_plan.ingress_plan is None


@pytest.mark.asyncio
async def test_rejects_worker_without_http_plan_interface() -> None:
    worker_id = ServiceId()
    directory = StubServiceDirectory(_worker_instance(worker_id, []))

    async with httpx.AsyncClient() as http_client:
        gateway = HttpPlanShareGateway(http_client, directory)

        with pytest.raises(ValueError, match="does not expose an HTTP plan interface"):
            await gateway.send_apply_plan_message(
                worker_id,
                PlanVersion(version_number=1),
            )
