from typing import override

import httpx
import pytest
from pydantic import HttpUrl

from controller.adapters.outbound.plan_share.http_plan_share_gateway import (
    HttpPlanShareGateway,
)
from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from controller.domain.plan.deployment_plan import DeploymentPlan
from controller.domain.plan.failure_plan import FailurePlan
from controller.domain.plan.hop_routing_plan import HopRoutingPlan
from controller.domain.plan.partition_execution_plan import PartitionExecutionPlan
from controller.domain.plan.scheduling_plan import SchedulingPlan
from controller.domain.plan.worker_plan import WorkerPlan
from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
)
from integration.service_instance.worker_instance import WorkerInstance
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareWorkerPlanRequest,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


class StubServiceDirectory(WorkerDirectoryGateway):
    def __init__(self, service_instance: WorkerInstance) -> None:
        self._service_instance = service_instance

    @override
    async def get_registered_worker_ids(self) -> set[ServiceId]:
        return {self._service_instance.service_id}

    @override
    async def get_worker_instance_by_id(
        self,
        service_id: ServiceId,
    ) -> WorkerInstance:
        if service_id != self._service_instance.service_id:
            raise KeyError(service_id)
        return self._service_instance


def _worker_plan(plan_version: PlanVersion) -> WorkerPlan:
    return WorkerPlan(
        plan_version=plan_version,
        deployment_plan=DeploymentPlan(
            executable_hops=(), allocations=(), ingress_hops=(), sink_hops=()
        ),
        scheduling_plan=SchedulingPlan(priority_keys=(), priorities=()),
        partition_execution_plan=PartitionExecutionPlan(partition_ids=(), schemes=()),
        hop_routing_plan=HopRoutingPlan(hop_addresses=(), routings=()),
        failure_plan=FailurePlan(flow_ids=(), sink_replica_ids=()),
    )


def _worker_instance(
    worker_id: ServiceId,
    interfaces: list[CapabilityInterface],
) -> WorkerInstance:
    capabilities = [
        Capability(type=capability_type, interfaces=[])
        for capability_type in (
            CapabilityType.HOP_INVOCATION,
            CapabilityType.IN_NETWORK_PROFILING,
            CapabilityType.RESULT_RETRIEVE,
            CapabilityType.FAILURE_RECEIVE,
        )
    ]
    capabilities.append(
        Capability(
            type=CapabilityType.PLAN_ACTUATION,
            interfaces=interfaces,
        )
    )
    return WorkerInstance(
        service_id=worker_id,
        capabilities=capabilities,
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
        plan = _worker_plan(plan_version)

        await gateway.send_prepare_plan_message(worker_id, plan)
        await gateway.send_apply_plan_message(worker_id, plan_version)

    assert [request.method for request in requests] == ["PUT", "PUT"]
    assert [request.url.path for request in requests] == [
        "/worker/deployment/plans/3",
        "/worker/deployment/active-plan/3",
    ]
    request = PrepareWorkerPlanRequest.model_validate_json(requests[0].content)
    assert request.worker_plan.plan_version == plan_version
    assert set(request.model_dump()) == {"worker_plan"}
    assert request.worker_plan.deployment_plan.executable_hops == ()
    assert request.worker_plan.failure_plan.flow_ids == ()
    assert request.worker_plan.failure_plan.sink_replica_ids == ()


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
