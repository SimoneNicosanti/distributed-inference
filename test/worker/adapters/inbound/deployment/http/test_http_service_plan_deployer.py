from typing import override

from fastapi import FastAPI
from fastapi.testclient import TestClient

from integration.dto.context.context_dto import (
    ExternalHopDto,
    IngressHopDto,
    SinkHopDto,
)
from integration.dto.worker_plan.worker_plan_dto import (
    DeploymentPlanDto,
    DestinationHopAddressDto,
    HopInputRoutingDto,
    HopOutputRoutingDto,
    HopRoutingPlanDto,
    HopRoutingsDto,
    InvocableHopAddressDto,
    PartitionExecutionPlanDto,
    SchedulingPlanDto,
    SourceHopAddressDto,
    WorkerPlanDto,
)
from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.inbound.deployment.http.http_service_plan_deployer import (
    build_worker_plan_deployment_router,
)
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.domain.hop.hop import ExternalHop, IngressHop, SinkHop
from worker.domain.plan.worker_plan import WorkerPlan


class RecordingPlanDeployer(PlanDeployer[WorkerPlan]):
    def __init__(self) -> None:
        self.prepared: list[WorkerPlan] = []
        self.applied: list[PlanVersion] = []

    @override
    async def prepare_plan(self, plan: WorkerPlan) -> None:
        self.prepared.append(plan)

    @override
    async def apply_plan(self, plan_version: PlanVersion) -> None:
        self.applied.append(plan_version)


def _empty_worker_plan(version_number: int) -> WorkerPlanDto:
    return WorkerPlanDto(
        plan_version=PlanVersion(version_number=version_number),
        deployment_plan=DeploymentPlanDto(replica_ids=(), allocations=()),
        scheduling_plan=SchedulingPlanDto(priority_keys=(), priorities=()),
        partition_execution_plan=PartitionExecutionPlanDto(
            partition_ids=(),
            schemes=(),
        ),
        hop_routing_plan=HopRoutingPlanDto(hop_addresses=(), routings=()),
    )


def _client(deployer: PlanDeployer[WorkerPlan]) -> TestClient:
    app = FastAPI()
    app.include_router(build_worker_plan_deployment_router(deployer))
    return TestClient(app)


def test_prepares_and_applies_worker_plan() -> None:
    deployer = RecordingPlanDeployer()
    client = _client(deployer)
    plan = _empty_worker_plan(version_number=1)

    prepare_response = client.put(
        PREPARE_SERVICE_PLAN_PATH.format(version_number=1),
        json={"worker_plan": plan.model_dump(mode="json")},
    )
    apply_response = client.put(
        APPLY_SERVICE_PLAN_PATH.format(version_number=1),
    )

    assert prepare_response.status_code == 204
    assert apply_response.status_code == 204
    assert [prepared.plan_version for prepared in deployer.prepared] == [
        plan.plan_version
    ]
    assert deployer.applied == [plan.plan_version]


def test_rejects_path_and_payload_version_mismatch() -> None:
    deployer = RecordingPlanDeployer()
    client = _client(deployer)

    response = client.put(
        PREPARE_SERVICE_PLAN_PATH.format(version_number=1),
        json={"worker_plan": _empty_worker_plan(2).model_dump(mode="json")},
    )

    assert response.status_code == 409
    assert deployer.prepared == []


def test_rejects_apply_when_plan_was_not_prepared() -> None:
    deployer = RecordingPlanDeployer()
    client = _client(deployer)

    response = client.put(
        APPLY_SERVICE_PLAN_PATH.format(version_number=1),
    )

    assert response.status_code == 409
    assert deployer.applied == []


def test_maps_hop_routing_plan() -> None:
    deployer = RecordingPlanDeployer()
    client = _client(deployer)
    plan = _empty_worker_plan(version_number=1)
    flow_id = FlowId(user_id=UserId())
    ingress_id = ServiceId()
    sink_id = ServiceId()
    plan = plan.model_copy(
        update={
            "hop_routing_plan": HopRoutingPlanDto(
                hop_addresses=(
                    InvocableHopAddressDto(
                        flow_id=flow_id,
                        hop=IngressHopDto(ingress_id=ingress_id),
                    ),
                ),
                routings=(
                    HopRoutingsDto(
                        input_routings=(
                            HopInputRoutingDto(
                                source=SourceHopAddressDto(
                                    flow_id=flow_id,
                                    hop=ExternalHopDto(),
                                ),
                                tensors=("input",),
                            ),
                        ),
                        output_routings=(
                            HopOutputRoutingDto(
                                destination=DestinationHopAddressDto(
                                    flow_id=flow_id,
                                    hop=SinkHopDto(sink_id=sink_id),
                                ),
                                tensors=("output",),
                            ),
                        ),
                    ),
                ),
            )
        }
    )

    response = client.put(
        PREPARE_SERVICE_PLAN_PATH.format(version_number=1),
        json={"worker_plan": plan.model_dump(mode="json")},
    )

    assert response.status_code == 204
    routing_plan = deployer.prepared[0].hop_routing_plan
    assert routing_plan.hop_addresses[0].hop == IngressHop(ingress_id)
    assert routing_plan.routings[0].input_routings[0].source.hop == ExternalHop()
    assert routing_plan.routings[0].output_routings[0].destination.hop == SinkHop(
        sink_id
    )
