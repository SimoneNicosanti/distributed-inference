from dataclasses import asdict
from typing import override

from fastapi import FastAPI
from fastapi.testclient import TestClient

from integration.dto.service_plan.service_plan_dto import ServicePlanDto
from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.inbound.deployment.http.http_service_plan_deployer import (
    build_service_plan_deployment_router,
)
from worker.application.ports.inbound.deployment.plan_deployer import PlanDeployer
from worker.domain.plan.ingress.flow_plan import FlowPlan, IngressInputRoute
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.plan.service_plan import ServicePlan
from worker.domain.plan.sink.sink_plan import SinkPlan
from worker.domain.plan.worker.deployment_plan import DeploymentPlan
from worker.domain.plan.worker.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.worker.routing_plan import RoutingPlan
from worker.domain.plan.worker.scheduling_plan import SchedulingPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


class RecordingPlanDeployer[PlanT](PlanDeployer[PlanT]):
    def __init__(self) -> None:
        self.prepared: list[PlanT] = []
        self.applied: list[PlanVersion] = []

    @override
    async def prepare_plan(self, plan: PlanT) -> None:
        self.prepared.append(plan)

    @override
    async def apply_plan(self, plan_version: PlanVersion) -> None:
        self.applied.append(plan_version)


def _empty_worker_plan(plan_version: PlanVersion) -> WorkerPlan:
    return WorkerPlan(
        plan_version=plan_version,
        deployment_plan=DeploymentPlan(replica_ids=(), allocations=()),
        scheduling_plan=SchedulingPlan(priority_keys=(), priorities=()),
        partition_execution_plan=PartitionExecutionPlan(
            partition_ids=(),
            schemes=(),
        ),
        routing_plan=RoutingPlan(replica_ids=(), routings=()),
    )


def _ingress_plan(plan_version: PlanVersion) -> IngressPlan:
    user_id = UserId()
    flow_id = FlowId(user_id=user_id)
    model_variant_id = ModelVariantId(
        model_id=ModelId(model_name="model"),
        variant_tag="variant",
    )
    replica_id = PartitionReplicaId(
        partition_id=ModelPartitionId(
            model_variant_id=model_variant_id,
            layers=("compute",),
        ),
        replica_idx=0,
    )
    return IngressPlan(
        plan_version=plan_version,
        flow_ids=(flow_id,),
        flow_plans=(
            FlowPlan(
                model_variant_id=model_variant_id,
                input_routes=(
                    IngressInputRoute(
                        target_replica_id=replica_id,
                        tensors=("input",),
                    ),
                ),
                sink_id=ServiceId(id="sink"),
            ),
        ),
    )


def _client(
    worker_deployer: PlanDeployer[WorkerPlan],
    ingress_deployer: PlanDeployer[IngressPlan] | None,
    sink_deployer: PlanDeployer[SinkPlan] | None,
) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_service_plan_deployment_router(
            worker_deployer,
            ingress_deployer,
            sink_deployer,
        )
    )
    return TestClient(app)


def _plan_json(plan: ServicePlan) -> dict[str, object]:
    return ServicePlanDto.model_validate(asdict(plan)).model_dump(mode="json")


def test_prepares_and_applies_worker_ingress_and_sink_plans() -> None:
    worker_deployer = RecordingPlanDeployer[WorkerPlan]()
    ingress_deployer = RecordingPlanDeployer[IngressPlan]()
    sink_deployer = RecordingPlanDeployer[SinkPlan]()
    client = _client(worker_deployer, ingress_deployer, sink_deployer)
    plan_version = PlanVersion(version_number=1)
    ingress_plan = _ingress_plan(plan_version)
    plan = ServicePlan(
        worker_plan=_empty_worker_plan(plan_version),
        ingress_plan=ingress_plan,
        sink_plan=SinkPlan(
            plan_version=plan_version,
            flow_ids=ingress_plan.flow_ids,
            result_tensors_by_flow=(("result",),),
        ),
    )

    prepare_response = client.put(
        PREPARE_SERVICE_PLAN_PATH.format(version_number=1),
        json={"service_plan": _plan_json(plan)},
    )
    apply_response = client.put(
        APPLY_SERVICE_PLAN_PATH.format(version_number=1),
    )

    assert prepare_response.status_code == 204
    assert apply_response.status_code == 204
    assert worker_deployer.prepared == [plan.worker_plan]
    assert worker_deployer.applied == [plan_version]
    assert ingress_deployer.prepared == [plan.ingress_plan]
    assert ingress_deployer.applied == [plan_version]
    assert sink_deployer.prepared == [plan.sink_plan]
    assert sink_deployer.applied == [plan_version]


def test_skips_ingress_deployer_when_plan_has_no_ingress_component() -> None:
    worker_deployer = RecordingPlanDeployer[WorkerPlan]()
    ingress_deployer = RecordingPlanDeployer[IngressPlan]()
    sink_deployer = RecordingPlanDeployer[SinkPlan]()
    client = _client(worker_deployer, ingress_deployer, sink_deployer)
    plan_version = PlanVersion(version_number=1)
    plan = ServicePlan(
        worker_plan=_empty_worker_plan(plan_version),
        ingress_plan=None,
    )

    prepare_response = client.put(
        PREPARE_SERVICE_PLAN_PATH.format(version_number=1),
        json={"service_plan": _plan_json(plan)},
    )
    apply_response = client.put(
        APPLY_SERVICE_PLAN_PATH.format(version_number=1),
    )

    assert prepare_response.status_code == 204
    assert apply_response.status_code == 204
    assert worker_deployer.prepared == [plan.worker_plan]
    assert worker_deployer.applied == [plan_version]
    assert ingress_deployer.prepared == []
    assert ingress_deployer.applied == []
    assert sink_deployer.prepared == []
    assert sink_deployer.applied == []


def test_rejects_path_and_payload_version_mismatch() -> None:
    worker_deployer = RecordingPlanDeployer[WorkerPlan]()
    client = _client(worker_deployer, None, None)
    plan = ServicePlan(
        worker_plan=_empty_worker_plan(PlanVersion(version_number=2)),
        ingress_plan=None,
    )

    response = client.put(
        PREPARE_SERVICE_PLAN_PATH.format(version_number=1),
        json={"service_plan": _plan_json(plan)},
    )

    assert response.status_code == 409
    assert worker_deployer.prepared == []
