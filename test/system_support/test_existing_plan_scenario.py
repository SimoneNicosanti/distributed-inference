import asyncio
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import fakeredis
import httpx
import pytest
from test.system import existing_plan_scenario as scenario_module
from test.system.existing_plan_scenario import (
    load_existing_plan_scenario,
)
from test.system.optimization_readiness import wait_for_flow_plans

from integration.dto.context.context_dto import (
    IngressHopDto,
    PartitionReplicaHopDto,
    SinkHopDto,
)
from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)
from integration.dto.worker_plan.worker_plan_dto import (
    DeploymentPlanDto,
    FailurePlanDto,
    HopPlacementPlanDto,
    HopRoutingPlanDto,
    PartitionExecutionPlanDto,
    PartitionIOSchemeDto,
    ResourceAllocationDto,
    SchedulingPlanDto,
    WorkerPlanDto,
)
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from integration.redis.worker_plan.keys_factory import StringWorkerPlanKeyFactory
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def existing_plan_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, redis: fakeredis.FakeRedis
) -> list[httpx.Request]:
    model_path = tmp_path / "reference.onnx"
    model_path.touch()
    monkeypatch.setenv("FRONT_END_BASE_URL", "http://front-end")
    monkeypatch.setenv("REDIS_URL", "redis://existing-plan-test")
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("PLAN_DEPLOYMENT_TIMEOUT_S", "0")
    monkeypatch.delenv("CONTROLLER_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_MANAGER_BASE_URL", raising=False)
    monkeypatch.setattr(
        scenario_module.Redis, "from_url", lambda *args, **kwargs: redis
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        return httpx.Response(200, json={"service_id": "first-worker"})

    client = httpx.Client(
        base_url="http://front-end", transport=httpx.MockTransport(handle)
    )
    monkeypatch.setattr(scenario_module.httpx, "Client", lambda **kwargs: client)
    return requests


def _flow(number: int) -> FlowId:
    return FlowId(flow_id=UUID(int=number))


def _publish_plan(
    redis: fakeredis.FakeRedis,
    flow: FlowId,
    version_number: int,
    *,
    endpoint_worker: str = "first-worker",
) -> FlowPlanDto:
    version = PlanVersion(version_number=version_number)
    variant = ModelVariantId(ModelId(model_name="existing-model"), "fp32")
    plan = FlowPlanDto(
        plan_version=version,
        model_specs=ModelSpecsDto(
            model_variant_id=variant,
            inputs=(InputSpecDto(name="input", shape=(1,), dtype="float32"),),
        ),
        ingress_replica_id=PlannedIngressReplicaId(variant, replica_idx=1),
        ingress_worker_id=ServiceId(id=endpoint_worker),
        sink_replica_id=PlannedSinkReplicaId(variant, replica_idx=2),
        sink_worker_id=ServiceId(id=endpoint_worker),
    )
    redis.hset(
        asyncio.run(StringFlowPlanKeyFactory.build_hash_key(flow)),
        mapping={
            asyncio.run(StringFlowPlanKeyFactory.build_current_plan_field_key()): str(
                version_number
            ),
            asyncio.run(
                StringFlowPlanKeyFactory.build_plan_field_key(version)
            ): plan.model_dump_json(),
        },
    )
    partition = PlannedPartitionReplicaId(
        ModelPartitionId(variant, layers=("layer",)), replica_idx=0
    )
    compute_worker = ServiceId(id="compute-worker")
    placement = HopPlacementPlanDto(
        hops=(
            IngressHopDto(ingress_replica_id=plan.ingress_replica_id),
            PartitionReplicaHopDto(partition_replica_id=partition),
            SinkHopDto(sink_replica_id=plan.sink_replica_id),
        ),
        worker_ids=(plan.ingress_worker_id, compute_worker, plan.sink_worker_id),
    )
    for worker in (plan.ingress_worker_id, compute_worker):
        is_compute = worker == compute_worker
        worker_plan = WorkerPlanDto(
            plan_version=version,
            deployment_plan=DeploymentPlanDto(
                executable_hops=(partition,) if is_compute else (),
                allocations=(ResourceAllocationDto(use_gpu=False),)
                if is_compute
                else (),
                ingress_hops=() if is_compute else (plan.ingress_replica_id,),
                sink_hops=() if is_compute else (plan.sink_replica_id,),
            ),
            scheduling_plan=SchedulingPlanDto(priority_keys=(), priorities=()),
            partition_execution_plan=PartitionExecutionPlanDto(
                partition_ids=(partition.partition_id,) if is_compute else (),
                schemes=(PartitionIOSchemeDto(inputs=("input",), outputs=("output",)),)
                if is_compute
                else (),
            ),
            hop_routing_plan=HopRoutingPlanDto(hop_addresses=(), routings=()),
            failure_plan=FailurePlanDto(
                flow_ids=(flow,), sink_replica_ids=(plan.sink_replica_id,)
            ),
            hop_placement_plan=placement,
        )
        redis.hset(
            StringWorkerPlanKeyFactory.build_hash_key(version),
            StringWorkerPlanKeyFactory.build_field_key(worker),
            worker_plan.model_dump_json(),
        )
    return plan


def test_load_existing_plan_reads_latest_active_version_without_writes(
    redis: fakeredis.FakeRedis,
    existing_plan_environment: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flows = (_flow(1), _flow(2))
    for flow in flows:
        _publish_plan(redis, flow, 3)
    _publish_plan(redis, _flow(3), 4)
    plans = {flow: _publish_plan(redis, flow, 7) for flow in flows}
    redis.set("current-plan-version", "100")
    inactive_plan = plans[flows[0]].model_copy(
        update={"plan_version": PlanVersion(version_number=99)}
    )
    redis.hset(
        asyncio.run(StringFlowPlanKeyFactory.build_hash_key(flows[0])),
        asyncio.run(
            StringFlowPlanKeyFactory.build_plan_field_key(inactive_plan.plan_version)
        ),
        inactive_plan.model_dump_json(),
    )
    commands = Mock(wraps=redis.execute_command)
    monkeypatch.setattr(redis, "execute_command", commands)

    scenario = load_existing_plan_scenario()

    assert scenario.flow_plans == plans
    assert tuple(scenario.flow_plans) == flows
    assert scenario.front_end_base_url == "http://front-end"
    assert len(existing_plan_environment) == 1
    assert all(call.args[0] in {"HGET", "SCAN"} for call in commands.call_args_list)


def test_load_existing_plan_fails_without_active_plans(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    with pytest.raises(pytest.fail.Exception, match="No active flow plans found"):
        load_existing_plan_scenario()


def test_load_existing_plan_does_not_reoptimize_missing_plans(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    redis.hset(
        asyncio.run(StringFlowPlanKeyFactory.build_hash_key(_flow(1))),
        asyncio.run(StringFlowPlanKeyFactory.build_current_plan_field_key()),
        "7",
    )

    with pytest.raises(AssertionError, match="Active version has no flow plan"):
        load_existing_plan_scenario()


def test_load_existing_plan_rejects_endpoints_on_another_worker(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    flow = _flow(1)
    _publish_plan(redis, flow, 7, endpoint_worker="another-worker")

    with pytest.raises(AssertionError):
        load_existing_plan_scenario()


@pytest.mark.parametrize("worker", ["first-worker", "compute-worker"])
def test_load_existing_plan_rejects_missing_worker_plan(
    redis: fakeredis.FakeRedis,
    existing_plan_environment: list[httpx.Request],
    worker: str,
) -> None:
    plan = _publish_plan(redis, _flow(1), 7)
    redis.hdel(
        StringWorkerPlanKeyFactory.build_hash_key(plan.plan_version),
        StringWorkerPlanKeyFactory.build_field_key(ServiceId(id=worker)),
    )

    with pytest.raises(AssertionError, match="Active flow has no worker plan"):
        load_existing_plan_scenario()


def test_load_existing_plan_ignores_staged_flows_and_legacy_keys(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    flow = _flow(1)
    plan = _publish_plan(redis, flow, 7)
    redis.hset(
        asyncio.run(StringFlowPlanKeyFactory.build_hash_key(_flow(2))),
        "version-99",
        plan.model_dump_json(),
    )
    redis.set(f"flow-plans:{_flow(3).flow_id.hex}:active-version", "100")

    assert load_existing_plan_scenario().flow_plans == {flow: plan}


def test_publication_readiness_allows_different_active_versions(
    redis: fakeredis.FakeRedis,
) -> None:
    flows = (_flow(1), _flow(2))
    expected = {
        flows[0]: _publish_plan(redis, flows[0], 3),
        flows[1]: _publish_plan(redis, flows[1], 7),
    }

    assert (
        wait_for_flow_plans(redis, flows, 0, ServiceId(id="first-worker")) == expected
    )


@pytest.mark.parametrize("worker", ["first-worker", "compute-worker"])
def test_publication_readiness_rejects_worker_version_mismatch(
    redis: fakeredis.FakeRedis, worker: str
) -> None:
    flow = _flow(1)
    plan = _publish_plan(redis, flow, 7)
    key = StringWorkerPlanKeyFactory.build_hash_key(plan.plan_version)
    field = StringWorkerPlanKeyFactory.build_field_key(ServiceId(id=worker))
    worker_plan = WorkerPlanDto.model_validate_json(redis.hget(key, field))
    redis.hset(
        key,
        field,
        worker_plan.model_copy(
            update={"plan_version": PlanVersion(version_number=8)}
        ).model_dump_json(),
    )

    with pytest.raises(AssertionError):
        wait_for_flow_plans(redis, (flow,), 0, ServiceId(id="first-worker"))


@pytest.mark.parametrize("worker", ["first-worker", "compute-worker"])
def test_publication_readiness_rejects_inconsistent_placement(
    redis: fakeredis.FakeRedis, worker: str
) -> None:
    flow = _flow(1)
    plan = _publish_plan(redis, flow, 7)
    key = StringWorkerPlanKeyFactory.build_hash_key(plan.plan_version)
    field = StringWorkerPlanKeyFactory.build_field_key(ServiceId(id=worker))
    worker_plan = WorkerPlanDto.model_validate_json(redis.hget(key, field))
    placement = worker_plan.hop_placement_plan
    redis.hset(
        key,
        field,
        worker_plan.model_copy(
            update={
                "hop_placement_plan": HopPlacementPlanDto(
                    hops=placement.hops,
                    worker_ids=tuple(
                        ServiceId(id="another-worker")
                        if owner == ServiceId(id=worker)
                        else owner
                        for owner in placement.worker_ids
                    ),
                )
            }
        ).model_dump_json(),
    )

    with pytest.raises(AssertionError, match="placement disagrees"):
        wait_for_flow_plans(redis, (flow,), 0, ServiceId(id="first-worker"))
