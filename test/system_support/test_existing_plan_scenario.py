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

from catalog.redis.hop.keys_factory import StringHopDeploymentKeyFactory
from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from integration.shared.id_json import (
    planned_invocable_hop_id_to_json,
    service_id_to_json,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
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
        sink_replica_id=PlannedSinkReplicaId(variant, replica_idx=2),
    )
    redis.set(
        asyncio.run(StringFlowPlanKeyFactory.build_current_plan_key(flow)),
        str(version_number),
    )
    redis.hset(
        asyncio.run(StringFlowPlanKeyFactory.build_flow_plans_dict_key(flow)),
        asyncio.run(StringFlowPlanKeyFactory.build_flow_plan_elem_key(version)),
        plan.model_dump_json(),
    )
    deployments_key = asyncio.run(
        StringHopDeploymentKeyFactory.build_hop_deployments_key(version)
    )
    for hop in (plan.ingress_replica_id, plan.sink_replica_id):
        redis.hset(
            deployments_key,
            planned_invocable_hop_id_to_json(hop),
            service_id_to_json(ServiceId(id=endpoint_worker)),
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
        asyncio.run(StringFlowPlanKeyFactory.build_flow_plans_dict_key(flows[0])),
        "plan-version-99",
        inactive_plan.model_dump_json(),
    )
    commands = Mock(wraps=redis.execute_command)
    monkeypatch.setattr(redis, "execute_command", commands)

    scenario = load_existing_plan_scenario()

    assert scenario.flow_plans == plans
    assert tuple(scenario.flow_plans) == flows
    assert scenario.front_end_base_url == "http://front-end"
    assert len(existing_plan_environment) == 1
    assert all(
        call.args[0] in {"GET", "HGET", "SCAN"} for call in commands.call_args_list
    )


def test_load_existing_plan_fails_without_active_plans(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    with pytest.raises(pytest.fail.Exception, match="No active flow plans found"):
        load_existing_plan_scenario()


def test_load_existing_plan_does_not_reoptimize_missing_plans(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    redis.set(
        asyncio.run(StringFlowPlanKeyFactory.build_current_plan_key(_flow(1))), "7"
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


def test_load_existing_plan_rejects_missing_endpoint_deployment(
    redis: fakeredis.FakeRedis, existing_plan_environment: list[httpx.Request]
) -> None:
    plan = _publish_plan(redis, _flow(1), 7)
    redis.hdel(
        asyncio.run(
            StringHopDeploymentKeyFactory.build_hop_deployments_key(plan.plan_version)
        ),
        planned_invocable_hop_id_to_json(plan.ingress_replica_id),
    )

    with pytest.raises(AssertionError, match="Published endpoint is not deployed"):
        load_existing_plan_scenario()
