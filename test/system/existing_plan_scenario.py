"""Reuse published flow plans without uploading models or requesting optimization."""

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from redis import Redis
from test.system.optimization_readiness import wait_for_flow_plans

from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.dto.worker_identity.worker_identity_dto import WorkerIdentityDto
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
)
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


@dataclass(frozen=True)
class ExistingPlanScenario:
    front_end_base_url: str
    model_path: Path
    flow_plans: dict[FlowId, FlowPlanDto]


def _latest_active_flow_ids(redis: Redis) -> tuple[FlowId, ...]:
    active_flows: dict[FlowId, PlanVersion] = {}
    for key in redis.scan_iter(match="flow-plans:*:active-version"):
        active_version = redis.get(key)
        if active_version is not None:
            flow_id = FlowId(flow_id=UUID(key.split(":")[1]))
            active_flows[flow_id] = PlanVersion(version_number=int(active_version))
    if not active_flows:
        pytest.fail(
            "No active flow plans found in Redis. "
            "Run test/system/test_optimization_request.py first."
        )
    latest_version = max(active_flows.values())
    return tuple(
        sorted(
            (
                flow
                for flow, version in active_flows.items()
                if version == latest_version
            ),
            key=lambda flow: flow.flow_id.hex,
        )
    )


def load_existing_plan_scenario() -> ExistingPlanScenario:
    front_end_base_url = os.getenv("FRONT_END_BASE_URL")
    if front_end_base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run system tests")
    redis_url = os.getenv("REDIS_URL")
    if redis_url is None:
        pytest.fail("Set REDIS_URL when FRONT_END_BASE_URL is configured")
    model_path_value = os.getenv("MODEL_PATH")
    if model_path_value is None:
        pytest.fail("Set MODEL_PATH to the reference ONNX model for the published plan")
    model_path = Path(model_path_value)
    if not model_path.is_file():
        pytest.fail(f"MODEL_PATH is not a file: {model_path}")

    with httpx.Client(base_url=front_end_base_url, timeout=30.0) as client:
        response = client.get(FRONT_END_LOCAL_WORKER_IDENTITY_PATH)
        assert response.status_code == 200, response.text
        identity = WorkerIdentityDto.model_validate(response.json())
    endpoint_worker_id = ServiceId(id=identity.service_id)

    with Redis.from_url(redis_url, decode_responses=True) as redis:
        flow_ids = _latest_active_flow_ids(redis)
        plans = wait_for_flow_plans(
            redis,
            flow_ids,
            float(os.getenv("PLAN_DEPLOYMENT_TIMEOUT_S", "120")),
            endpoint_worker_id,
        )

    return ExistingPlanScenario(
        front_end_base_url=front_end_base_url,
        model_path=model_path,
        flow_plans=plans,
    )
