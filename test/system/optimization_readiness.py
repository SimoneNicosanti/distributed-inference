"""Observe public Redis contracts without importing service implementations."""

import asyncio
import os
import time

import httpx
import pytest
from redis import Redis
from redis import asyncio as redis_asyncio
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from catalog.redis.hop.keys_factory import StringHopDeploymentKeyFactory
from catalog.redis.service.redis_service_directory import RedisServiceDirectory
from catalog.view.worker.default_worker_directory import DefaultWorkerDirectory
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from integration.dto.worker_profiles.network_profile_dto import NetworkProfileDto
from integration.dto.worker_profiles.resource_profile_dto import ResourceProfileDto
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from integration.redis.worker_profiles.execution_profile_keys import (
    StringExecutionProfileKeyFactory,
)
from integration.redis.worker_profiles.network_profile_keys import (
    StringNetworkProfileKeyFactory,
)
from integration.redis.worker_profiles.resource_profile_keys import (
    StringResourceProfileKeyFactory,
)
from integration.shared.id_json import (
    planned_invocable_hop_id_to_json,
    service_id_from_json,
)
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


def wait_for_http(base_url: str, path: str) -> None:
    deadline = time.monotonic() + float(os.getenv("STACK_READY_TIMEOUT_S", "120"))
    last_error = "No response"
    with httpx.Client(base_url=base_url, timeout=5.0) as client:
        while True:
            try:
                response = client.get(path)
                if response.is_success:
                    return
                last_error = f"{response.status_code}: {response.text}"
            except httpx.TransportError as error:
                last_error = str(error)
            if time.monotonic() >= deadline:
                pytest.fail(f"{base_url}{path} did not become ready: {last_error}")
            time.sleep(1.0)


def wait_for_worker_profiles(
    redis_url: str,
    ingress_id: WorkerId,
    variant_ids: set[ModelVariantId],
    timeout_s: float,
) -> None:
    asyncio.run(
        _wait_for_worker_profiles(redis_url, ingress_id, variant_ids, timeout_s)
    )


async def _wait_for_worker_profiles(
    redis_url: str,
    ingress_id: WorkerId,
    variant_ids: set[ModelVariantId],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    expected_count = int(os.getenv("EXPECTED_WORKER_COUNT", "1"))
    assert expected_count > 0
    async with redis_asyncio.Redis.from_url(redis_url, decode_responses=True) as redis:
        directory = DefaultWorkerDirectory(RedisServiceDirectory(redis))
        while True:
            pending: list[str] = []
            try:
                workers = await directory.get_all_worker_instances()
                worker_ids = {worker.service_id for worker in workers}
                if len(worker_ids) < expected_count or ingress_id not in worker_ids:
                    pending.append(f"Workers not ready: {worker_ids}")
                for worker_id in worker_ids:
                    execution_key = await StringExecutionProfileKeyFactory.build_execution_profile_key_per_worker(
                        worker_id
                    )
                    resource_key = await StringResourceProfileKeyFactory.build_resource_profile_key_per_worker(
                        worker_id
                    )
                    network_key = await StringNetworkProfileKeyFactory.build_network_profile_key_per_worker(
                        worker_id
                    )
                    profiles = [
                        ModelExecutionProfileDto.model_validate_json(value)
                        for value in await redis.hvals(execution_key)
                    ]
                    assert all(profile.worker_id == worker_id for profile in profiles)
                    if not variant_ids <= {
                        profile.model_variant_id for profile in profiles
                    }:
                        pending.append(f"Execution profiles missing: {worker_id}")
                    resource = await redis.get(resource_key)
                    if resource is None:
                        pending.append(f"Resource profile missing: {worker_id}")
                    else:
                        assert (
                            ResourceProfileDto.model_validate_json(resource).worker_id
                            == worker_id
                        )
                    network = await redis.get(network_key)
                    if network is None:
                        pending.append(f"Network profile missing: {worker_id}")
                    else:
                        profile = NetworkProfileDto.model_validate_json(network)
                        assert profile.source_worker_id == worker_id
                        targets = {
                            connection.target_worker_id
                            for connection in profile.connections
                        }
                        if not worker_ids - {worker_id} <= targets:
                            pending.append(f"Network connections missing: {worker_id}")
            except (RedisConnectionError, RedisTimeoutError) as error:
                pending.append(str(error))
            if not pending:
                return
            if time.monotonic() >= deadline:
                pytest.fail(f"Worker readiness timed out: {'; '.join(pending)}")
            await asyncio.sleep(1.0)


def wait_for_flow_plans(
    redis: Redis,
    flow_ids: tuple[FlowId, ...],
    timeout_s: float,
    endpoint_worker_id: WorkerId,
) -> dict[FlowId, FlowPlanDto]:
    keys = {flow_id: asyncio.run(_flow_keys(flow_id)) for flow_id in flow_ids}
    deadline = time.monotonic() + timeout_s
    while True:
        plans: dict[FlowId, FlowPlanDto] = {}
        pending: list[str] = []
        for flow_id, (active_key, versions_key) in keys.items():
            active = redis.get(active_key)
            if active is None:
                pending.append(f"No active version: {flow_id}")
                continue
            version = PlanVersion(version_number=int(active))
            field = asyncio.run(
                StringFlowPlanKeyFactory.build_flow_plan_elem_key(version)
            )
            value = redis.hget(versions_key, field)
            assert value is not None, f"Active version has no flow plan: {flow_id}"
            plan = FlowPlanDto.model_validate_json(value)
            assert plan.plan_version == version
            deployments_key = asyncio.run(
                StringHopDeploymentKeyFactory.build_hop_deployments_key(version)
            )
            for hop in (plan.ingress_replica_id, plan.sink_replica_id):
                worker = redis.hget(
                    deployments_key, planned_invocable_hop_id_to_json(hop)
                )
                assert worker is not None, f"Published endpoint is not deployed: {hop}"
                assert service_id_from_json(worker) == endpoint_worker_id
            plans[flow_id] = plan
        if not pending:
            assert len({plan.plan_version for plan in plans.values()}) == 1
            return plans
        if time.monotonic() >= deadline:
            pytest.fail(f"Flow plan publication timed out: {'; '.join(pending)}")
        time.sleep(1.0)


async def _flow_keys(flow_id: FlowId) -> tuple[str, str]:
    return (
        await StringFlowPlanKeyFactory.build_current_plan_key(flow_id),
        await StringFlowPlanKeyFactory.build_flow_plans_dict_key(flow_id),
    )
