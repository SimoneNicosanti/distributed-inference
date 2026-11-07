import asyncio
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from catalog.api.worker.worker_directory import WorkerDirectory
from integration.dto.worker_plan.worker_plan_dto import FailurePlanDto
from integration.service_instance.service_instance import (
    CapabilityInterface,
    CapabilityProtocol,
)
from integration.worker.failure.http_failure_schema import NOTIFY_FAILURE_PATH
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.context_mapper import model_pass_context_to_dto
from worker.adapters.inbound.failure.http.http_failure_receiver import (
    build_failure_receiver_router,
)
from worker.adapters.outbound.failure.http.http_failure_sender import HttpFailureSender
from worker.application.failure.default_failure_propagator import (
    DefaultFailurePropagator,
)
from worker.application.failure.default_failure_receiver import DefaultFailureReceiver
from worker.application.hop_handler.default_planned_partition_replica_hop_handler import (
    DefaultPlannedPartitionReplicaHopHandler,
)
from worker.application.ports.outbound.failure_snd.failure_sender import (
    FailureSendError,
)
from worker.application.postbox.keyed_postbox import KeyedPostbox
from worker.application.sink.default_model_pass_outcome_retriever import (
    DefaultModelPassResultRetriever,
)
from worker.domain.model.model_pass import ModelPassError
from worker.domain.plan.failure_plan import FailurePlan


def _context():
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId())
        ),
        plan_version=PlanVersion(1),
        model_pass_type=ModelPassType.FORWARD,
    )


@pytest.mark.asyncio
async def test_failure_reaches_sink_through_http():
    context = _context()
    outcome = ModelPassError(context=context, message="execution failed")
    sink = PlannedSinkReplicaId(ModelVariantId(ModelId("model"), "variant"), 0)
    plan = FailurePlan((context.flow_id,), (sink,))
    plans = SimpleNamespace(get_failure_plan_by_version=AsyncMock(return_value=plan))
    worker_id = ServiceId()
    hops = SimpleNamespace(
        get_worker_for_planned_invocable_hop=AsyncMock(return_value=worker_id)
    )
    interfaces = [
        CapabilityInterface(protocol=CapabilityProtocol.HTTP, endpoint="http://sink")
    ]
    worker = SimpleNamespace(
        get_capability_interfaces_by_type_and_protocol=lambda *args: interfaces
    )
    workers = AsyncMock(spec=WorkerDirectory)
    workers.get_worker_instance_by_id.return_value = worker
    postbox = KeyedPostbox(lambda value: value.context.model_pass_id)
    app = FastAPI()
    app.include_router(build_failure_receiver_router(DefaultFailureReceiver(postbox)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        propagator = DefaultFailurePropagator(
            plans, hops, HttpFailureSender(workers, client)
        )
        await propagator.propagate_failure(outcome)
    received = await DefaultModelPassResultRetriever(postbox).retrieve_outcome(
        context.model_pass_id
    )
    assert received == outcome
    plans.get_failure_plan_by_version.assert_awaited_once_with(context.plan_version)
    hops.get_worker_for_planned_invocable_hop.assert_awaited_once_with(
        context.plan_version, sink
    )
    workers.get_worker_instance_by_id.assert_awaited_once_with(worker_id)


@pytest.mark.asyncio
async def test_receiver_rejects_mismatched_pass_id():
    receiver = AsyncMock()
    app = FastAPI()
    app.include_router(build_failure_receiver_router(receiver))
    context = _context()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sink"
    ) as client:
        response = await client.post(
            NOTIFY_FAILURE_PATH.format(model_pass_id=uuid4()),
            json={
                "context": model_pass_context_to_dto(context).model_dump(mode="json"),
                "message": "failed",
            },
        )
    assert response.status_code == 409
    receiver.notify_failure.assert_not_awaited()


@pytest.mark.asyncio
async def test_sender_preserves_http_failure():
    worker = SimpleNamespace(
        get_capability_interfaces_by_type_and_protocol=lambda *args: [
            CapabilityInterface(
                protocol=CapabilityProtocol.HTTP, endpoint="http://sink"
            )
        ]
    )
    workers = AsyncMock(spec=WorkerDirectory)
    workers.get_worker_instance_by_id.return_value = worker
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(FailureSendError) as caught:
            await HttpFailureSender(workers, client).send_failure(
                ModelPassError(_context(), "failed"), ServiceId()
            )
    assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.parametrize("duplicate", [False, True])
def test_failure_plan_rejects_invalid_flow_mapping(duplicate):
    flow = FlowId(user_id=UserId())
    sink = PlannedSinkReplicaId(ModelVariantId(ModelId("model"), "variant"), 0)
    flows = (flow, flow) if duplicate else (flow,)
    sinks = (sink, sink) if duplicate else ()
    with pytest.raises(ValueError):
        FailurePlan(flows, sinks)
    with pytest.raises(ValidationError):
        FailurePlanDto(flow_ids=flows, sink_replica_ids=sinks)


@pytest.mark.asyncio
async def test_partition_handler_completes_failed_future_and_continues():
    queue = asyncio.Queue()
    scheduler = SimpleNamespace(dequeue=queue.get)

    @asynccontextmanager
    async def grant():
        yield

    activities = SimpleNamespace(
        acquire_activity_grant=AsyncMock(side_effect=lambda request: grant())
    )
    factory = SimpleNamespace(
        create_request_for_activity_type=lambda activity: object()
    )
    handler = DefaultPlannedPartitionReplicaHopHandler(
        factory, activities, scheduler, object()
    )
    failure = RuntimeError("execution failed")
    success = object()
    handler._execute_partition_invocation = AsyncMock(side_effect=[failure, success])
    first = asyncio.get_running_loop().create_future()
    second = asyncio.get_running_loop().create_future()
    await queue.put((object(), first))
    await queue.put((object(), second))
    task = asyncio.create_task(handler._controller_loop())
    try:
        with pytest.raises(RuntimeError) as caught:
            await asyncio.wait_for(asyncio.shield(first), timeout=1)
        assert caught.value is failure
        assert await asyncio.wait_for(asyncio.shield(second), timeout=1) is success
        assert not task.done()
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
