from unittest.mock import AsyncMock

import numpy as np
import pytest

from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassContext, ModelPassType
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.outbound.plan_store.in_memory_plan_store import InMemoryPlanStore
from worker.application.ingress.abc.model_pass_coordinator import ModelPassRequestInfo
from worker.application.ingress.default_model_invocation_coordinator import (
    DefaultModelInvocationCoordinator,
)
from worker.application.ingress.default_model_pass_coordinator import (
    DefaultModelPassCoordinator,
)
from worker.application.ports.outbound.sink.model_pass_result_gateway import (
    ModelPassResultRetrieverGateway,
)
from worker.domain.model_invocation.model_invocation import ModelInvocationResult
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.plan.ingress.flow_plan import FlowPlan, IngressInputRoute
from worker.domain.plan.ingress.ingress_plan import IngressPlan
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


def _plan() -> tuple[IngressPlan, FlowId, ServiceId]:
    flow_id = FlowId(user_id=UserId())
    sink_id = ServiceId(id="sink")
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
    return (
        IngressPlan(
            plan_version=PlanVersion(version_number=1),
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
                    sink_id=sink_id,
                ),
            ),
        ),
        flow_id,
        sink_id,
    )


async def _active_plan_store(plan: IngressPlan) -> InMemoryPlanStore[IngressPlan]:
    store = InMemoryPlanStore[IngressPlan]()
    await store.put_plan(plan)
    await store.activate_plan(plan.plan_version)
    return store


@pytest.mark.asyncio
async def test_model_invocation_context_contains_only_flow_identity() -> None:
    plan, flow_id, _ = _plan()
    store = await _active_plan_store(plan)
    bundle = TensorBundle({"input": Tensor.from_numpy(np.array([1], dtype=np.float32))})
    model_pass_coordinator = AsyncMock()
    model_pass_coordinator.run_model_pass.return_value = ModelPassResult(
        context=ModelPassContext(
            model_invocation_context=ModelInvocationContext(
                flow_id=flow_id,
            ),
            plan_version=plan.plan_version,
            model_pass_type=ModelPassType.FORWARD,
        ),
        bundle=bundle,
    )
    coordinator = DefaultModelInvocationCoordinator(
        active_plan_version_reader=store,
        model_pass_coordinator=model_pass_coordinator,
    )

    result: ModelInvocationResult = await coordinator.run_model_invocation(
        flow_id,
        bundle,
    )

    request_info = model_pass_coordinator.run_model_pass.await_args.kwargs[
        "model_pass_request_info"
    ]
    assert request_info.model_invocation_context.flow_id == flow_id
    assert result.context.flow_id == flow_id


@pytest.mark.asyncio
async def test_model_pass_gateway_uses_flow_sink_id() -> None:
    plan, flow_id, sink_id = _plan()
    store = await _active_plan_store(plan)
    invocation_context = ModelInvocationContext(
        flow_id=flow_id,
    )
    expected = ModelPassResult(
        context=ModelPassContext(
            model_invocation_context=invocation_context,
            plan_version=plan.plan_version,
            model_pass_type=ModelPassType.FORWARD,
        ),
        bundle=TensorBundle(
            {"output": Tensor.from_numpy(np.array([2], dtype=np.float32))}
        ),
    )
    input_router = AsyncMock()
    input_router.route.return_value = []
    contribution_sender = AsyncMock()
    gateway = AsyncMock(spec=ModelPassResultRetrieverGateway)
    gateway.retrieve_result.return_value = expected
    coordinator = DefaultModelPassCoordinator(
        input_router=input_router,
        ingress_plan_reader=store,
        contribution_sender_coordinator=contribution_sender,
        result_retriever_gateway=gateway,
    )

    result = await coordinator.run_model_pass(
        ModelPassRequestInfo(
            model_invocation_context=invocation_context,
            plan_version=plan.plan_version,
            model_pass_type=ModelPassType.FORWARD,
        ),
        TensorBundle({"input": Tensor.from_numpy(np.array([1], dtype=np.float32))}),
    )

    assert result == expected
    assert gateway.retrieve_result.await_args.kwargs["result_sink_id"] == sink_id
