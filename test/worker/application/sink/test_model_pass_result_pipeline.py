import asyncio

import numpy as np
import pytest

from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import (
    ModelPassContext,
    ModelPassId,
    ModelPassType,
)
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId
from worker.adapters.outbound.contribution.in_memory_contribution_store import (
    InMemoryContributionStore,
)
from worker.adapters.outbound.plan_store.in_memory_plan_store import InMemoryPlanStore
from worker.application.contribution.default_contribution_pipeline import (
    DefaultContributionPipeline,
)
from worker.application.contribution.default_postbox import DefaultPostbox
from worker.application.sink.default_model_pass_result_contribution_sink import (
    DefaultModelPassResultContributionSink,
)
from worker.application.sink.default_model_pass_result_retriever import (
    DefaultModelPassResultRetriever,
)
from worker.application.sink.model_pass_result_assembler import (
    ModelPassResultAssembler,
)
from worker.application.sink.model_pass_result_collection import (
    ModelPassResultCollection,
)
from worker.application.sink.model_pass_result_contribution_gathering_policy import (
    ModelPassResultContributionGatheringPolicy,
)
from worker.domain.model_pass.model_pass_error import ModelPassError
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.plan.sink.sink_plan import SinkPlan
from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


@pytest.mark.asyncio
async def test_collects_and_waits_for_a_model_pass_result() -> None:
    flow_id = FlowId(user_id=UserId())
    plan_version = PlanVersion(version_number=1)
    context = ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=flow_id,
        ),
        plan_version=plan_version,
        model_pass_type=ModelPassType.FORWARD,
    )

    plan_store = InMemoryPlanStore[SinkPlan]()
    await plan_store.put_plan(
        SinkPlan(
            plan_version=plan_version,
            flow_ids=(flow_id,),
            result_tensors_by_flow=(("output",),),
        )
    )

    result_postbox = DefaultPostbox[ModelPassId, ModelPassResultCollection](
        key_compute_callable=lambda collection: collection.key
    )
    error_postbox = DefaultPostbox[ModelPassId, ModelPassError](
        key_compute_callable=lambda error: error.context.model_pass_id
    )
    pipeline = DefaultContributionPipeline(
        gathering_policy=ModelPassResultContributionGatheringPolicy(plan_store),
        contribution_store=InMemoryContributionStore[
            ModelPassId,
            ModelPassResultContribution,
        ](),
        outbox=result_postbox,
    )
    sink = DefaultModelPassResultContributionSink(pipeline, error_postbox)
    retriever = DefaultModelPassResultRetriever(
        inbox=result_postbox,
        error_inbox=error_postbox,
        assembler=ModelPassResultAssembler(),
    )

    retrieve_task = asyncio.create_task(
        retriever.retrieve_result(context.model_pass_id)
    )
    await sink.sink_model_pass_contribution(
        ModelPassResultContribution(
            model_pass_context=context,
            bundle=TensorBundle(
                {"output": Tensor.from_numpy(np.array([1], dtype=np.float32))}
            ),
        )
    )

    result = await retrieve_task
    assert result.context == context
    assert result.bundle.get_tensor_names() == {"output"}


@pytest.mark.asyncio
async def test_propagates_a_model_pass_error_to_the_retriever() -> None:
    context = ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId()),
        ),
        plan_version=PlanVersion(version_number=1),
        model_pass_type=ModelPassType.FORWARD,
    )
    result_postbox = DefaultPostbox[
        ModelPassId,
        ModelPassResultCollection,
    ](key_compute_callable=lambda collection: collection.key)
    error_postbox = DefaultPostbox[ModelPassId, ModelPassError](
        key_compute_callable=lambda error: error.context.model_pass_id
    )
    retriever = DefaultModelPassResultRetriever(
        inbox=result_postbox,
        error_inbox=error_postbox,
        assembler=ModelPassResultAssembler(),
    )
    pipeline = DefaultContributionPipeline(
        gathering_policy=ModelPassResultContributionGatheringPolicy(
            InMemoryPlanStore[SinkPlan]()
        ),
        contribution_store=InMemoryContributionStore[
            ModelPassId,
            ModelPassResultContribution,
        ](),
        outbox=result_postbox,
    )
    sink = DefaultModelPassResultContributionSink(pipeline, error_postbox)

    retrieve_task = asyncio.create_task(
        retriever.retrieve_result(context.model_pass_id)
    )
    await sink.notify_error(ModelPassError(context=context, message="failed"))

    with pytest.raises(RuntimeError, match="failed"):
        await retrieve_task
