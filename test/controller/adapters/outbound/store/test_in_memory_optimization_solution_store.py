import pytest

from controller.adapters.outbound.store.in_memory_optimization_solution_store import (
    InMemoryOptimizationSolutionStore,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.profile.model_variant_static_profile import TensorProperty
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from shared.user.user import UserId


def _solution() -> EnrichedOptimizationSolution:
    variant = ModelVariantId(ModelId("model"), "variant")
    worker_id = ServiceId()
    component = ComponentSolution(
        component_index=1,
        worker_id=worker_id,
        layer_keys=("compute",),
        input_layer_keys=(),
        output_layer_keys=(),
        tensor_contract=ComponentTensorContract(("x",), ("y",)),
        priority=1,
        mean_execution_time_s=0,
        mean_execution_queue_time_s=0,
        mean_transmission_time_s=0,
        transmission_latency_s=0,
    )
    flow = FlowOptimizationSolution(
        flow_id=FlowId(user_id=UserId()),
        ingress_worker_id=worker_id,
        sink_worker_id=worker_id,
        model_variant_id=variant,
        components=(component,),
        cuts=(),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )
    return EnrichedOptimizationSolution(
        OptimizationSolution((flow,), 0, True),
        {variant: {"x": TensorProperty("float32", 4, ("batch", 3))}},
    )


@pytest.mark.asyncio
async def test_initially_has_no_current_solution() -> None:
    store = InMemoryOptimizationSolutionStore()

    assert await store.get_current_solution() is None


@pytest.mark.asyncio
async def test_preserves_enriched_solution_for_reads_and_observation() -> None:
    store = InMemoryOptimizationSolutionStore()
    solution = _solution()

    await store.put_solution(solution)

    assert await store.get_current_solution() is solution
    assert await store.observe_new_solution() is solution
    assert await store.get_current_solution() is solution


@pytest.mark.asyncio
async def test_replaces_unobserved_solution_with_latest_enriched_solution() -> None:
    store = InMemoryOptimizationSolutionStore()
    first, latest = _solution(), _solution()

    await store.put_solution(first)
    await store.put_solution(latest)

    assert await store.get_current_solution() is latest
    assert await store.observe_new_solution() is latest


@pytest.mark.asyncio
async def test_observation_keeps_accepting_subsequent_updates() -> None:
    store = InMemoryOptimizationSolutionStore()
    first, latest = _solution(), _solution()

    await store.put_solution(first)
    assert await store.observe_new_solution() is first
    await store.put_solution(latest)

    assert await store.observe_new_solution() is latest
    assert await store.get_current_solution() is latest
