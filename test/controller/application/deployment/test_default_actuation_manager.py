from collections.abc import Mapping
from unittest.mock import AsyncMock, Mock

import pytest

from controller.application.deployment.abc.solution_observer import SolutionObserver
from controller.application.deployment.abc.solution_to_plan_converter import (
    SolutionToPlanConverter,
)
from controller.application.ports.outbound.plan_version.plan_version_factory import (
    PlanVersionBuilder,
)
from controller.application.deployment.default_actuation_manager import (
    DefaultActuationManager,
)
from controller.application.deployment.default_all_worker_plan_builder import (
    DefaultSolutionToPlanConverter,
)
from controller.application.ports.outbound.partition.partition_builder_gateway import (
    PartitionBuilderGateway,
)
from controller.application.ports.outbound.plan_share_gateway import PlanShareGateway
from controller.application.ports.outbound.store.flow_plan_publisher import (
    FlowPlanPublisher,
)
from controller.domain.backend import Backend
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.plan.front_end_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from controller.domain.profile.model_variant_static_profile import TensorProperty
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId


@pytest.mark.asyncio
@pytest.mark.parametrize("prepare_fails", [False, True])
async def test_prepares_all_workers_including_endpoints_before_applying(
    prepare_fails: bool,
) -> None:
    ingress, compute, sink = ServiceId(), ServiceId(), ServiceId()
    variant = ModelVariantId(ModelId(model_name="model"), "variant")
    components = tuple(
        ComponentSolution(
            component_index=index,
            worker_id=worker,
            backend=Backend.CPU,
            layer_keys=(layer,),
            input_layer_keys=("input",) if layer == "input" else (),
            output_layer_keys=("output",) if layer == "output" else (),
            tensor_contract=ComponentTensorContract(inputs, outputs),
            priority=1,
            mean_execution_time_s=0,
            mean_execution_queue_time_s=0,
            mean_transmission_time_s=0,
            transmission_latency_s=0,
        )
        for index, (worker, layer, inputs, outputs) in enumerate(
            (
                (ingress, "input", ("x",), ("x",)),
                (compute, "compute", ("x",), ("y",)),
                (sink, "output", ("y",), ("y",)),
            ),
            start=1,
        )
    )
    solution = OptimizationSolution(
        flow_solutions=(
            FlowOptimizationSolution(
                flow_id=FlowId(user_id=UserId()),
                ingress_worker_id=ingress,
                sink_worker_id=sink,
                model_variant_id=variant,
                components=components,
                cuts=(CutSolution(1, 2, ("x",)), CutSolution(2, 3, ("y",))),
                mean_batch_fill_time_s=0,
                mean_response_time_s=0,
            ),
        ),
        objective_value=0,
        is_optimal=True,
    )
    enriched_solution = EnrichedOptimizationSolution(
        solution, {variant: {"x": TensorProperty("float32", 4, (1,))}}
    )
    partition_gateway = AsyncMock(spec=PartitionBuilderGateway)
    share_gateway = AsyncMock(spec=PlanShareGateway)
    publisher = AsyncMock(spec=FlowPlanPublisher)
    prepared: dict[ServiceId, WorkerPlan] = {}
    applied: set[ServiceId] = set()

    async def prepare(worker_id: ServiceId, plan: WorkerPlan) -> None:
        partition_gateway.build_all_partitions_by_ids.assert_awaited_once_with(
            {ModelPartitionId(model_variant_id=variant, layers=("compute",))}
        )
        assert plan.plan_version == PlanVersion(version_number=1)
        if prepare_fails and worker_id == sink:
            raise RuntimeError("prepare failed")
        prepared[worker_id] = plan

    async def apply(worker_id: ServiceId, version: PlanVersion) -> None:
        assert set(prepared) == {ingress, compute, sink}
        assert version == PlanVersion(version_number=1)
        applied.add(worker_id)

    async def publish(flow_plans: Mapping[FlowId, FlowPlan]) -> None:
        assert applied == {ingress, compute, sink}
        flow_plan = flow_plans[solution.flow_solutions[0].flow_id]
        assert flow_plan.plan_version == PlanVersion(version_number=1)
        assert (
            flow_plan.ingress_replica_id
            == prepared[ingress].deployment_plan.ingress_hops[0]
        )
        assert flow_plan.sink_replica_id == prepared[sink].deployment_plan.sink_hops[0]

    share_gateway.send_prepare_plan_message.side_effect = prepare
    share_gateway.send_apply_plan_message.side_effect = apply
    publisher.publish_flow_plans.side_effect = publish
    manager = DefaultActuationManager(
        solution_observer=Mock(spec=SolutionObserver),
        partition_builder_gateway=partition_gateway,
        plan_version_builder=PlanVersionBuilder(),
        worker_plan_builder=DefaultSolutionToPlanConverter(),
        plan_share_gateway=share_gateway,
        flow_plan_publisher=publisher,
    )

    if prepare_fails:
        with pytest.raises(ExceptionGroup, match="Worker deploy failed") as caught:
            await manager.actuate_solution(enriched_solution)
        assert any(str(error) == "prepare failed" for error in caught.value.exceptions)
        share_gateway.send_apply_plan_message.assert_not_awaited()
        publisher.publish_flow_plans.assert_not_awaited()
    else:
        await manager.actuate_solution(enriched_solution)
        assert applied == {ingress, compute, sink}
        assert prepared[ingress].deployment_plan.executable_hops == ()
        assert prepared[sink].deployment_plan.executable_hops == ()
        assert prepared[ingress].deployment_plan.ingress_hops
        assert prepared[sink].deployment_plan.sink_hops
        publisher.publish_flow_plans.assert_awaited_once()


@pytest.mark.asyncio
async def test_passes_enriched_solution_to_plan_conversion() -> None:
    solution = OptimizationSolution((), 0, True)
    enriched_solution = EnrichedOptimizationSolution(solution, {})
    partition_gateway = AsyncMock(spec=PartitionBuilderGateway)
    converter = AsyncMock(spec=SolutionToPlanConverter)
    converter.convert_solution_to_plans.side_effect = RuntimeError("conversion failed")
    share_gateway = AsyncMock(spec=PlanShareGateway)
    publisher = AsyncMock(spec=FlowPlanPublisher)
    manager = DefaultActuationManager(
        solution_observer=Mock(spec=SolutionObserver),
        partition_builder_gateway=partition_gateway,
        plan_version_builder=PlanVersionBuilder(),
        worker_plan_builder=converter,
        plan_share_gateway=share_gateway,
        flow_plan_publisher=publisher,
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        await manager.actuate_solution(enriched_solution)

    partition_gateway.build_all_partitions_by_ids.assert_awaited_once_with(set())
    converter.convert_solution_to_plans.assert_awaited_once_with(
        PlanVersion(version_number=1), enriched_solution
    )
    share_gateway.send_prepare_plan_message.assert_not_awaited()
    share_gateway.send_apply_plan_message.assert_not_awaited()
    publisher.publish_flow_plans.assert_not_awaited()
