import pytest

from controller.application.deployment.default_service_plan_builder import (
    DefaultServicePlanBuilder,
)
from controller.domain.optimization.optimization_solution import (
    ComponentSolution,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.profile.model_variant_topology import ComponentTensorContract
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId


def _component(
    index: int,
    worker_id: ServiceId,
    layers: tuple[str, ...],
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    *,
    input_layers: tuple[str, ...] = (),
    output_layers: tuple[str, ...] = (),
    priority: int = 1,
) -> ComponentSolution:
    return ComponentSolution(
        component_index=index,
        worker_id=worker_id,
        layer_keys=layers,
        input_layer_keys=input_layers,
        output_layer_keys=output_layers,
        tensor_contract=ComponentTensorContract(
            input_tensor_keys=inputs,
            output_tensor_keys=outputs,
        ),
        priority=priority,
        mean_execution_time_s=0,
        mean_execution_queue_time_s=0,
        mean_transmission_time_s=0,
        transmission_latency_s=0,
    )


@pytest.mark.asyncio
async def test_maps_solution_to_all_involved_workers() -> None:
    user_id = UserId()
    model_variant_id = ModelVariantId(
        model_id=ModelId(owner_id=user_id, model_name="model"),
        variant_tag="variant",
    )
    ingress_one = ServiceId()
    ingress_two = ServiceId()
    compute_one = ServiceId()
    compute_two = ServiceId()
    first_flow_id = FlowId(user_id=user_id)
    second_flow_id = FlowId(user_id=user_id)

    first_flow = FlowOptimizationSolution(
        flow_id=first_flow_id,
        ingress_id=ingress_one,
        model_variant_id=model_variant_id,
        components=(
            _component(
                1,
                ingress_one,
                ("input",),
                ("x",),
                ("x",),
                input_layers=("input",),
            ),
            _component(2, compute_one, ("encoder",), ("x",), ("hidden",)),
            _component(
                3,
                compute_two,
                ("head",),
                ("hidden",),
                ("result",),
                priority=2,
            ),
            _component(
                4,
                ingress_one,
                ("output",),
                ("result",),
                ("result",),
                output_layers=("output",),
            ),
        ),
        cuts=(
            CutSolution(
                source_component_index=1,
                destination_component_index=2,
                tensor_keys=("x",),
            ),
            CutSolution(
                source_component_index=2,
                destination_component_index=3,
                tensor_keys=("hidden",),
            ),
            CutSolution(
                source_component_index=3,
                destination_component_index=4,
                tensor_keys=("result",),
            ),
        ),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )
    second_flow = FlowOptimizationSolution(
        flow_id=second_flow_id,
        ingress_id=ingress_two,
        model_variant_id=model_variant_id,
        components=(
            _component(
                1,
                compute_one,
                ("input", "encoder", "output"),
                ("x",),
                ("hidden",),
                input_layers=("input",),
                output_layers=("output",),
                priority=3,
            ),
        ),
        cuts=(),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )
    solution = OptimizationSolution(
        flow_solutions=(first_flow, second_flow),
        objective_value=0,
        is_optimal=True,
    )

    plans = await DefaultServicePlanBuilder().build_all_plans_from_solution(
        PlanVersion(version_number=1), solution
    )

    assert set(plans) == {ingress_one, ingress_two, compute_one, compute_two}
    for ingress_id in (ingress_one, ingress_two):
        assert plans[ingress_id].ingress_plan is not None
        assert plans[ingress_id].worker_plan.deployment_plan.replica_ids == ()

    compute_one_plan = plans[compute_one].worker_plan
    encoder_replicas = compute_one_plan.deployment_plan.replica_ids
    assert tuple(replica.replica_idx for replica in encoder_replicas) == (0, 1)
    assert compute_one_plan.partition_execution_plan.partition_ids == (
        encoder_replicas[0].partition_id,
    )
    assert compute_one_plan.scheduling_plan.priorities == (1, 3)
    assert plans[compute_one].ingress_plan is None

    compute_two_plan = plans[compute_two].worker_plan
    head_replica = compute_two_plan.deployment_plan.replica_ids[0]
    first_encoder_replica = encoder_replicas[0]
    assert (
        compute_one_plan.routing_plan.routings[0].output_routes[0].next_replica_id
        == head_replica
    )
    assert (
        compute_two_plan.routing_plan.routings[0].input_bindings[0].prev_replica_id
        == first_encoder_replica
    )
    assert compute_two_plan.routing_plan.routings[0].output_routes[0].tensors == (
        "result",
    )

    first_ingress_plan = plans[ingress_one].ingress_plan
    assert first_ingress_plan is not None
    assert first_ingress_plan.flow_ids == (first_flow_id,)
    assert first_ingress_plan.flow_plans[0].input_routes[0].tensors == ("x",)
    assert first_ingress_plan.flow_plans[0].result_tensors == ("result",)


@pytest.mark.asyncio
async def test_rejects_flow_without_executable_components() -> None:
    user_id = UserId()
    ingress_id = ServiceId()
    model_variant_id = ModelVariantId(
        model_id=ModelId(owner_id=user_id, model_name="model"),
        variant_tag="variant",
    )
    solution = OptimizationSolution(
        flow_solutions=(
            FlowOptimizationSolution(
                flow_id=FlowId(user_id=user_id),
                ingress_id=ingress_id,
                model_variant_id=model_variant_id,
                components=(
                    _component(
                        1,
                        ingress_id,
                        ("input",),
                        ("x",),
                        ("x",),
                        input_layers=("input",),
                    ),
                ),
                cuts=(),
                mean_batch_fill_time_s=0,
                mean_response_time_s=0,
            ),
        ),
        objective_value=0,
        is_optimal=True,
    )

    with pytest.raises(ValueError, match="has no executable components"):
        await DefaultServicePlanBuilder().build_all_plans_from_solution(
            PlanVersion(version_number=1), solution
        )
