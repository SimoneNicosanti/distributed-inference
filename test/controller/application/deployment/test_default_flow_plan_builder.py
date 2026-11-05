import pytest

from controller.application.deployment.builder.default_flow_plan_builder import (
    DefaultFlowPlanBuilder,
)
from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointReplicaIdBuilder,
)
from controller.domain.backend import Backend
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.plan.front_end_plan import FlowPlan
from controller.domain.profile.model_variant_static_profile import TensorProperty
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId


def _flow(variant: ModelVariantId) -> FlowOptimizationSolution:
    worker_id = ServiceId()
    component = ComponentSolution(
        component_index=1,
        worker_id=worker_id,
        backend=Backend.CPU,
        layer_keys=("compute",),
        input_layer_keys=(),
        output_layer_keys=(),
        tensor_contract=ComponentTensorContract(("input",), ("output",)),
        priority=1,
        mean_execution_time_s=0,
        mean_execution_queue_time_s=0,
        mean_transmission_time_s=0,
        transmission_latency_s=0,
    )
    return FlowOptimizationSolution(
        flow_id=FlowId(user_id=UserId()),
        ingress_worker_id=worker_id,
        sink_worker_id=worker_id,
        model_variant_id=variant,
        components=(component,),
        cuts=(),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )


def _solution() -> EnrichedOptimizationSolution:
    first = ModelVariantId(ModelId("model"), "first")
    second = ModelVariantId(ModelId("model"), "second")
    return EnrichedOptimizationSolution(
        solution=OptimizationSolution(
            (_flow(first), _flow(second), _flow(first)), 0, True
        ),
        input_tensor_properties_by_variant={
            first: {
                "features": TensorProperty("float32", 4, ("batch", 3)),
                "scale": TensorProperty("float32", 4, ()),
            },
            second: {"tokens": TensorProperty("int64", 8, ("batch", "sequence"))},
        },
    )


def test_uses_selected_variant_metadata_and_shared_endpoint_ids_for_each_flow() -> None:
    solution = _solution()
    version = PlanVersion(version_number=7)
    endpoints = EndpointReplicaIdBuilder().build_all(solution.solution)

    plans = DefaultFlowPlanBuilder().build_all(version, solution, endpoints)

    assert set(plans) == {flow.flow_id for flow in solution.solution.flow_solutions}
    for flow in solution.solution.flow_solutions:
        plan = plans[flow.flow_id]
        assert plan.plan_version == version
        assert plan.model_variant_id == flow.model_variant_id
        assert plan.ingress_replica_id is endpoints[flow.flow_id].ingress
        assert plan.sink_replica_id is endpoints[flow.flow_id].sink
        assert (
            plan.input_tensor_properties
            == (solution.input_tensor_properties_by_variant[flow.model_variant_id])
        )
        assert "input" not in plan.input_tensor_properties


@pytest.mark.parametrize("extra_endpoint", [False, True])
def test_rejects_endpoint_mapping_that_does_not_match_the_flows(
    extra_endpoint: bool,
) -> None:
    solution = _solution()
    endpoints = dict(EndpointReplicaIdBuilder().build_all(solution.solution))
    if extra_endpoint:
        endpoints[FlowId(user_id=UserId())] = next(iter(endpoints.values()))
    else:
        endpoints.pop(next(iter(endpoints)))

    with pytest.raises(ValueError, match="Endpoint ids must match the flows"):
        DefaultFlowPlanBuilder().build_all(
            PlanVersion(version_number=1), solution, endpoints
        )


def test_empty_solution_produces_no_flow_plans() -> None:
    solution = EnrichedOptimizationSolution(OptimizationSolution((), 0, True), {})

    assert (
        DefaultFlowPlanBuilder().build_all(PlanVersion(version_number=1), solution, {})
        == {}
    )


@pytest.mark.parametrize("invalid_endpoint", ["ingress", "sink"])
def test_flow_plan_requires_consistent_variant_identity(invalid_endpoint: str) -> None:
    variant = ModelVariantId(ModelId("model"), "variant")
    other = ModelVariantId(ModelId("other-model"), "variant")

    with pytest.raises(ValueError, match="must be consistent across ingress and sink"):
        FlowPlan(
            plan_version=PlanVersion(version_number=1),
            model_variant_id=variant,
            input_tensor_properties={"x": TensorProperty("float32", 4, (1,))},
            ingress_replica_id=PlannedIngressReplicaId(
                other if invalid_endpoint == "ingress" else variant, 0
            ),
            sink_replica_id=PlannedSinkReplicaId(
                other if invalid_endpoint == "sink" else variant, 0
            ),
        )


def test_flow_plan_keeps_an_independent_input_mapping_and_endpoint_indexes() -> None:
    variant = ModelVariantId(ModelId("model"), "variant")
    property_ = TensorProperty("float32", 4, ("batch", 0, 3))
    inputs = {"x": property_}
    plan = FlowPlan(
        plan_version=PlanVersion(version_number=1),
        model_variant_id=variant,
        input_tensor_properties=inputs,
        ingress_replica_id=PlannedIngressReplicaId(variant, 2),
        sink_replica_id=PlannedSinkReplicaId(variant, 5),
    )

    inputs.clear()

    assert plan.input_tensor_properties == {"x": property_}
    assert plan.ingress_replica_id.replica_idx == 2
    assert plan.sink_replica_id.replica_idx == 5


def test_flow_plan_rejects_missing_input_properties() -> None:
    variant = ModelVariantId(ModelId("model"), "variant")

    with pytest.raises(ValueError, match="must contain model input tensor properties"):
        FlowPlan(
            plan_version=PlanVersion(version_number=1),
            model_variant_id=variant,
            input_tensor_properties={},
            ingress_replica_id=PlannedIngressReplicaId(variant, 0),
            sink_replica_id=PlannedSinkReplicaId(variant, 0),
        )
