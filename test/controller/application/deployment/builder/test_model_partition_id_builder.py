import pytest

from controller.application.deployment.builder.model_partition_id_builder import (
    ModelPartitionIdBuilder,
)
from controller.domain.optimization.optimization_solution import (
    ComponentSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.profile.model_variant_topology import ComponentTensorContract
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from shared.user.user import UserId


def _component(
    index: int,
    layer_keys: tuple[str, ...],
    *,
    worker_id: ServiceId | None = None,
    input_layer_keys: tuple[str, ...] = (),
    output_layer_keys: tuple[str, ...] = (),
) -> ComponentSolution:
    return ComponentSolution(
        component_index=index,
        worker_id=worker_id or ServiceId(),
        layer_keys=layer_keys,
        input_layer_keys=input_layer_keys,
        output_layer_keys=output_layer_keys,
        tensor_contract=ComponentTensorContract(
            input_tensor_keys=("input",),
            output_tensor_keys=("output",),
        ),
        priority=1,
        mean_execution_time_s=0,
        mean_execution_queue_time_s=0,
        mean_transmission_time_s=0,
        transmission_latency_s=0,
    )


def test_builds_partitions_only_from_executable_layers() -> None:
    user_id = UserId()
    model_variant_id = ModelVariantId(
        model_id=ModelId(model_name="model"),
        variant_tag="variant",
    )
    ingress_id = ServiceId()
    components = (
        _component(
            1,
            ("input",),
            worker_id=ingress_id,
            input_layer_keys=("input",),
        ),
        _component(2, ("compute",)),
        _component(3, ("output",), output_layer_keys=("output",)),
    )
    solution = OptimizationSolution(
        flow_solutions=(
            FlowOptimizationSolution(
                flow_id=FlowId(user_id=user_id),
                ingress_id=ingress_id,
                model_variant_id=model_variant_id,
                components=components,
                cuts=(),
                mean_batch_fill_time_s=0,
                mean_response_time_s=0,
            ),
        ),
        objective_value=0,
        is_optimal=True,
    )

    partition_ids = ModelPartitionIdBuilder.build_all_model_partition_ids_from_solution(
        solution
    )

    assert partition_ids == {
        ModelPartitionId(
            model_variant_id=model_variant_id,
            layers=("compute",),
        )
    }
    assert tuple(component.executable_layer_keys for component in components) == (
        (),
        ("compute",),
        (),
    )


def test_rejects_boundary_layer_outside_component() -> None:
    with pytest.raises(
        ValueError,
        match="Component boundary layers must belong to the component",
    ):
        _component(1, ("compute",), input_layer_keys=("input",))
