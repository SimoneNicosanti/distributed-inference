from controller.domain.optimization.optimization_solution import (
    ComponentSolution,
    FlowOptimizationSolution,
)
from shared.model.keys import TensorKey


def executable_components(
    flow_solution: FlowOptimizationSolution,
) -> tuple[ComponentSolution, ...]:
    component_indexes = [
        component.component_index for component in flow_solution.components
    ]
    if component_indexes != sorted(component_indexes):
        raise ValueError("Flow components must be ordered by component index")
    if len(component_indexes) != len(set(component_indexes)):
        raise ValueError("Flow component indexes must be unique")

    executable = tuple(
        component
        for component in flow_solution.components
        if component.executable_layer_keys
    )
    if not executable:
        raise ValueError(f"Flow {flow_solution.flow_id} has no executable components")
    return executable


def input_tensors(
    flow_solution: FlowOptimizationSolution,
    first_executable: ComponentSolution,
) -> tuple[TensorKey, ...]:
    position = flow_solution.components.index(first_executable)
    if position == 0:
        return first_executable.tensor_contract.input_tensor_keys
    return cut_tensors(
        flow_solution,
        flow_solution.components[position - 1],
        first_executable,
    )


def result_tensors(
    flow_solution: FlowOptimizationSolution,
    last_executable: ComponentSolution,
) -> tuple[TensorKey, ...]:
    position = flow_solution.components.index(last_executable)
    if position + 1 == len(flow_solution.components):
        return last_executable.tensor_contract.output_tensor_keys
    return cut_tensors(
        flow_solution,
        last_executable,
        flow_solution.components[position + 1],
    )


def tensors_between(
    flow_solution: FlowOptimizationSolution,
    source: ComponentSolution,
    destination: ComponentSolution,
) -> tuple[TensorKey, ...]:
    source_position = flow_solution.components.index(source)
    destination_position = flow_solution.components.index(destination)
    cuts = tuple(
        cut_tensors(
            flow_solution,
            flow_solution.components[position],
            flow_solution.components[position + 1],
        )
        for position in range(source_position, destination_position)
    )
    if any(tensors != cuts[0] for tensors in cuts[1:]):
        raise ValueError(
            "Cannot collapse a non-executable component that changes routed tensors"
        )
    return cuts[0]


def cut_tensors(
    flow_solution: FlowOptimizationSolution,
    source: ComponentSolution,
    destination: ComponentSolution,
) -> tuple[TensorKey, ...]:
    matching = tuple(
        cut
        for cut in flow_solution.cuts
        if cut.source_component_index == source.component_index
        and cut.destination_component_index == destination.component_index
    )
    if len(matching) != 1:
        raise ValueError("Each adjacent component pair must have exactly one cut")
    return matching[0].tensor_keys
