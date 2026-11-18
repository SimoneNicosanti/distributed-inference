import gurobipy as gp

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


def compact_solution_decisions(
    formulation_model: GurobiCompiledFormulation[
        MisocpCompactFormulation, CompactVariables
    ],
    warm_start_solution: OptimizationSolution,
    *,
    allow_partial: bool,
) -> tuple[tuple[gp.Var, float], ...]:
    """Validate completely before mutation; map only seed flows using domain IDs."""
    data = formulation_model.formulation
    assignment = formulation_model.variables.assignment
    compute_queue = formulation_model.variables.compute_queue
    flow_solutions = warm_start_solution.flow_solutions
    seed_flows = {solution.flow_id for solution in flow_solutions}
    if not seed_flows or not seed_flows <= set(data.flows):
        raise ValueError("Warm start flows must be a nonempty subset of target flows")
    if not allow_partial and seed_flows != set(data.flows):
        raise ValueError("Warm start must contain exactly the target model flows")

    decision_variables = [
        variable
        for group in (
            assignment.configuration_selected,
            assignment.component_worker,
            assignment.component_active,
            assignment.component_last,
            compute_queue.component_priority,
        )
        for key, variable in group.items()
        if key[0] in seed_flows
    ]
    decision_variables.extend(
        variable
        for (flow, _, _, _), variable in assignment.layer_decision_items()
        if flow in seed_flows
    )
    selected_variables: list[gp.Var] = []
    for solution in flow_solutions:
        flow = solution.flow_id
        candidates = tuple(
            configuration
            for configuration in data.configurations_by_flow[flow]
            if configuration.model_variant_id == solution.model_variant_id
            and (
                solution.input_shapes is None
                or configuration.input_shapes.input_shapes == solution.input_shapes
            )
        )
        if len(candidates) != 1:
            raise ValueError("Warm start must identify one target configuration")
        variant = candidates[0]
        if (flow, variant) not in assignment.configuration_selected:
            raise ValueError("Warm start model is not available in the target")

        components = solution.components
        if (
            not components
            or len(components) > data.problem.policy.max_components
            or tuple(component.component_index for component in components)
            != tuple(range(1, len(components) + 1))
        ):
            raise ValueError("Warm start components must form an available prefix")

        selected_variables.append(assignment.configuration_selected[flow, variant])
        expected_layers = set(data.problem.variants[variant.model_variant_id].layers)
        assigned_layers: set[str] = set()
        for component in components:
            index = component.component_index
            worker_key = flow, index, component.worker_id
            priority_key = (*worker_key, component.priority)
            if worker_key not in assignment.component_worker:
                raise ValueError("Warm start worker is not available in the target")
            if priority_key not in compute_queue.component_priority:
                raise ValueError("Warm start priority is not available in the target")

            selected_variables.extend(
                (
                    assignment.component_active[flow, index],
                    assignment.component_worker[worker_key],
                    compute_queue.component_priority[priority_key],
                )
            )
            for layer in component.layer_keys:
                if layer not in expected_layers or layer in assigned_layers:
                    raise ValueError("Warm start contains unknown or repeated layers")
                assigned_layers.add(layer)
                selected_variables.extend(
                    assignment.layer_start_variables(flow, variant, layer, index)
                )
        if assigned_layers != expected_layers:
            raise ValueError("Warm start must assign every selected model layer")
        selected_variables.append(assignment.component_last[flow, len(components)])

    selected_indices = {variable.index for variable in selected_variables}
    decisions = {variable.index: variable for variable in decision_variables}
    return tuple(
        (variable, float(index in selected_indices))
        for index, variable in decisions.items()
    )
