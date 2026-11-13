import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.memory.memory_variables import (
    MemoryVariables,
)
from optimizer.adapters.outbound.gurobi.gurobi_units import (
    bytes_to_megabytes,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)


def add_memory_constraints(
    model: gp.Model,
    data: MisocpCompactFormulation,
    *,
    assignment: AssignmentVariables,
    memory: MemoryVariables,
) -> None:
    problem = data
    max_components = data.problem.policy.max_components
    p = problem
    for flow in p.flows:
        for variant in p.configurations_by_flow[flow]:
            profile = problem.problem.variants[variant.model_variant_id]
            weights = {
                layer: bytes_to_megabytes(profile.weight_bytes_by_layer[layer])
                for layer in profile.layers
            }
            upper = sum(weights.values())
            activation = bytes_to_megabytes(
                problem.problem.configurations[variant].max_cut_size_bytes
            )
            for component in range(1, max_components + 1):
                expression = gp.quicksum(
                    (
                        weight
                        * assignment.layer_component[flow, variant, layer, component]
                        for layer, weight in weights.items()
                    )
                )
                for worker in p.problem.flows[flow].workers:
                    e = assignment.component_worker[flow, component, worker]
                    suffix = f"{p.flow_index[flow]},{p.configuration_index[flow, variant]},{component},{p.worker_index[worker]}"
                    model.addConstr(
                        memory.component_weight_mb[flow, variant, component, worker]
                        >= expression - upper * (1 - e),
                        name=f"compact_weight[{suffix}]",
                    )
                    model.addConstr(
                        memory.max_activation_mb[worker]
                        >= activation
                        * (assignment.configuration_selected[flow, variant] + e - 1),
                        name=f"compact_activation[{suffix}]",
                    )
    for worker in p.workers:
        model.addConstr(
            gp.quicksum(
                (
                    weight
                    for (*_, host), weight in memory.component_weight_mb.items()
                    if host == worker
                )
            )
            + memory.max_activation_mb[worker]
            <= bytes_to_megabytes(
                problem.problem.workers[worker].available_inference_memory_bytes
            ),
            name=f"memory_capacity[{p.worker_index[worker]}]",
        )
