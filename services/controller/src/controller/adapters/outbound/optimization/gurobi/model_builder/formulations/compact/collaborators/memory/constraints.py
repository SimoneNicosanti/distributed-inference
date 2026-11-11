import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.memory.memory_variables import (
    MemoryVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.profile_bounds import (
    cut_upper_mb,
)
from controller.adapters.outbound.optimization.gurobi.gurobi_units import (
    bytes_to_megabytes,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.application.optimization.formulation.compact.profile_data import (
    layer_weight_size_bytes,
)


def add_memory_constraints(
    model: gp.Model,
    data: CompactFormulationData,
    *,
    assignment: AssignmentVariables,
    memory: MemoryVariables,
) -> None:
    problem = data
    max_components = data.max_components
    p = problem
    for flow in p.flows:
        for variant in p.models_by_flow[flow]:
            profile = problem.static_profiles_by_variant[variant]
            weights = {
                layer: bytes_to_megabytes(layer_weight_size_bytes(profile, layer))
                for layer in profile.topology.layers()
            }
            upper = sum(weights.values())
            activation = cut_upper_mb(profile)
            for component in range(1, max_components + 1):
                expression = gp.quicksum(
                    (
                        weight
                        * assignment.layer_component[flow, variant, layer, component]
                        for layer, weight in weights.items()
                    )
                )
                for worker in p.workers_by_flow[flow]:
                    e = assignment.component_worker[flow, component, worker]
                    suffix = f"{p.flow_index[flow]},{p.model_index[flow, variant]},{component},{p.worker_index[worker]}"
                    model.addConstr(
                        memory.component_weight_mb[flow, variant, component, worker]
                        >= expression - upper * (1 - e),
                        name=f"compact_weight[{suffix}]",
                    )
                    model.addConstr(
                        memory.max_activation_mb[worker]
                        >= activation
                        * (assignment.model_selected[flow, variant] + e - 1),
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
                problem.resource_profiles_by_worker[
                    worker
                ].available_inference_memory_bytes
            ),
            name=f"memory_capacity[{p.worker_index[worker]}]",
        )
