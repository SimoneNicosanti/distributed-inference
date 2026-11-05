"""Resident weights plus one maximum-cut activation epigraph per CPU server."""

import gurobipy as gp

from controller.adapters.outbound.optimization.compact_transmission import cut_upper_mb
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    layer_weight_size_bytes,
)
from controller.adapters.outbound.optimization.gurobi_units import bytes_to_megabytes
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    MemoryVariables,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)


def add_compact_memory_variables(
    model: gp.Model,
    problem: GurobiProblem,
    context: OptimizationContext,
    *,
    max_components: int,
) -> MemoryVariables:
    weights = {}
    for flow in problem.flows:
        for variant in problem.models_by_flow[flow]:
            for component in range(1, max_components + 1):
                for worker in problem.workers_by_flow[flow]:
                    weights[flow, variant, component, worker] = model.addVar(
                        lb=0,
                        ub=bytes_to_megabytes(
                            context.resource_profiles_by_worker[
                                worker
                            ].ram_profile.available_bytes
                        ),
                        name=f"mem_weight[{problem.flow_index[flow]},{problem.model_index[flow, variant]},{component},{problem.worker_index[worker]}]",
                    )
    return MemoryVariables(
        component_weight_mb=weights,
        max_activation_mb={
            worker: model.addVar(
                lb=0,
                ub=bytes_to_megabytes(
                    context.resource_profiles_by_worker[
                        worker
                    ].ram_profile.available_bytes
                ),
                name=f"max_activation[{problem.worker_index[worker]}]",
            )
            for worker in problem.workers
        },
    )


def add_compact_memory_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    memory: MemoryVariables,
    *,
    context: OptimizationContext,
    max_components: int,
) -> None:
    p = problem
    for flow in p.flows:
        for variant in p.models_by_flow[flow]:
            profile = context.static_profiles_by_variant[variant]
            weights = {
                layer: bytes_to_megabytes(layer_weight_size_bytes(profile, layer))
                for layer in profile.topology.layers()
            }
            upper = sum(weights.values())
            activation = cut_upper_mb(profile)
            for component in range(1, max_components + 1):
                expression = gp.quicksum(
                    weight * assignment.layer_component[flow, variant, layer, component]
                    for layer, weight in weights.items()
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
                weight
                for (*_, host), weight in memory.component_weight_mb.items()
                if host == worker
            )
            + memory.max_activation_mb[worker]
            <= bytes_to_megabytes(
                context.resource_profiles_by_worker[worker].ram_profile.available_bytes
            ),
            name=f"memory_capacity[{p.worker_index[worker]}]",
        )
