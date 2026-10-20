import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    layer_activation_sizes_bytes,
    layer_weight_size_bytes,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    MemoryVariables,
)
from controller.domain.optimization.optimization_context import OptimizationContext


def add_memory_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    memory: MemoryVariables,
    *,
    context: OptimizationContext,
    max_components: int,
) -> None:
    p = problem
    selected_models = {
        model_variant
        for models in p.models_by_flow.values()
        for model_variant in models
    }
    layers_by_model = {
        model_variant: context.static_profiles_by_variant[
            model_variant
        ].topology.layers_in_topological_order()
        for model_variant in selected_models
    }
    weights_by_layer = {
        (model_variant, layer): layer_weight_size_bytes(
            context.static_profiles_by_variant[model_variant], layer
        )
        for model_variant in selected_models
        for layer in layers_by_model[model_variant]
    }
    activations_by_layer = {
        (model_variant, layer): size
        for model_variant in selected_models
        for layer, size in layer_activation_sizes_bytes(
            context.static_profiles_by_variant[model_variant]
        ).items()
    }
    for worker in p.workers:
        worker_index = p.worker_index[worker]
        relevant_flows = tuple(
            flow for flow in p.flows if worker in p.workers_by_flow[flow]
        )
        weights = gp.quicksum(
            weights_by_layer[model_variant, layer]
            * assignment.layer_component_worker[
                flow, model_variant, layer, component, worker
            ]
            for flow in relevant_flows
            for model_variant in p.models_by_flow[flow]
            for layer in layers_by_model[model_variant]
            for component in range(1, max_components + 1)
        )
        for flow in relevant_flows:
            for model_variant in p.models_by_flow[flow]:
                for component in range(1, max_components + 1):
                    activations = gp.quicksum(
                        activations_by_layer[model_variant, layer]
                        * assignment.layer_component_worker[
                            flow, model_variant, layer, component, worker
                        ]
                        for layer in layers_by_model[model_variant]
                    )
                    model.addConstr(
                        memory.max_activation_bytes[worker] >= activations,
                        name=(
                            f"max_activation[{worker_index},{p.flow_index[flow]},"
                            f"{p.model_index[flow, model_variant]},{component}]"
                        ),
                    )
        model.addConstr(
            weights + memory.max_activation_bytes[worker]
            <= context.resource_profiles_by_worker[worker].ram_profile.available_bytes,
            name=f"memory_capacity[{worker_index}]",
        )
