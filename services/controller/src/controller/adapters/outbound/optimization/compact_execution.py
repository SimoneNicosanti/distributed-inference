"""Execution epigraph using the worker's backend and relay eligibility rules."""

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    layer_execution_profiles,
)
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    _minimum_execution_time_by_layer,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    ExecutionVariables,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)


def add_compact_execution_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    max_components: int,
) -> None:
    p = problem
    for flow in p.flows:
        local = requests_by_flow[flow].local_worker_config
        # Later aggregate strengthening disabled; retain the original compact.
        # workers = p.workers_by_flow[flow]
        # max_execution_time_by_worker = dict.fromkeys(workers, 0.0)
        # aggregate_execution = {
        #     (component, worker): gp.LinExpr()
        #     for component in range(1, max_components + 1)
        #     for worker in workers
        # }
        for variant in p.models_by_flow[flow]:
            topology = context.static_profiles_by_variant[variant].topology
            layers = topology.layers_in_topological_order()
            profiles = {
                worker: layer_execution_profiles(context, variant, worker)
                for worker in p.workers_by_flow[flow]
            }
            if (
                isinstance(local, ValidLocalWorkerConfig)
                and not local.is_inference_runner
            ):
                local_profiles = profiles.get(local.worker_id)
                if local_profiles is not None:
                    profiles[local.worker_id] = {
                        layer: profile
                        for layer, profile in local_profiles.items()
                        if topology.layers()[layer].is_input
                        or topology.layers()[layer].is_output
                    }
            minimum = _minimum_execution_time_by_layer(layers, profiles)
            for component in range(1, max_components + 1):
                for worker in p.workers_by_flow[flow]:
                    available = profiles[worker] or {}
                    placement = assignment.component_worker[flow, component, worker]
                    terms = []
                    upper = 0.0
                    for layer in layers:
                        x = assignment.layer_component[flow, variant, layer, component]
                        if layer not in available:
                            model.addConstr(
                                x + placement <= 1,
                                name=f"compact_unsupported[{p.flow_index[flow]},{p.model_index[flow, variant]},{p.layer_index[variant, layer]},{component},{p.worker_index[worker]}]",
                            )
                        else:
                            time = float(available[layer].execution_time_s)
                            upper += time
                            terms.append(time * x)
                    execution_time = execution.execution_time_s[
                        flow, variant, component, worker
                    ]
                    expression = gp.quicksum(terms)
                    # aggregate_execution[component, worker] += expression
                    # max_execution_time_by_worker[worker] = max(
                    #     max_execution_time_by_worker[worker], upper
                    # )
                    execution_time.UB = upper
                    model.addConstr(
                        execution_time >= expression - upper * (1 - placement),
                        name=f"compact_execution[{p.flow_index[flow]},{p.model_index[flow, variant]},{component},{p.worker_index[worker]}]",
                    )
                model.addConstr(
                    gp.quicksum(
                        execution.execution_time_s[flow, variant, component, worker]
                        for worker in p.workers_by_flow[flow]
                    )
                    >= gp.quicksum(
                        time
                        * assignment.layer_component[flow, variant, layer, component]
                        for layer, time in minimum.items()
                    ),
                    name=f"execution_minimum[{p.flow_index[flow]},{p.model_index[flow, variant]},{component}]",
                )
        # for component in range(1, max_components + 1):
        #     for worker in workers:
        #         model.addConstr(
        #             gp.quicksum(
        #                 execution.execution_time_s[flow, variant, component, worker]
        #                 for variant in p.models_by_flow[flow]
        #             )
        #             >= aggregate_execution[component, worker]
        #             - max_execution_time_by_worker[worker]
        #             * (1 - assignment.component_worker[flow, component, worker]),
        #             name=f"compact_execution_aggregate[{p.flow_index[flow]},{component},{p.worker_index[worker]}]",
        #         )
