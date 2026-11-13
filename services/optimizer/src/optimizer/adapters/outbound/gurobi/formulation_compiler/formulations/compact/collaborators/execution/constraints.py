from collections.abc import Mapping

import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.other.local_worker_config import (
    ValidLocalWorkerConfig,
)
from shared.model.keys import LayerKey
from shared.service.service import WorkerId


def add_execution_constraints(
    model: gp.Model,
    data: MisocpCompactFormulation,
    *,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
) -> None:
    """Bound execution epigraphs cumulatively and zero unselected candidates."""
    problem = data
    max_components = data.problem.policy.max_components
    p = problem
    for flow in p.flows:
        local = problem.problem.flows[flow].local_worker_config
        for variant in p.configurations_by_flow[flow]:
            topology = problem.problem.variants[variant.model_variant_id]
            layers = topology.layers
            profiles = {
                worker: problem.problem.execution_times_s.get((variant, worker), {})
                for worker in p.problem.flows[flow].workers
            }
            if isinstance(local, ValidLocalWorkerConfig) and (
                not local.is_inference_runner
            ):
                local_profiles = profiles.get(local.worker_id)
                if local_profiles is not None:
                    profiles[local.worker_id] = {
                        layer: profile
                        for layer, profile in local_profiles.items()
                        if layer in topology.input_layers
                        or layer in topology.output_layers
                    }
            minimum = _minimum_execution_time_by_layer(layers, profiles)
            upper_by_worker = {
                worker: sum(float(available.get(layer, 0.0)) for layer in layers)
                for worker, available in profiles.items()
            }
            for component in range(1, max_components + 1):
                for worker in p.problem.flows[flow].workers:
                    available = profiles[worker] or {}
                    placement = assignment.component_worker[flow, component, worker]
                    terms = []
                    upper = upper_by_worker[worker]
                    for layer in layers:
                        x = assignment.layer_component[flow, variant, layer, component]
                        if layer not in available:
                            model.addConstr(
                                x + placement <= 1,
                                name=f"compact_unsupported[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{p.layer_index[variant.model_variant_id, layer]},{component},{p.worker_index[worker]}]",
                            )
                        else:
                            time = float(available[layer])
                            terms.append(time * x)
                    execution_time = execution.execution_time_s[
                        flow, variant, component, worker
                    ]
                    expression = gp.quicksum(terms)
                    execution_time.UB = upper
                    model.addConstr(
                        execution_time >= expression - upper * (1 - placement),
                        name=f"compact_execution[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{component},{p.worker_index[worker]}]",
                    )
                model.addConstr(
                    gp.quicksum(
                        execution.execution_time_s[flow, variant, component, worker]
                        for worker in p.problem.flows[flow].workers
                    )
                    >= gp.quicksum(
                        (
                            time
                            * assignment.layer_component[
                                flow, variant, layer, component
                            ]
                            for layer, time in minimum.items()
                        )
                    ),
                    name=f"execution_minimum[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{component}]",
                )
            for worker in p.problem.flows[flow].workers:
                # Nonnegativity makes every term zero when this candidate is off.
                # Selected candidates share one full-model budget on each worker.
                model.addConstr(
                    gp.quicksum(
                        execution.execution_time_s[flow, variant, component, worker]
                        for component in range(1, max_components + 1)
                    )
                    <= upper_by_worker[worker]
                    * assignment.configuration_selected[flow, variant],
                    name=f"compact_execution_cumulative_upper[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{p.worker_index[worker]}]",
                )


def _minimum_execution_time_by_layer(
    layers: tuple[LayerKey, ...],
    profiles_by_worker: Mapping[WorkerId, Mapping[LayerKey, float] | None],
) -> dict[LayerKey, float]:
    result: dict[LayerKey, float] = {}
    for layer in layers:
        compatible_times = tuple(
            float(profiles[layer])
            for profiles in profiles_by_worker.values()
            if profiles is not None and layer in profiles
        )
        if compatible_times:
            result[layer] = min(compatible_times)
    return result
