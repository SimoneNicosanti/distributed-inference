from collections.abc import Mapping

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
)
from shared.model.keys import LayerKey
from shared.service.service import WorkerId


def add_execution_constraints(
    model: gp.Model,
    data: CompactFormulationData,
    *,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
) -> None:
    problem = data
    max_components = data.max_components
    p = problem
    for flow in p.flows:
        local = problem.local_worker_config_by_flow[flow]
        for variant in p.models_by_flow[flow]:
            topology = problem.static_profiles_by_variant[variant].topology
            layers = topology.layers_in_topological_order()
            profiles = {
                worker: problem.layer_execution_profiles_by_model_worker[
                    variant, worker
                ]
                for worker in p.workers_by_flow[flow]
            }
            if isinstance(local, ValidLocalWorkerConfig) and (
                not local.is_inference_runner
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
                        (
                            time
                            * assignment.layer_component[
                                flow, variant, layer, component
                            ]
                            for layer, time in minimum.items()
                        )
                    ),
                    name=f"execution_minimum[{p.flow_index[flow]},{p.model_index[flow, variant]},{component}]",
                )


def _minimum_execution_time_by_layer(
    layers: tuple[LayerKey, ...],
    profiles_by_worker: dict[WorkerId, Mapping[LayerKey, LayerExecutionProfile] | None],
) -> dict[LayerKey, float]:
    result: dict[LayerKey, float] = {}
    for layer in layers:
        compatible_times = tuple(
            float(profiles[layer].execution_time_s)
            for profiles in profiles_by_worker.values()
            if profiles is not None and layer in profiles
        )
        if compatible_times:
            result[layer] = min(compatible_times)
    return result
