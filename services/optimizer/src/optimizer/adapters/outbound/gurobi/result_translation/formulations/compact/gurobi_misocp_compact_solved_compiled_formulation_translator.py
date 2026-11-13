from itertools import pairwise
from typing import override

import gurobipy as gp
from gurobipy import GRB

from integration.shared.id_json import flow_id_to_json
from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.expressions import (
    GurobiAffineTerm,
    tensor_crossing_term,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedCompiledFormulationTranslator,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from optimizer.domain.problem.other.flow_endpoint_placement import (
    FixedPlacement,
)
from optimizer.domain.problem.other.local_worker_config import (
    ValidLocalWorkerConfig,
)
from optimizer.domain.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from shared.flow.flow import FlowId


class GurobiSolutionError(RuntimeError):
    pass


def _nonnegative(value: float) -> float:
    return max(0.0, value)


def _flow_key(flow: FlowId) -> str:
    return flow_id_to_json(flow)


def _solution_value(term: GurobiAffineTerm) -> float:
    return float(term.X if isinstance(term, gp.Var) else term.getValue())


def _validate_flow_placements(
    data: MisocpCompactFormulation,
    flow: FlowId,
    components: list[ComponentSolution],
) -> None:
    if not components:
        raise GurobiSolutionError(
            f"Expected at least one component for flow {_flow_key(flow)}"
        )

    placement = data.problem.flows[flow].endpoint_placement
    for endpoint, endpoint_placement, component in (
        ("ingress", placement.ingress_placement, components[0]),
        ("sink", placement.sink_placement, components[-1]),
    ):
        if (
            isinstance(endpoint_placement, FixedPlacement)
            and component.worker_id != endpoint_placement.worker_id
        ):
            raise GurobiSolutionError(
                f"Fixed {endpoint} placement violated for flow {_flow_key(flow)}"
            )

    local_worker = data.problem.flows[flow].local_worker_config
    if (
        isinstance(local_worker, ValidLocalWorkerConfig)
        and not local_worker.is_inference_runner
        and any(
            component.worker_id == local_worker.worker_id
            and component.executable_layer_keys
            for component in components
        )
    ):
        raise GurobiSolutionError(
            f"Relay-only worker executes computational layers for flow "
            f"{_flow_key(flow)}"
        )


class GurobiMisocpCompactSolvedCompiledFormulationTranslator(
    SolvedCompiledFormulationTranslator[
        GurobiCompiledFormulation[MisocpCompactFormulation, CompactVariables]
    ]
):
    @override
    async def translate_solved_formulation_model(
        self,
        solved_compiled_formulation: GurobiCompiledFormulation[
            MisocpCompactFormulation, CompactVariables
        ],
    ) -> OptimizationSolution:
        model = solved_compiled_formulation.native_model
        data = solved_compiled_formulation.formulation
        variables = solved_compiled_formulation.variables
        return OptimizationSolution(
            flow_solutions=tuple(
                self._translate_flow(data, variables, flow) for flow in data.flows
            ),
            objective_value=model.ObjVal,
            is_optimal=model.Status == GRB.OPTIMAL,
        )

    def _translate_flow(
        self,
        data: MisocpCompactFormulation,
        variables: CompactVariables,
        flow: FlowId,
    ) -> FlowOptimizationSolution:
        assignment = variables.assignment
        network = variables.network
        selected_models = tuple(
            model_variant
            for model_variant in data.configurations_by_flow[flow]
            if assignment.configuration_selected[flow, model_variant].X > 0.5
        )
        if len(selected_models) != 1:
            raise GurobiSolutionError(
                f"Expected one selected model for flow {_flow_key(flow)}"
            )
        selected_model = selected_models[0]
        tensor_routes = data.tensor_routes_by_variant[selected_model.model_variant_id]
        components = self._translate_components(data, variables, flow, selected_model)
        _validate_flow_placements(data, flow, components)

        cuts = tuple(
            CutSolution(
                source_component_index=source.component_index,
                destination_component_index=destination.component_index,
                tensor_keys=tuple(
                    tensor
                    for tensor, route in tensor_routes.items()
                    if _solution_value(
                        tensor_crossing_term(
                            assignment,
                            network,
                            flow=flow,
                            model_variant=selected_model,
                            tensor=tensor,
                            route=route,
                            cut=source.component_index,
                        )
                    )
                    > 0.5
                ),
            )
            for source, destination in pairwise(components)
        )

        return FlowOptimizationSolution(
            flow_id=flow,
            ingress_worker_id=components[0].worker_id,
            sink_worker_id=components[-1].worker_id,
            model_variant_id=selected_model.model_variant_id,
            input_shapes=selected_model.input_shapes.input_shapes,
            components=tuple(components),
            cuts=cuts,
            mean_batch_fill_time_s=data.batch_fill_time_s[flow, selected_model],
            # Execution epigraphs can have slack. Report a total consistent with
            # the recomputed execution and the retained queue/network estimates.
            mean_response_time_s=data.batch_fill_time_s[flow, selected_model]
            + sum(
                component.mean_execution_time_s
                + component.mean_execution_queue_time_s
                + component.mean_transmission_time_s
                + component.transmission_latency_s
                for component in components
            ),
        )

    @staticmethod
    def _translate_components(
        data: MisocpCompactFormulation,
        variables: CompactVariables,
        flow: FlowId,
        selected_model: ModelConfigurationKey,
    ) -> list[ComponentSolution]:
        assignment = variables.assignment
        compute_queue = variables.compute_queue
        network = variables.network
        topology = data.problem.variants[selected_model.model_variant_id]
        connections = data.connections_by_pair
        workers = data.problem.flows[flow].workers
        successors_by_worker = {
            source: data.successors_by_worker[source] for source in workers
        }
        components: list[ComponentSolution] = []

        for component in range(1, data.problem.policy.max_components + 1):
            if assignment.component_active[flow, component].X <= 0.5:
                continue
            selected_workers = tuple(
                worker
                for worker in data.problem.flows[flow].workers
                if assignment.component_worker[flow, component, worker].X > 0.5
            )
            if len(selected_workers) != 1:
                raise GurobiSolutionError(
                    f"Expected one worker for flow {_flow_key(flow)}, "
                    f"component {component}"
                )
            worker = selected_workers[0]
            selected_priorities = tuple(
                priority
                for priority in range(1, data.problem.policy.priority_count + 1)
                if compute_queue.component_priority[flow, component, worker, priority].X
                > 0.5
            )
            if len(selected_priorities) != 1:
                raise GurobiSolutionError(
                    f"Expected one priority for flow {_flow_key(flow)}, "
                    f"component {component}"
                )
            layers = tuple(
                layer
                for layer in topology.layers
                if assignment.layer_component[flow, selected_model, layer, component].X
                > 0.5
            )
            transmission_response = (
                network.component_transmission_response_s[flow, component].X
                if component < data.problem.policy.max_components
                else 0.0
            )
            transmission_latency = (
                sum(
                    (
                        0.0
                        if source == target
                        else connections[source, target].round_trip_time_s / 2
                    )
                    * network.adjacent_component_workers[
                        flow, component, source, target
                    ].X
                    for source in workers
                    for target in workers
                    if source == target or target in successors_by_worker[source]
                )
                if component < data.problem.policy.max_components
                else 0.0
            )
            components.append(
                ComponentSolution(
                    component_index=component,
                    worker_id=worker,
                    backend=data.problem.workers[worker].backend,
                    layer_keys=layers,
                    input_layer_keys=tuple(
                        layer for layer in layers if layer in topology.input_layers
                    ),
                    output_layer_keys=tuple(
                        layer for layer in layers if layer in topology.output_layers
                    ),
                    tensor_contract=ComponentTensorContract(
                        input_tensor_keys=tuple(
                            {
                                tensor
                                for edge in topology.edges
                                if (edge.target in layers and edge.source not in layers)
                                or (
                                    edge.source in layers
                                    and edge.source in topology.input_layers
                                )
                                for tensor in edge.tensors
                            }
                        ),
                        output_tensor_keys=tuple(
                            {
                                tensor
                                for edge in topology.edges
                                if (edge.source in layers and edge.target not in layers)
                                or (
                                    edge.target in layers
                                    and edge.target in topology.output_layers
                                )
                                for tensor in edge.tensors
                            }
                        ),
                    ),
                    priority=selected_priorities[0],
                    mean_execution_time_s=sum(
                        data.problem.execution_times_s[selected_model, worker][layer]
                        for layer in layers
                    ),
                    mean_execution_queue_time_s=_nonnegative(
                        compute_queue.component_queue_wait_s[flow, component].X
                    ),
                    mean_transmission_time_s=_nonnegative(transmission_response),
                    transmission_latency_s=_nonnegative(transmission_latency),
                )
            )
        return components
