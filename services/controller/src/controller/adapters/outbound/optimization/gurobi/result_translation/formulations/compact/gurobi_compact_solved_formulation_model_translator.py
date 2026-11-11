from itertools import pairwise
from typing import override

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_formulation_model import (
    GurobiFormulationModel,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.network.expressions import (
    GurobiAffineTerm,
    tensor_crossing_term,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.compact_variables import (
    CompactVariables,
)
from controller.application.ports.outbound.optimization.solved_formulation_model_translator import (
    SolvedFormulationModelTranslator,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    FixedPlacement,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from integration.shared.id_json import flow_id_to_json
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId


class GurobiSolutionError(RuntimeError):
    pass


def _nonnegative(value: float) -> float:
    return max(0.0, value)


def _flow_key(flow: FlowId) -> str:
    return flow_id_to_json(flow)


def _solution_value(term: GurobiAffineTerm) -> float:
    return float(term.X if isinstance(term, gp.Var) else term.getValue())


def _validate_flow_placements(
    data: CompactFormulationData,
    flow: FlowId,
    components: list[ComponentSolution],
) -> None:
    if not components:
        raise GurobiSolutionError(
            f"Expected at least one component for flow {_flow_key(flow)}"
        )

    placement = data.endpoint_placement_by_flow[flow]
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

    local_worker = data.local_worker_config_by_flow[flow]
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


class GurobiCompactSolvedFormulationModelTranslator(
    SolvedFormulationModelTranslator[
        GurobiFormulationModel[CompactFormulationData, CompactVariables]
    ]
):
    @override
    async def translate_solved_formulation_model(
        self,
        solved_formulation_model: GurobiFormulationModel[
            CompactFormulationData, CompactVariables
        ],
    ) -> OptimizationSolution:
        model = solved_formulation_model.native_model
        data = solved_formulation_model.data
        variables = solved_formulation_model.variables
        return OptimizationSolution(
            flow_solutions=tuple(
                self._translate_flow(data, variables, flow) for flow in data.flows
            ),
            objective_value=model.ObjVal,
            is_optimal=model.Status == GRB.OPTIMAL,
        )

    def _translate_flow(
        self,
        data: CompactFormulationData,
        variables: CompactVariables,
        flow: FlowId,
    ) -> FlowOptimizationSolution:
        assignment = variables.assignment
        network = variables.network
        qos = variables.qos
        selected_models = tuple(
            model_variant
            for model_variant in data.models_by_flow[flow]
            if assignment.model_selected[flow, model_variant].X > 0.5
        )
        if len(selected_models) != 1:
            raise GurobiSolutionError(
                f"Expected one selected model for flow {_flow_key(flow)}"
            )
        selected_model = selected_models[0]
        topology = data.static_profiles_by_variant[selected_model].topology
        tensor_routes = topology.tensor_routes()
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
            model_variant_id=selected_model,
            components=tuple(components),
            cuts=cuts,
            mean_batch_fill_time_s=data.batch_fill_time_s[flow, selected_model],
            mean_response_time_s=_nonnegative(qos.response_time_s[flow].X),
        )

    @staticmethod
    def _translate_components(
        data: CompactFormulationData,
        variables: CompactVariables,
        flow: FlowId,
        selected_model: ModelVariantId,
    ) -> list[ComponentSolution]:
        assignment = variables.assignment
        execution = variables.execution
        compute_queue = variables.compute_queue
        network = variables.network
        topology = data.static_profiles_by_variant[selected_model].topology
        layer_descriptions = topology.layers()
        connections = data.network_topology.connections()
        workers = data.workers_by_flow[flow]
        successors_by_worker = {
            source: data.network_topology.successors(source) for source in workers
        }
        components: list[ComponentSolution] = []

        for component in range(1, data.max_components + 1):
            if assignment.component_active[flow, component].X <= 0.5:
                continue
            selected_workers = tuple(
                worker
                for worker in data.workers_by_flow[flow]
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
                for priority in range(1, data.priority_count + 1)
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
                for layer in topology.layers_in_topological_order()
                if assignment.layer_component[flow, selected_model, layer, component].X
                > 0.5
            )
            transmission_response = (
                network.component_transmission_response_s[flow, component].X
                if component < data.max_components
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
                if component < data.max_components
                else 0.0
            )
            components.append(
                ComponentSolution(
                    component_index=component,
                    worker_id=worker,
                    backend=data.resource_profiles_by_worker[worker].execution_backend,
                    layer_keys=layers,
                    input_layer_keys=tuple(
                        layer for layer in layers if layer_descriptions[layer].is_input
                    ),
                    output_layer_keys=tuple(
                        layer for layer in layers if layer_descriptions[layer].is_output
                    ),
                    tensor_contract=topology.component_tensor_contract(layers),
                    priority=selected_priorities[0],
                    mean_execution_time_s=_nonnegative(
                        execution.execution_time_s[
                            flow, selected_model, component, worker
                        ].X
                    ),
                    mean_execution_queue_time_s=_nonnegative(
                        compute_queue.component_queue_wait_s[flow, component].X
                    ),
                    mean_transmission_time_s=_nonnegative(transmission_response),
                    transmission_latency_s=_nonnegative(transmission_latency),
                )
            )
        return components
