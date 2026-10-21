from itertools import pairwise

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_expressions import (
    GurobiAffineTerm,
    tensor_crossing_term,
)
from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    ComputeQueueVariables,
    ExecutionVariables,
    NetworkVariables,
    QoSVariables,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.optimization_solution import (
    ComponentSolution,
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
    return flow.model_dump_json()


def _solution_value(term: GurobiAffineTerm) -> float:
    return float(term.X if isinstance(term, gp.Var) else term.getValue())


def extract_solution(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    compute_queue: ComputeQueueVariables,
    network: NetworkVariables,
    qos: QoSVariables,
    *,
    requests_by_flow: OptimizationRequestsByFlow,
    context: OptimizationContext,
    max_components: int,
    priority_count: int,
) -> OptimizationSolution:
    p = problem
    flow_solutions: list[FlowOptimizationSolution] = []
    connections = context.network_topology.connections()

    for flow in p.flows:
        workers = p.workers_by_flow[flow]
        successors_by_worker = {
            source: context.network_topology.successors(source) for source in workers
        }
        selected_models = tuple(
            model_variant
            for model_variant in p.models_by_flow[flow]
            if assignment.model_selected[flow, model_variant].X > 0.5
        )
        if len(selected_models) != 1:
            raise GurobiSolutionError(
                f"Expected one selected model for flow {_flow_key(flow)}"
            )
        selected_model = selected_models[0]
        topology = context.static_profiles_by_variant[selected_model].topology
        layer_descriptions = topology.layers()
        tensor_routes = topology.tensor_routes()
        components: list[ComponentSolution] = []

        for component in range(1, max_components + 1):
            if assignment.component_active[flow, component].X <= 0.5:
                continue
            selected_workers = tuple(
                worker
                for worker in p.workers_by_flow[flow]
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
                for priority in range(1, priority_count + 1)
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
                network.component_transmission_response[flow, component].X
                if component < max_components
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
                if component < max_components
                else 0.0
            )
            components.append(
                ComponentSolution(
                    component_index=component,
                    worker_id=worker,
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
                        execution.execution_time[
                            flow, selected_model, component, worker
                        ].X
                    ),
                    mean_execution_queue_time_s=_nonnegative(
                        compute_queue.component_queue_wait[flow, component].X
                    ),
                    mean_transmission_time_s=_nonnegative(transmission_response),
                    transmission_latency_s=_nonnegative(transmission_latency),
                )
            )

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

        flow_solutions.append(
            FlowOptimizationSolution(
                flow_id=flow,
                ingress_id=requests_by_flow[flow].ingress_id,
                model_variant_id=selected_model,
                components=tuple(components),
                cuts=cuts,
                mean_batch_fill_time_s=p.batch_fill_time[flow, selected_model],
                mean_response_time_s=_nonnegative(qos.response_time[flow].X),
            )
        )

    return OptimizationSolution(
        flow_solutions=tuple(flow_solutions),
        objective_value=model.ObjVal,
        is_optimal=model.Status == GRB.OPTIMAL,
    )
