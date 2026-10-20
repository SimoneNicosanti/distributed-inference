import gurobipy as gp

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


def add_response_time_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    compute_queue: ComputeQueueVariables,
    network: NetworkVariables,
    qos: QoSVariables,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    max_components: int,
) -> None:
    p = problem
    connections = context.network_topology.connections()
    for flow in p.flows:
        flow_index = p.flow_index[flow]
        workers = p.workers_by_flow[flow]
        successors_by_worker = {
            source: context.network_topology.successors(source) for source in workers
        }
        batch_fill = gp.quicksum(
            p.batch_fill_time[flow, model_variant]
            * assignment.model_selected[flow, model_variant]
            for model_variant in p.models_by_flow[flow]
        )
        execution_response = gp.quicksum(
            execution.execution_time[flow, model_variant, component, worker]
            for model_variant in p.models_by_flow[flow]
            for component in range(1, max_components + 1)
            for worker in p.workers_by_flow[flow]
        )
        execution_queue = gp.quicksum(
            compute_queue.component_queue_wait[flow, component]
            for component in range(1, max_components + 1)
        )
        transmission_response = gp.quicksum(
            network.component_transmission_response[flow, cut]
            for cut in range(1, max_components)
        )
        latency = gp.quicksum(
            (
                0.0
                if source == target
                else connections[source, target].round_trip_time_s / 2
            )
            * network.adjacent_component_workers[flow, cut, source, target]
            for cut in range(1, max_components)
            for source in workers
            for target in workers
            if source == target or target in successors_by_worker[source]
        )
        model.addConstr(
            qos.response_time[flow]
            == batch_fill
            + execution_response
            + execution_queue
            + transmission_response
            + latency,
            name=f"response_definition[{flow_index}]",
        )
        model.addConstr(
            qos.response_time[flow]
            <= float(requests_by_flow[flow].qos_requirements.max_mean_response_time),
            name=f"response_requirement[{flow_index}]",
        )
