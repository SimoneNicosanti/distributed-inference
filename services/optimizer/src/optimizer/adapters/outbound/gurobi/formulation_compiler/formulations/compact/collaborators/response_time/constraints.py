import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.compute_queue.compute_queue_variables import (
    ComputeQueueVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.response_time.response_time_variables import (
    QoSVariables,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)


def add_response_time_constraints(
    model: gp.Model,
    data: MisocpCompactFormulation,
    *,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    compute_queue: ComputeQueueVariables,
    network: NetworkVariables,
    qos: QoSVariables,
) -> None:
    problem = data
    max_components = data.problem.policy.max_components
    objective_kind = data.problem.policy.objective
    p = problem
    connections = problem.connections_by_pair
    for flow in p.flows:
        flow_index = p.flow_index[flow]
        workers = p.problem.flows[flow].workers
        successors_by_worker = {
            source: problem.successors_by_worker[source] for source in workers
        }
        batch_fill = gp.quicksum(
            p.batch_fill_time_s[flow, model_variant]
            * assignment.configuration_selected[flow, model_variant]
            for model_variant in p.configurations_by_flow[flow]
        )
        execution_response = gp.quicksum(
            execution.execution_time_s[flow, model_variant, component, worker]
            for model_variant in p.configurations_by_flow[flow]
            for component in range(1, max_components + 1)
            for worker in p.problem.flows[flow].workers
        )
        execution_queue = gp.quicksum(
            compute_queue.component_queue_wait_s[flow, component]
            for component in range(1, max_components + 1)
        )
        transmission_response = gp.quicksum(
            network.component_transmission_response_s[flow, cut]
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
            qos.response_time_s[flow]
            == batch_fill
            + execution_response
            + execution_queue
            + transmission_response
            + latency,
            name=f"response_definition[{flow_index}]",
        )
        if objective_kind in {
            ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
        }:
            model.addConstr(
                qos.response_time_s[flow]
                <= float(problem.problem.flows[flow].max_response_time_s),
                name=f"response_requirement[{flow_index}]",
            )
