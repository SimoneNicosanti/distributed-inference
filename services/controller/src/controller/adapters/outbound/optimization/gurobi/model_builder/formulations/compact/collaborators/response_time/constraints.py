import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.compute_queue.compute_queue_variables import (
    ComputeQueueVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.response_time.response_time_variables import (
    QoSVariables,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)


def add_response_time_constraints(
    model: gp.Model,
    data: CompactFormulationData,
    *,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    compute_queue: ComputeQueueVariables,
    network: NetworkVariables,
    qos: QoSVariables,
) -> None:
    problem = data
    max_components = data.max_components
    objective_kind = data.objective_kind
    p = problem
    connections = problem.network_topology.connections()
    for flow in p.flows:
        flow_index = p.flow_index[flow]
        workers = p.workers_by_flow[flow]
        successors_by_worker = {
            source: problem.network_topology.successors(source) for source in workers
        }
        batch_fill = gp.quicksum(
            p.batch_fill_time_s[flow, model_variant]
            * assignment.model_selected[flow, model_variant]
            for model_variant in p.models_by_flow[flow]
        )
        execution_response = gp.quicksum(
            execution.execution_time_s[flow, model_variant, component, worker]
            for model_variant in p.models_by_flow[flow]
            for component in range(1, max_components + 1)
            for worker in p.workers_by_flow[flow]
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
            CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
        }:
            model.addConstr(
                qos.response_time_s[flow]
                <= float(problem.max_response_time_s_by_flow[flow]),
                name=f"response_requirement[{flow_index}]",
            )
