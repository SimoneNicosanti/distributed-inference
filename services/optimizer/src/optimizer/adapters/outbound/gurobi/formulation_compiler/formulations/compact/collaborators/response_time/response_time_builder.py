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

from .constraints import add_response_time_constraints


class CompactResponseTimeBuilder:
    def create_variables(
        self, model: gp.Model, data: MisocpCompactFormulation
    ) -> QoSVariables:
        variables = _add_qos_variables(model, data)
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: MisocpCompactFormulation,
        *,
        assignment: AssignmentVariables,
        execution: ExecutionVariables,
        compute_queue: ComputeQueueVariables,
        network: NetworkVariables,
        qos: QoSVariables,
    ) -> None:
        add_response_time_constraints(
            model,
            data,
            assignment=assignment,
            execution=execution,
            compute_queue=compute_queue,
            network=network,
            qos=qos,
        )


def _add_qos_variables(
    model: gp.Model,
    problem: MisocpCompactFormulation,
) -> QoSVariables:
    return QoSVariables(
        response_time_s={
            flow: model.addVar(
                lb=0.0,
                name=f"response[{problem.flow_index[flow]}]",
            )
            for flow in problem.flows
        }
    )
