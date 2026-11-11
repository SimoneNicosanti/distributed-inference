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

from .constraints import add_response_time_constraints


class CompactResponseTimeBuilder:
    def create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> QoSVariables:
        variables = _add_qos_variables(model, data)
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: CompactFormulationData,
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
    problem: CompactFormulationData,
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
