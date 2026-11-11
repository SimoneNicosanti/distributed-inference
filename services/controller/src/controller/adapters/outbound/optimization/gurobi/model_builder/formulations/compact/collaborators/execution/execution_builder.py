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
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId

from .constraints import add_execution_constraints


class CompactExecutionBuilder:
    def create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> ExecutionVariables:
        variables = _add_execution_variables(
            model, data, max_components=data.max_components
        )
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: CompactFormulationData,
        *,
        assignment: AssignmentVariables,
        execution: ExecutionVariables,
    ) -> None:
        add_execution_constraints(
            model, data, assignment=assignment, execution=execution
        )


def _add_execution_variables(
    model: gp.Model,
    problem: CompactFormulationData,
    *,
    max_components: int,
) -> ExecutionVariables:
    execution_time_s: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var] = {}
    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.models_by_flow[flow]:
            model_index = problem.model_index[flow, model_variant]
            for component in range(1, max_components + 1):
                for worker in problem.workers_by_flow[flow]:
                    worker_index = problem.worker_index[worker]
                    execution_time_s[flow, model_variant, component, worker] = (
                        model.addVar(
                            lb=0.0,
                            name=(
                                f"tex[{flow_index},{model_index},{component},"
                                f"{worker_index}]"
                            ),
                        )
                    )
    return ExecutionVariables(execution_time_s=execution_time_s)
