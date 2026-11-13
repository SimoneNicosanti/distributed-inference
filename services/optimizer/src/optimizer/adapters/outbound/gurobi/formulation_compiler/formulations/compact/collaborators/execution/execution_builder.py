import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId

from .constraints import add_execution_constraints


class CompactExecutionBuilder:
    def create_variables(
        self, model: gp.Model, data: MisocpCompactFormulation
    ) -> ExecutionVariables:
        variables = _add_execution_variables(
            model, data, max_components=data.problem.policy.max_components
        )
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: MisocpCompactFormulation,
        *,
        assignment: AssignmentVariables,
        execution: ExecutionVariables,
    ) -> None:
        add_execution_constraints(
            model, data, assignment=assignment, execution=execution
        )


def _add_execution_variables(
    model: gp.Model,
    problem: MisocpCompactFormulation,
    *,
    max_components: int,
) -> ExecutionVariables:
    execution_time_s: dict[
        tuple[FlowId, ModelConfigurationKey, int, WorkerId], gp.Var
    ] = {}
    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.configurations_by_flow[flow]:
            model_index = problem.configuration_index[flow, model_variant]
            for component in range(1, max_components + 1):
                for worker in problem.problem.flows[flow].workers:
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
