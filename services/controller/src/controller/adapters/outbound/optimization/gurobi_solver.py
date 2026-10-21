import asyncio
from typing import override

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_execution_queue_constraints import (
    add_execution_queue_constraints,
)
from controller.adapters.outbound.optimization.gurobi_memory_constraints import (
    add_memory_constraints,
)
from controller.adapters.outbound.optimization.gurobi_network_queue_constraints import (
    add_network_queue_constraints,
)
from controller.adapters.outbound.optimization.gurobi_objective import (
    set_objective,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_problem_builder import (
    DefaultGurobiProblemBuilder,
    GurobiProblemBuilder,
)
from controller.adapters.outbound.optimization.gurobi_qos_constraints import (
    add_response_time_constraints,
)
from controller.adapters.outbound.optimization.gurobi_solution import (
    GurobiSolutionError,
    extract_solution,
)
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    add_assignment_constraints,
    add_execution_constraints,
    add_transmission_constraints,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    add_gurobi_variables,
)
from controller.application.optimization.abc.solver import Solver
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.optimization.solver_config import ProblemConfig, SolverConfig


class OptimizationInfeasibleError(RuntimeError):
    pass


class OptimizationNotSolvedError(RuntimeError):
    pass


class GurobiSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        problem_builder: GurobiProblemBuilder | None = None,
    ) -> None:
        self._config = config
        self._problem_builder = problem_builder or DefaultGurobiProblemBuilder()

    @override
    async def solve(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
        problem_config: ProblemConfig,
    ) -> OptimizationSolution:
        return await asyncio.to_thread(
            self._solve,
            requests_by_flow,
            context,
            problem_config,
        )

    def _solve(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
        problem_config: ProblemConfig,
    ) -> OptimizationSolution:
        problem: GurobiProblem = self._problem_builder.build(
            requests_by_flow,
            context,
            problem_config,
        )
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", int(self._config.output_flag))
            environment.start()
            with gp.Model("distributed_inference", env=environment) as model:
                model.Params.NonConvex = 0
                model.Params.DualReductions = 0
                if self._config.time_limit_s is not None:
                    model.Params.TimeLimit = self._config.time_limit_s
                if self._config.mip_gap is not None:
                    model.Params.MIPGap = self._config.mip_gap

                variables = add_gurobi_variables(
                    model,
                    problem,
                    context=context,
                    requests_by_flow=requests_by_flow,
                    max_components=problem_config.max_components,
                    priority_count=problem_config.priority_count,
                    compute_utilization_limit=(
                        problem_config.compute_utilization_limit
                    ),
                    network_utilization_limit=(
                        problem_config.network_utilization_limit
                    ),
                )
                add_assignment_constraints(
                    model,
                    problem,
                    variables.assignment,
                    variables.network,
                    context=context,
                    requests_by_flow=requests_by_flow,
                    max_components=problem_config.max_components,
                )
                add_execution_constraints(
                    model,
                    problem,
                    variables.assignment,
                    variables.execution,
                    context=context,
                    max_components=problem_config.max_components,
                )
                add_transmission_constraints(
                    model,
                    problem,
                    variables.assignment,
                    variables.network,
                    context=context,
                    requests_by_flow=requests_by_flow,
                    max_components=problem_config.max_components,
                )
                add_execution_queue_constraints(
                    model,
                    problem,
                    variables.assignment,
                    variables.execution,
                    variables.compute_queue,
                    max_components=problem_config.max_components,
                    priority_count=problem_config.priority_count,
                    compute_utilization_limit=(
                        problem_config.compute_utilization_limit
                    ),
                )
                add_network_queue_constraints(
                    model,
                    problem,
                    variables.network,
                    max_components=problem_config.max_components,
                    network_utilization_limit=(
                        problem_config.network_utilization_limit
                    ),
                )
                add_memory_constraints(
                    model,
                    problem,
                    variables.assignment,
                    variables.memory,
                    context=context,
                    max_components=problem_config.max_components,
                )
                add_response_time_constraints(
                    model,
                    problem,
                    variables.assignment,
                    variables.execution,
                    variables.compute_queue,
                    variables.network,
                    variables.qos,
                    context=context,
                    requests_by_flow=requests_by_flow,
                    max_components=problem_config.max_components,
                )
                set_objective(
                    model,
                    problem,
                    variables.qos,
                    self._config.objective,
                    requests_by_flow=requests_by_flow,
                )
                model.optimize()

                if model.Status == GRB.INFEASIBLE:
                    raise OptimizationInfeasibleError(
                        "The optimization problem is infeasible"
                    )
                if model.SolCount == 0:
                    raise OptimizationNotSolvedError(
                        f"Gurobi stopped with status {model.Status} without an incumbent"
                    )
                try:
                    return extract_solution(
                        model,
                        problem,
                        variables.assignment,
                        variables.execution,
                        variables.compute_queue,
                        variables.network,
                        variables.qos,
                        requests_by_flow=requests_by_flow,
                        context=context,
                        max_components=problem_config.max_components,
                        priority_count=problem_config.priority_count,
                    )
                except GurobiSolutionError as error:
                    raise OptimizationNotSolvedError(str(error)) from error
