import asyncio
import math
from time import perf_counter
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
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.optimization.solving.solver_config import (
    ProblemConfig,
    SolverConfig,
    SolverRunStatistics,
)


class OptimizationInfeasibleError(RuntimeError):
    pass


class OptimizationNotSolvedError(RuntimeError):
    pass


def _finite_attribute(model: gp.Model, name: str) -> float | None:
    try:
        value = float(model.getAttr(name))
    except AttributeError, gp.GurobiError:
        return None
    return value if math.isfinite(value) and abs(value) < GRB.INFINITY else None


def _observe_model(
    model: gp.Model, config: SolverConfig, construction_time_s: float
) -> None:
    if config.observer is None:
        return
    status = _finite_attribute(model, "Status")
    solution_count = int(_finite_attribute(model, "SolCount") or 0)
    config.observer(
        SolverRunStatistics(
            status=int(status) if status is not None else None,
            runtime_s=_finite_attribute(model, "Runtime"),
            solution_count=solution_count,
            objective_value=(
                _finite_attribute(model, "ObjVal") if solution_count else None
            ),
            objective_bound=_finite_attribute(model, "ObjBound"),
            relative_gap=(
                _finite_attribute(model, "MIPGap") if solution_count else None
            ),
            construction_time_s=construction_time_s,
            time_limit_s=float(model.Params.TimeLimit),
            mip_gap_target=float(model.Params.MIPGap),
            threads=int(model.Params.Threads),
            seed=int(model.Params.Seed),
        )
    )


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
        construction_started = perf_counter()
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
                if self._config.threads is not None:
                    model.Params.Threads = self._config.threads
                if self._config.seed is not None:
                    model.Params.Seed = self._config.seed

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
                    requests_by_flow=requests_by_flow,
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
                construction_time_s = perf_counter() - construction_started
                try:
                    model.optimize()
                finally:
                    _observe_model(model, self._config, construction_time_s)

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
