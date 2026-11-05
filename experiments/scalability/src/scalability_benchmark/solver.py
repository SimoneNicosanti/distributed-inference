"""Invoke the existing solver and retain native termination evidence."""

from dataclasses import asdict
from time import perf_counter

from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_solver import (
    GurobiSolver,
    OptimizationInfeasibleError,
    OptimizationNotSolvedError,
)
from controller.domain.optimization.solving.solver_config import (
    FormulationKind,
    ProblemConfig,
    SolverConfig,
    SolverRunStatistics,
)
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.instances import Scenario, make_instance


def classify(
    statistics: SolverRunStatistics | None, error: Exception | None
) -> dict[str, object]:
    status = statistics.status if statistics else None
    incumbent = statistics is not None and statistics.solution_count > 0
    if error is not None and not isinstance(
        error, (OptimizationInfeasibleError, OptimizationNotSolvedError)
    ):
        termination = "solver_error"
    elif status == GRB.INFEASIBLE:
        termination = "proven_infeasible"
    elif status == GRB.TIME_LIMIT:
        termination = (
            "timeout_with_incumbent" if incumbent else "timeout_without_incumbent"
        )
    elif status == GRB.OPTIMAL:
        # Native OPTIMAL also covers positive configured optimality tolerances.
        gap = statistics.relative_gap if statistics else None
        if gap == 0:
            termination = "optimal_closed_gap"
        elif (
            gap is not None
            and statistics is not None
            and gap <= statistics.mip_gap_target
        ):
            termination = "gap_target_reached"
        else:
            termination = "optimal_within_tolerance"
    elif status == GRB.INTERRUPTED:
        termination = "interrupted"
    else:
        termination = "other_termination" if statistics else "solver_error"
    names = (
        "LOADED",
        "OPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "CUTOFF",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "TIME_LIMIT",
        "SOLUTION_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "SUBOPTIMAL",
        "INPROGRESS",
        "USER_OBJ_LIMIT",
        "WORK_LIMIT",
        "MEM_LIMIT",
    )
    native_name = next((name for name in names if getattr(GRB, name) == status), None)
    return {
        "termination": termination,
        "solver_status": status,
        "solver_status_name": native_name,
        "feasible_incumbent": incumbent,
        "timeout_reached": status == GRB.TIME_LIMIT,
        "infeasibility_proven": status == GRB.INFEASIBLE,
        "objective_value": statistics.objective_value
        if statistics and incumbent
        else None,
        "final_gap": statistics.relative_gap if statistics and incumbent else None,
        "objective_bound": statistics.objective_bound if statistics else None,
        "solver_runtime_s": statistics.runtime_s if statistics else None,
        "construction_time_s": statistics.construction_time_s if statistics else None,
        "time_to_first_incumbent_s": (
            statistics.time_to_first_incumbent_s if statistics else None
        ),
        "effective_solver_configuration": {
            "formulation": statistics.formulation,
            "objective": statistics.objective,
            "time_limit_s": statistics.time_limit_s,
            "mip_gap": statistics.mip_gap_target,
            "threads": statistics.threads,
            "seed": statistics.seed,
        }
        if statistics
        else None,
        "error_type": type(error).__name__ if error else None,
        "error_details": str(error) if error else None,
    }


async def execute_solver(
    scenario: Scenario, spec: RunSpec, config: ExperimentConfig
) -> dict[str, object]:
    observations: list[SolverRunStatistics] = []
    solver_config = SolverConfig(
        formulation=FormulationKind(config.solver.formulation),
        objective=spec.objective,
        output_flag=True,
        time_limit_s=spec.timeout_s,
        mip_gap=config.solver.relative_gap,
        threads=config.solver.threads,
        seed=config.solver.seed,
        observer=observations.append,
        record_incumbent_history=True,
    )
    started = perf_counter()
    failure = None
    solution = None
    try:
        requests, context = make_instance(scenario, spec)
        solution = await GurobiSolver(solver_config).solve(
            requests,
            context,
            ProblemConfig(
                max_components=config.optimization.max_components,
                priority_count=config.optimization.priority_count,
                compute_utilization_limit=config.optimization.compute_utilization_limit,
                network_utilization_limit=config.optimization.network_utilization_limit,
                relative_error=config.optimization.relative_error,
            ),
        )
    except Exception as error:
        failure = error
    statistics = observations[-1] if observations else None
    return {
        **classify(statistics, failure),
        "wall_time_s": perf_counter() - started,
        "raw_solver_statistics": asdict(statistics) if statistics else None,
        "solution_extracted": solution is not None,
        "requested_solver_configuration": {
            "formulation": config.solver.formulation,
            "time_limit_s": spec.timeout_s,
            "mip_gap": config.solver.relative_gap,
            "threads": config.solver.threads,
            "seed": config.solver.seed,
            "objective": spec.objective.value,
        },
    }
