"""Run the production full strategy and retain its standardized solve output."""

from dataclasses import asdict
from time import perf_counter
from typing import override

from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import (
    GurobiSolver,
)
from optimizer.adapters.outbound.gurobi.strategy_factory import (
    build_compact_gurobi_solver_strategy,
)
from optimizer.application.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
)
from optimizer.application.strategies.solver_strategy_kind import SolverStrategyKind
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ObjectiveKind,
)
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.instances import Scenario, make_instance

type CompactGurobiModel = GurobiCompiledFormulation[
    MisocpCompactFormulation, CompactVariables
]
type CompactSolverOutput = SolverOutput[CompactGurobiModel, GurobiRunStatistics]


class _RecordingSolver(
    Solver[
        CompactGurobiModel,
        GurobiSolverConfig,
        CompactGurobiModel,
        GurobiRunStatistics,
    ]
):
    """Retain evidence even when the strategy cannot extract a solution."""

    def __init__(
        self, solver: GurobiSolver[MisocpCompactFormulation, CompactVariables]
    ) -> None:
        self._solver = solver
        self.output: CompactSolverOutput | None = None

    @override
    async def solve_compiled_formulation(
        self,
        compiled_formulation: CompactGurobiModel,
        solver_config: GurobiSolverConfig,
    ) -> CompactSolverOutput:
        self.output = None
        self.output = await self._solver.solve_compiled_formulation(
            compiled_formulation, solver_config
        )
        return self.output


def build_solver_strategy(
    formulation_config: MisocpCompactFormulationConfig,
    solver_config: GurobiSolverConfig,
    *,
    strategy_config: GurobiStrategyConfig | None = None,
) -> tuple[SolverStrategy[CompactOptimizationProblem], _RecordingSolver]:
    recorder = _RecordingSolver(
        GurobiSolver[MisocpCompactFormulation, CompactVariables]()
    )
    strategy = build_compact_gurobi_solver_strategy(
        formulation_config,
        solver_config,
        strategy_config=strategy_config,
        solver=recorder,
    )
    return strategy, recorder


def classify(
    output: CompactSolverOutput | None,
    error: Exception | None,
    *,
    objective: ObjectiveKind,
    formulation: CompactFormulationKind = CompactFormulationKind.ASSIGNMENT,
) -> dict[str, object]:
    statistics = output.statistics if output is not None else None
    status = statistics.native_status if statistics is not None else None
    kind = output.termination if output is not None else None
    incumbent = output is not None and output.solved_model is not None
    no_incumbent = (
        isinstance(error, SolveError) and output is not None and not incumbent
    )
    if error is not None and not no_incumbent:
        termination = "solver_error"
    elif kind == SolverTerminationKind.INFEASIBLE:
        termination = "proven_infeasible"
    elif kind == SolverTerminationKind.TIME_LIMIT:
        termination = (
            "timeout_with_incumbent" if incumbent else "timeout_without_incumbent"
        )
    elif kind == SolverTerminationKind.OPTIMAL:
        gap = statistics.relative_gap if statistics is not None else None
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
    elif kind == SolverTerminationKind.INTERRUPTED:
        termination = "interrupted"
    else:
        termination = "other_termination" if output is not None else "solver_error"
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
        "LOCALLY_OPTIMAL",
        "LOCALLY_INFEASIBLE",
    )
    native_name = next(
        (
            name
            for name in names
            if status is not None and getattr(GRB, name, None) == status
        ),
        None,
    )
    return {
        "termination": termination,
        "solver_status": status,
        "solver_status_name": native_name,
        "feasible_incumbent": incumbent,
        "timeout_reached": kind == SolverTerminationKind.TIME_LIMIT,
        "infeasibility_proven": kind == SolverTerminationKind.INFEASIBLE,
        "objective_value": output.objective_value if output and incumbent else None,
        "final_gap": statistics.relative_gap if statistics and incumbent else None,
        "objective_bound": output.objective_bound if output is not None else None,
        "solver_runtime_s": output.runtime_s if output is not None else None,
        "construction_time_s": statistics.construction_time_s if statistics else None,
        "time_to_first_incumbent_s": (
            statistics.time_to_first_incumbent_s if statistics else None
        ),
        "time_to_zero_tardiness_s": statistics.time_to_zero_objective_s
        if statistics
        and objective
        in {
            ObjectiveKind.DEADLINE_VIOLATIONS,
            ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
        }
        else None,
        "original_variables": statistics.original_variables if statistics else None,
        "original_binary_variables": statistics.binary_variables
        if statistics
        else None,
        "original_constraints": statistics.linear_constraints
        + statistics.quadratic_constraints
        + statistics.general_constraints
        if statistics
        else None,
        "original_linear_nonzeros": statistics.linear_nonzeros if statistics else None,
        "original_quadratic_constraint_nonzeros": statistics.quadratic_constraint_nonzeros
        if statistics
        else None,
        "presolve_columns_removed": statistics.presolve_columns_removed
        if statistics
        else None,
        "presolve_rows_removed": statistics.presolve_rows_removed
        if statistics
        else None,
        "node_count": statistics.node_count if statistics else None,
        "effective_solver_configuration": {
            "formulation": formulation.value,
            "objective": objective.value,
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
    started = perf_counter()
    failure = None
    solution = None
    recorder = None
    strategy = None
    try:
        problem = make_instance(scenario, spec, config=config)
        strategy, recorder = build_solver_strategy(
            MisocpCompactFormulationConfig(
                relative_error=config.optimization.relative_error,
                representation=spec.formulation,
            ),
            GurobiSolverConfig(
                output_flag=config.solver.output_flag,
                time_limit_s=spec.timeout_s,
                mip_gap=config.solver.relative_gap,
                threads=config.solver.threads,
                seed=config.solver.seed,
                record_incumbent_history=True,
            ),
            strategy_config=GurobiStrategyConfig(
                strategy=SolverStrategyKind(spec.strategy),
                insertion_time_limit_s=spec.insertion_time_limit_s,
                repair_time_limit_s=spec.repair_time_limit_s,
                final_time_limit_s=spec.final_time_limit_s,
            ),
        )
        solution = await strategy.solve(problem)
    except Exception as error:
        failure = error
    output = recorder.output if recorder is not None else None
    diagnostics = (
        strategy.diagnostics
        if isinstance(strategy, IncrementalFlowSolverStrategy)
        else None
    )
    if (
        diagnostics
        and diagnostics.attempts
        and diagnostics.attempts[-1].termination is None
    ):
        output = None
    result = classify(
        output, failure, objective=spec.objective, formulation=spec.formulation
    )
    extra: dict[str, object] = {}
    if diagnostics is not None:
        record = diagnostics.as_record()
        extra = {
            "strategy_diagnostics": record,
            **{
                key: value
                for key, value in record.items()
                if key not in {"attempts", "wall_time_s"}
            },
        }
        result.update(
            {
                "feasible_incumbent": solution is not None,
                "objective_value": solution.objective_value
                if solution is not None
                else None,
                "solver_runtime_s": sum(
                    attempt.solver_runtime_s or 0 for attempt in diagnostics.attempts
                ),
                "construction_time_s": sum(
                    attempt.construction_time_s for attempt in diagnostics.attempts
                ),
                # Native history/bounds belong to the last attempt, not the strategy.
                "time_to_first_incumbent_s": None,
                "time_to_zero_tardiness_s": None,
                "final_gap": None,
                "objective_bound": None,
                "infeasibility_proven": False,
            }
        )
        last = diagnostics.attempts[-1] if diagnostics.attempts else None
        full_infeasible = (
            failure is not None
            and last is not None
            and last.stage != "insertion"
            and last.flow_count == spec.flow_count
            and last.termination == SolverTerminationKind.INFEASIBLE
        )
        result["infeasibility_proven"] = full_infeasible
        if solution is not None:
            result["termination"] = (
                "feasible_fallback"
                if diagnostics.returned_source == "incremental"
                else result["termination"]
            )
        elif not full_infeasible and result["termination"] == "proven_infeasible":
            result["termination"] = "incremental_failure"
    return {
        **result,
        **extra,
        "strategy": spec.strategy,
        "model_selection": spec.model_selection,
        "formulation": spec.formulation.value,
        "wall_time_s": perf_counter() - started,
        "raw_solver_statistics": asdict(output.statistics)
        if output is not None and diagnostics is None
        else None,
        "last_attempt_solver_statistics": asdict(output.statistics)
        if output is not None and diagnostics is not None
        else None,
        "solution_extracted": solution is not None,
        "requested_solver_configuration": {
            "strategy": spec.strategy,
            "insertion_time_limit_s": spec.insertion_time_limit_s,
            "repair_time_limit_s": spec.repair_time_limit_s,
            "final_time_limit_s": spec.final_time_limit_s,
            "formulation": spec.formulation.value,
            "time_limit_s": spec.timeout_s,
            "mip_gap": config.solver.relative_gap,
            "threads": config.solver.threads,
            "seed": config.solver.seed,
            "objective": spec.objective.value,
        },
    }
