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
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import (
    GurobiSolver,
)
from optimizer.application.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)
from optimizer.application.strategies.full_solver_strategy import (
    FullSolverStrategy,
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
) -> tuple[SolverStrategy[CompactOptimizationProblem], _RecordingSolver]:
    recorder = _RecordingSolver(
        GurobiSolver[MisocpCompactFormulation, CompactVariables]()
    )
    strategy = FullSolverStrategy(
        formulation_builder=MisocpCompactFormulationBuilder(),
        formulation_compiler=GurobiFormulationCompiler(
            GurobiMisocpCompactFormulationModelBuilder(),
            non_convex=0,
            output_flag=solver_config.output_flag,
        ),
        solver_config=solver_config,
        solver=recorder,
        solved_model_translator=GurobiMisocpCompactSolvedCompiledFormulationTranslator(),
        formulation_config=formulation_config,
    )
    return strategy, recorder


def classify(
    output: CompactSolverOutput | None,
    error: Exception | None,
    *,
    objective: ObjectiveKind,
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
        "effective_solver_configuration": {
            "formulation": "compact",
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
    try:
        problem = make_instance(scenario, spec, config=config)
        strategy, recorder = build_solver_strategy(
            MisocpCompactFormulationConfig(
                relative_error=config.optimization.relative_error,
            ),
            GurobiSolverConfig(
                output_flag=config.solver.output_flag,
                time_limit_s=spec.timeout_s,
                mip_gap=config.solver.relative_gap,
                threads=config.solver.threads,
                seed=config.solver.seed,
                record_incumbent_history=True,
            ),
        )
        solution = await strategy.solve(problem)
    except Exception as error:
        failure = error
    output = recorder.output if recorder is not None else None
    return {
        **classify(output, failure, objective=spec.objective),
        "wall_time_s": perf_counter() - started,
        "raw_solver_statistics": asdict(output.statistics)
        if output is not None
        else None,
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
