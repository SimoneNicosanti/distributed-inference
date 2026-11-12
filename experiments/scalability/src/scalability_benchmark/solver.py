"""Run the production full strategy and retain its standardized solve output."""

from dataclasses import asdict
from time import perf_counter
from typing import override

from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_formulation_model import (
    GurobiFormulationModel,
)
from controller.adapters.outbound.optimization.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from controller.adapters.outbound.optimization.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.compact_variables import (
    CompactVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.gurobi_compact_formulation_compiler import (
    GurobiCompactFormulationCompiler,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.gurobi_formulation_model_builder import (
    GurobiFormulationModelBuilder,
)
from controller.adapters.outbound.optimization.gurobi.result_translation.formulations.compact.gurobi_compact_solved_formulation_model_translator import (
    GurobiCompactSolvedFormulationModelTranslator,
)
from controller.adapters.outbound.optimization.gurobi.solver.gurobi_solver import (
    GurobiSolver,
)
from controller.adapters.outbound.optimization.gurobi.solver_config_builder.gurobi_solver_config_builder import (
    GurobiSolverConfigBuilder,
)
from controller.application.optimization.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from controller.application.optimization.formulation.compact.compact_formulation_data_builder import (
    CompactFormulationDataBuilder,
)
from controller.application.optimization.strategies.full_solver_strategy import (
    FullSolverStrategy,
)
from controller.application.ports.outbound.optimization.solver import Solver
from controller.domain.optimization.solving.formulation.compact.compact_formulation_config import (
    CompactFormulationConfig,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.optimization.solving.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.instances import Scenario, make_instance

type CompactGurobiModel = GurobiFormulationModel[
    CompactFormulationData, CompactVariables
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
        self, solver: GurobiSolver[CompactFormulationData, CompactVariables]
    ) -> None:
        self._solver = solver
        self.output: CompactSolverOutput | None = None

    @override
    async def solve_formulation_model(
        self, model: CompactGurobiModel, config: GurobiSolverConfig
    ) -> CompactSolverOutput:
        self.output = None
        self.output = await self._solver.solve_formulation_model(model, config)
        return self.output


def build_solver_strategy(
    formulation_config: CompactFormulationConfig,
    solver_config: GurobiSolverConfig,
) -> tuple[SolverStrategy, _RecordingSolver]:
    recorder = _RecordingSolver(
        GurobiSolver[CompactFormulationData, CompactVariables]()
    )
    strategy = FullSolverStrategy(
        formulation_data_builder=CompactFormulationDataBuilder(),
        formulation_model_builder=GurobiFormulationModelBuilder(
            GurobiCompactFormulationCompiler(),
            non_convex=0,
            output_flag=solver_config.output_flag,
        ),
        solver_config_builder=GurobiSolverConfigBuilder(
            output_flag=solver_config.output_flag,
            time_limit_s=solver_config.time_limit_s,
            mip_gap=solver_config.mip_gap,
            threads=solver_config.threads,
            seed=solver_config.seed,
            observer=solver_config.observer,
            record_incumbent_history=solver_config.record_incumbent_history,
        ),
        solver=recorder,
        solved_model_translator=GurobiCompactSolvedFormulationModelTranslator(),
        formulation_config=formulation_config,
    )
    return strategy, recorder


def classify(
    output: CompactSolverOutput | None,
    error: Exception | None,
    *,
    objective: CompactObjectiveKind,
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
        requests, context = make_instance(scenario, spec)
        strategy, recorder = build_solver_strategy(
            CompactFormulationConfig(
                max_components=config.optimization.max_components,
                priority_count=config.optimization.priority_count,
                compute_utilization_limit=config.optimization.compute_utilization_limit,
                network_utilization_limit=config.optimization.network_utilization_limit,
                relative_error=config.optimization.relative_error,
                objective=spec.objective,
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
        solution = await strategy.solve(requests, context)
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
