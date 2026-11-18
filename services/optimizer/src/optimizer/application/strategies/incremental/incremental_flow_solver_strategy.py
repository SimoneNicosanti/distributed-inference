import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol, override

from optimizer.application.abc.formulation_builder import FormulationBuilder
from optimizer.application.abc.solver_strategy import SolveError, SolverStrategy
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedCompiledFormulationTranslator,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.application.strategies.incremental.compact_problem_prefix import (
    compact_problem_prefix,
)
from optimizer.application.strategies.incremental.incremental_flow_diagnostics import (
    IncrementalAttemptDiagnostics,
    IncrementalFlowDiagnostics,
    IncrementalStage,
)
from optimizer.application.strategies.incremental.incremental_model_adapter import (
    IncrementalModelAdapter,
)
from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.formulation.formulation_config import FormulationConfig
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ObjectiveKind,
)
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class IncrementalSolveStatistics(Protocol):
    @property
    def time_to_first_incumbent_s(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class IncrementalFlowStrategyConfig[SolverConfigT]:
    insertion_solver_config: SolverConfigT
    repair_solver_config: SolverConfigT
    final_solver_config: SolverConfigT


class IncrementalFlowSolveError(SolveError):
    def __init__(self, attempt: IncrementalAttemptDiagnostics) -> None:
        self.attempt = attempt
        self.termination = attempt.termination
        super().__init__(
            f"{attempt.stage}: no valid incumbent for {attempt.flow_count} flows "
            f"({attempt.termination or attempt.error_type})"
        )


class IncrementalFlowSolverStrategy[
    FormulationT: Formulation,
    FormulationConfigT: FormulationConfig,
    CompiledFormulationT,
    SolverConfigT,
    SolvedCompiledFormulationT,
    SolverStatisticsT: IncrementalSolveStatistics,
](SolverStrategy[CompactOptimizationProblem]):
    """Fixed-decision insertion, conditional repair, then one unrestricted solve."""

    def __init__(
        self,
        *,
        formulation_builder: FormulationBuilder[
            CompactOptimizationProblem, FormulationT, FormulationConfigT
        ],
        formulation_config: FormulationConfigT,
        formulation_compiler: FormulationCompiler[FormulationT, CompiledFormulationT],
        config: IncrementalFlowStrategyConfig[SolverConfigT],
        solver: Solver[
            CompiledFormulationT,
            SolverConfigT,
            SolvedCompiledFormulationT,
            SolverStatisticsT,
        ],
        solved_model_translator: SolvedCompiledFormulationTranslator[
            SolvedCompiledFormulationT
        ],
        model_adapter: IncrementalModelAdapter[
            CompiledFormulationT, SolvedCompiledFormulationT
        ],
        observer: Callable[[IncrementalFlowDiagnostics], None] | None = None,
    ) -> None:
        self._formulation_builder = formulation_builder
        self._formulation_config = formulation_config
        self._formulation_compiler = formulation_compiler
        self._config = config
        self._solver = solver
        self._translator = solved_model_translator
        self._model_adapter = model_adapter
        self._observer = observer
        self.diagnostics: IncrementalFlowDiagnostics | None = None

    @override
    async def solve(self, problem: CompactOptimizationProblem) -> OptimizationSolution:
        self.diagnostics = None
        if not problem.flows:
            raise ValueError("Incremental optimization requires at least one flow")
        started = perf_counter()
        attempts: list[IncrementalAttemptDiagnostics] = []
        incumbent = None
        returned = None
        source = None
        completed = 0
        before_final = None
        after_final = None
        try:
            for count in range(1, len(problem.flows) + 1):
                prefix = compact_problem_prefix(problem, count)
                candidate = await self._attempt(
                    prefix,
                    IncrementalStage.FIRST_FLOW
                    if count == 1
                    else IncrementalStage.INSERTION,
                    self._config.insertion_solver_config,
                    incumbent,
                    fix_decisions=incumbent is not None,
                    reason=None,
                    attempts=attempts,
                )
                if candidate is None and incumbent is not None:
                    termination = attempts[-1].termination
                    if termination in {
                        SolverTerminationKind.INFEASIBLE,
                        SolverTerminationKind.TIME_LIMIT,
                    }:
                        candidate = await self._attempt(
                            prefix,
                            IncrementalStage.REPAIR,
                            self._config.repair_solver_config,
                            incumbent,
                            fix_decisions=False,
                            reason="restricted_infeasible"
                            if termination == SolverTerminationKind.INFEASIBLE
                            else "restricted_timeout_without_incumbent",
                            attempts=attempts,
                        )
                if candidate is None:
                    raise IncrementalFlowSolveError(attempts[-1])
                incumbent = candidate
                completed = count

            assert incumbent is not None
            before_final = incumbent.objective_value
            final = await self._attempt(
                problem,
                IncrementalStage.FINAL,
                self._config.final_solver_config,
                incumbent,
                fix_decisions=False,
                reason=None,
                attempts=attempts,
            )
            after_final = final.objective_value if final is not None else None
            maximize = (
                problem.policy.objective == ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY
            )
            final_is_better = final is not None and (
                final.objective_value >= incumbent.objective_value
                if maximize
                else final.objective_value <= incumbent.objective_value
            )
            if final_is_better:
                returned, source = final, "final"
            else:
                returned, source = replace(incumbent, is_optimal=False), "incremental"
            assert returned is not None
            return returned
        finally:
            self.diagnostics = IncrementalFlowDiagnostics(
                attempts=tuple(attempts),
                completed_flow_count=completed,
                objective_before_final=before_final,
                objective_after_final=after_final,
                returned_objective_value=returned.objective_value if returned else None,
                returned_source=source,
                wall_time_s=perf_counter() - started,
            )
            if self._observer is not None:
                self._observer(self.diagnostics)

    async def _attempt(
        self,
        problem: CompactOptimizationProblem,
        stage: IncrementalStage,
        solver_config: SolverConfigT,
        seed: OptimizationSolution | None,
        *,
        fix_decisions: bool,
        reason: str | None,
        attempts: list[IncrementalAttemptDiagnostics],
    ) -> OptimizationSolution | None:
        started = perf_counter()
        construction_time = 0.0
        output = None
        solution = None
        error_type = None
        error_details = None
        solver_called = False
        try:
            formulation = self._formulation_builder.build_formulation(
                problem, self._formulation_config
            )
            async with self._formulation_compiler.compile_formulation(
                formulation
            ) as compiled:
                construction_time = perf_counter() - started
                if seed is not None:
                    await self._model_adapter.initialize_model(
                        compiled, seed, fix_decisions=fix_decisions
                    )
                solver_called = True
                output = await self._solver.solve_compiled_formulation(
                    compiled, solver_config
                )
                if output.solved_model is not None:
                    if output.termination in {
                        SolverTerminationKind.NUMERICAL_FAILURE,
                        SolverTerminationKind.UNKNOWN,
                        SolverTerminationKind.INFEASIBLE,
                        SolverTerminationKind.UNBOUNDED,
                        SolverTerminationKind.INFEASIBLE_OR_UNBOUNDED,
                    } or not self._model_adapter.is_valid_solution(output.solved_model):
                        raise ValueError(
                            "Solver incumbent failed feasibility validation"
                        )
                    solution = (
                        await self._translator.translate_solved_formulation_model(
                            output.solved_model
                        )
                    )
                    if {flow.flow_id for flow in solution.flow_solutions} != set(
                        problem.flows
                    ) or not math.isfinite(solution.objective_value):
                        raise ValueError(
                            "Solver solution does not cover the complete prefix"
                        )
        except asyncio.CancelledError:
            error_type = "CancelledError"
            raise
        except Exception as error:
            solution = None
            error_type, error_details = type(error).__name__, str(error)
        finally:
            attempts.append(
                IncrementalAttemptDiagnostics(
                    stage=stage,
                    flow_count=len(problem.flows),
                    fixed_flow_count=len(seed.flow_solutions)
                    if fix_decisions and seed
                    else 0,
                    reason=reason,
                    termination=output.termination if output else None,
                    feasible_incumbent=solution is not None,
                    objective_value=output.objective_value if output else None,
                    construction_time_s=construction_time or perf_counter() - started,
                    solver_runtime_s=output.runtime_s if output else None,
                    time_to_first_incumbent_s=output.statistics.time_to_first_incumbent_s
                    if output
                    else None,
                    wall_time_s=perf_counter() - started,
                    error_type=error_type,
                    error_details=error_details,
                    solver_called=solver_called,
                )
            )
        return solution
