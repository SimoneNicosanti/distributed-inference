from dataclasses import replace
from time import perf_counter
from typing import override

from optimizer.application.abc.solver_strategy import SolverStrategy
from optimizer.application.ports.outbound.persistent_formulation_compiler import (
    PersistentFormulationCompiler,
)
from optimizer.application.strategies.incremental.incremental_flow_diagnostics import (
    IncrementalFlowDiagnostics,
)
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
    IncrementalSolveStatistics,
)
from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.formulation.formulation_config import FormulationConfig
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class PersistentIncrementalFlowSolverStrategy[
    FormulationT: Formulation,
    FormulationConfigT: FormulationConfig,
    CompiledFormulationT,
    SolverConfigT,
    SolvedCompiledFormulationT,
    SolverStatisticsT: IncrementalSolveStatistics,
](SolverStrategy[CompactOptimizationProblem]):
    """Run unrestricted prefix solves within one persistent compiler session."""

    def __init__(
        self,
        strategy: IncrementalFlowSolverStrategy[
            FormulationT,
            FormulationConfigT,
            CompiledFormulationT,
            SolverConfigT,
            SolvedCompiledFormulationT,
            SolverStatisticsT,
        ],
        compiler: PersistentFormulationCompiler[FormulationT, CompiledFormulationT],
    ) -> None:
        self._strategy = strategy
        self._compiler = compiler
        self.diagnostics: IncrementalFlowDiagnostics | None = None

    @override
    async def solve(self, problem: CompactOptimizationProblem) -> OptimizationSolution:
        self.diagnostics = None
        started = perf_counter()
        entered = False
        try:
            async with self._compiler.open_session():
                entered = True
                return await self._strategy.solve(problem)
        finally:
            if entered and self._strategy.diagnostics is not None:
                self.diagnostics = replace(
                    self._strategy.diagnostics, wall_time_s=perf_counter() - started
                )
