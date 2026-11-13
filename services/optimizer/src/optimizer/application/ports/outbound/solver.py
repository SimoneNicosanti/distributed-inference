from abc import ABC, abstractmethod

from optimizer.application.ports.outbound.solver_result import SolverOutput


class Solver[
    CompiledFormulationT,
    SolverConfigT,
    SolvedCompiledFormulationT,
    SolverStatisticsT,
](ABC):
    @abstractmethod
    async def solve_compiled_formulation(
        self, compiled_formulation: CompiledFormulationT, solver_config: SolverConfigT
    ) -> SolverOutput[SolvedCompiledFormulationT, SolverStatisticsT]: ...
