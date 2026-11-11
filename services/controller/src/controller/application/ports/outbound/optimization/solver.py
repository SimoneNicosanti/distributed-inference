from abc import ABC, abstractmethod

from controller.domain.optimization.solving.solver_result import SolverOutput


class Solver[
    FormulationModelT,
    SolverConfigT,
    SolvedFormulationModelT,
    SolverStatisticsT,
](ABC):
    @abstractmethod
    async def solve_formulation_model(
        self, model: FormulationModelT, config: SolverConfigT
    ) -> SolverOutput[SolvedFormulationModelT, SolverStatisticsT]: ...
