from typing import override

from optimizer.application.abc.formulation_builder import FormulationBuilder
from optimizer.application.abc.solver_strategy import SolveError
from optimizer.application.abc.warm_start_alternated_formulation_config_builder import (
    WarmStartFormulationConfigBuilder,
)
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedCompiledFormulationTranslator,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.strategies.alternated.abc.warm_start_searcher import (
    WarmStartSearcher,
)
from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.formulation.formulation_config import FormulationConfig
from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class DefaultWarmStartSearcher[
    OptimizationProblemT: OptimizationProblem,
    FormulationT: Formulation,
    FormulationConfigT: FormulationConfig,
    CompiledFormulationT,
    SolverConfigT,
    SolvedCompiledFormulationT,
    SolverStatisticsT,
](WarmStartSearcher[OptimizationProblemT]):
    def __init__(
        self,
        full_formulation_config: FormulationConfigT,
        warm_start_formulation_config_builder: WarmStartFormulationConfigBuilder[
            FormulationConfigT
        ],
        warm_start_formulation_builder: FormulationBuilder[
            OptimizationProblemT, FormulationT, FormulationConfigT
        ],
        formulation_compiler: FormulationCompiler[FormulationT, CompiledFormulationT],
        warm_start_solver_config: SolverConfigT,
        solver: Solver[
            CompiledFormulationT,
            SolverConfigT,
            SolvedCompiledFormulationT,
            SolverStatisticsT,
        ],
        solved_model_translator: SolvedCompiledFormulationTranslator[
            SolvedCompiledFormulationT
        ],
    ) -> None:
        super().__init__()
        self._full_formulation_config = full_formulation_config
        self._warm_start_formulation_config_builder = (
            warm_start_formulation_config_builder
        )

        self._warm_start_formulation_builder = warm_start_formulation_builder
        self._formulation_compiler = formulation_compiler
        self._warm_start_solver_config = warm_start_solver_config
        self._solver = solver
        self._solved_model_translator = solved_model_translator

    @override
    async def search_warm_start(
        self, problem: OptimizationProblemT
    ) -> OptimizationSolution:

        for (
            warm_start_formulation_config
        ) in self._warm_start_formulation_config_builder.warm_start_formulation_configs(
            self._full_formulation_config
        ):
            warm_start_formulation = (
                self._warm_start_formulation_builder.build_formulation(
                    problem, warm_start_formulation_config
                )
            )
            async with self._formulation_compiler.compile_formulation(
                warm_start_formulation
            ) as warm_start_compiled_formulation:
                warm_start_solver_output = (
                    await self._solver.solve_compiled_formulation(
                        warm_start_compiled_formulation, self._warm_start_solver_config
                    )
                )

                if warm_start_solver_output.solved_model is not None:
                    return await self._solved_model_translator.translate_solved_formulation_model(
                        warm_start_solver_output.solved_model
                    )

        raise SolveError("Could not find a valid warm start point")
