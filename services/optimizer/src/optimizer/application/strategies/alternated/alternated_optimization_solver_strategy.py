from typing import override

from optimizer.application.abc.formulation_builder import (
    FormulationBuilder,
)
from optimizer.application.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedCompiledFormulationTranslator,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_config_builder import (
    SolverConfigBuilder,
)
from optimizer.application.strategies.alternated.abc.phase_constraints_applier import (
    PhaseConstraintsApplier,
)
from optimizer.application.strategies.alternated.abc.phase_generator import (
    PhaseGenerationContext,
    PhaseGenerator,
)
from optimizer.application.strategies.alternated.abc.warm_start_applier import (
    WarmStartApplier,
)
from optimizer.application.strategies.alternated.abc.warm_start_searcher import (
    WarmStartSearcher,
)
from optimizer.application.strategies.alternated.alternated_strategy_config import (
    AlternatedStrategyConfig,
)
from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.formulation.formulation_config import FormulationConfig
from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class AlternatedOptimizationSolverStrategy[
    OptimizationProblemT: OptimizationProblem,
    FormulationT: Formulation,
    FormulationConfigT: FormulationConfig,
    CompiledFormulationT,
    SolverConfigT,
    SolvedCompiledFormulationT,
    SolverStatisticsT,
](SolverStrategy[OptimizationProblemT]):
    def __init__(
        self,
        formulation_data_builder: FormulationBuilder[
            OptimizationProblemT, FormulationT, FormulationConfigT
        ],
        formulation_model_builder: FormulationCompiler[
            FormulationT, CompiledFormulationT
        ],
        solver_config_builder: SolverConfigBuilder[SolverConfigT],
        solver: Solver[
            CompiledFormulationT,
            SolverConfigT,
            SolvedCompiledFormulationT,
            SolverStatisticsT,
        ],
        solved_model_translator: SolvedCompiledFormulationTranslator[
            SolvedCompiledFormulationT
        ],
        formulation_config: FormulationConfigT,
        warm_start_searcher: WarmStartSearcher[OptimizationProblemT],
        warm_start_applier: WarmStartApplier[CompiledFormulationT],
        alternated_strategy_config: AlternatedStrategyConfig,
        phase_generator: PhaseGenerator[CompiledFormulationT],
        phase_constraints_applier: PhaseConstraintsApplier[CompiledFormulationT],
    ) -> None:
        super().__init__()
        self._formulation_data_builder = formulation_data_builder
        self._formulation_model_builder = formulation_model_builder
        self._solver_config_builder = solver_config_builder
        self._solver = solver
        self._solved_model_translator = solved_model_translator
        self._formulation_config = formulation_config

        self._warm_start_searcher = warm_start_searcher
        self._warm_start_applier = warm_start_applier

        self._phase_generator = phase_generator
        self._phase_constraints_applier = phase_constraints_applier

    @override
    async def solve(
        self,
        problem: OptimizationProblemT,
    ) -> OptimizationSolution:

        warm_start_solution = await self._warm_start_searcher.search_warm_start(problem)

        formulation = self._formulation_data_builder.build_formulation(
            problem, self._formulation_config
        )
        async with self._formulation_model_builder.compile_formulation(
            formulation
        ) as formulation_model:
            incumbent = warm_start_solution
            while True:
                phase_context = PhaseGenerationContext()
                phase_info = self._phase_generator.next_phase(phase_context)

                if phase_info.ends:
                    break

                await self._warm_start_applier.apply_warm_start(
                    formulation_model, incumbent
                )

                async with self._phase_constraints_applier.apply_next_phase_constraints(
                    formulation_model, incumbent, phase_info
                ):
                    solved_formulation = await self._solver.solve_compiled_formulation(
                        formulation_model,
                        self._solver_config_builder.build_solver_config(),
                    )

                    if solved_formulation.solved_model is not None:
                        incumbent = await self._solved_model_translator.translate_solved_formulation_model(
                            solved_formulation.solved_model
                        )

        raise SolveError()
