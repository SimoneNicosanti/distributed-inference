from typing import override

from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
)
from optimizer.application.abc.formulation_builder import (
    FormulationBuilder,
)
from optimizer.application.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
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
from optimizer.application.strategies.alternated.alternated_strategy_config import (
    AlternatedStrategyConfig,
)
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)
from shared.flow.flow import FlowId


class AlternatedOptimizationSolverStrategy[
    FormulationConfigT,
    FormulationDataT,
    FormulationModelT,
    SolverConfigT,
    SolvedFormulationModelT,
    SolverStatisticsT,
](SolverStrategy):
    def __init__(
        self,
        formulation_data_builder: FormulationBuilder[
            FormulationDataT, FormulationConfigT
        ],
        formulation_model_builder: FormulationCompiler[
            FormulationDataT, FormulationModelT
        ],
        solver_config_builder: SolverConfigBuilder[SolverConfigT],
        solver: Solver[
            FormulationModelT, SolverConfigT, SolvedFormulationModelT, SolverStatisticsT
        ],
        solved_model_translator: SolvedCompiledFormulationTranslator[
            SolvedFormulationModelT
        ],
        warm_start_formulation_config_builder: WarmStartFormulationConfigBuilder[
            FormulationConfigT
        ],
        warm_start_solver_config_builder: SolverConfigBuilder[SolverConfigT],
        warm_start_applier: WarmStartApplier[FormulationModelT],
        formulation_config: FormulationConfigT,
        alternated_strategy_config: AlternatedStrategyConfig,
        phase_generator: PhaseGenerator[FormulationModelT],
        phase_constraints_applier: PhaseConstraintsApplier[FormulationModelT],
    ) -> None:
        super().__init__()
        self._formulation_data_builder = formulation_data_builder
        self._formulation_model_builder = formulation_model_builder
        self._solver_config_builder = solver_config_builder
        self._solver = solver
        self._solved_model_translator = solved_model_translator
        self._formulation_config = formulation_config

        self._warm_start_formulation_config_builder = (
            warm_start_formulation_config_builder
        )
        self._warm_start_applier = warm_start_applier
        self._warm_start_solver_config_builder = warm_start_solver_config_builder

        self._phase_generator = phase_generator
        self._phase_constraints_applier = phase_constraints_applier

    @override
    async def solve(
        self,
        requests_by_flow: dict[FlowId, OptimizationRequest],
        context: OptimizationContext,
    ) -> OptimizationSolution:

        warm_start_solution = await self._warm_start_search(requests_by_flow, context)

        formulation_data = self._formulation_data_builder.build_formulation(
            requests_by_flow, context, self._formulation_config
        )
        async with self._formulation_model_builder.compile_formulation(
            formulation_data
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

    async def _warm_start_search(
        self,
        requests_by_flow: dict[FlowId, OptimizationRequest],
        context: OptimizationContext,
    ) -> OptimizationSolution:

        warm_start_solver_config = (
            self._warm_start_solver_config_builder.build_solver_config()
        )
        for (
            warm_start_formulation_config
        ) in self._warm_start_formulation_config_builder.warm_start_formulation_configs(
            self._formulation_config
        ):
            warm_start_formulation_data = (
                self._formulation_data_builder.build_formulation(
                    requests_by_flow, context, warm_start_formulation_config
                )
            )
            async with self._formulation_model_builder.compile_formulation(
                warm_start_formulation_data
            ) as warm_start_formulation_model:
                warm_start_solver_output = (
                    await self._solver.solve_compiled_formulation(
                        warm_start_formulation_model, warm_start_solver_config
                    )
                )

                if warm_start_solver_output.solved_model is not None:
                    return await self._solved_model_translator.translate_solved_formulation_model(
                        warm_start_solver_output.solved_model
                    )

        raise SolveError("Could not find a valid warm start point")
