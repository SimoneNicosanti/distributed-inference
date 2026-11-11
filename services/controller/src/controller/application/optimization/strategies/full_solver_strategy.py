from typing import override

from controller.application.optimization.abc.formulation_data_builder import (
    FormulationDataBuilder,
)
from controller.application.optimization.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from controller.application.ports.outbound.optimization.formulation_model_builder import (
    FormulationModelBuilder,
)
from controller.application.ports.outbound.optimization.solved_formulation_model_translator import (
    SolvedFormulationModelTranslator,
)
from controller.application.ports.outbound.optimization.solver import Solver
from controller.application.ports.outbound.optimization.solver_config_builder import (
    SolverConfigBuilder,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from shared.flow.flow import FlowId


class FullSolverStrategy[
    FormulationConfigT,
    FormulationDataT,
    FormulationModelT,
    SolverConfigT,
    SolvedFormulationModelT,
    SolverStatisticsT,
](SolverStrategy):
    def __init__(
        self,
        formulation_data_builder: FormulationDataBuilder[
            FormulationDataT, FormulationConfigT
        ],
        formulation_model_builder: FormulationModelBuilder[
            FormulationDataT, FormulationModelT
        ],
        solver_config_builder: SolverConfigBuilder[SolverConfigT],
        solver: Solver[
            FormulationModelT, SolverConfigT, SolvedFormulationModelT, SolverStatisticsT
        ],
        solved_model_translator: SolvedFormulationModelTranslator[
            SolvedFormulationModelT
        ],
        formulation_config: FormulationConfigT,
    ) -> None:
        super().__init__()
        self._formulation_data_builder = formulation_data_builder
        self._formulation_model_builder = formulation_model_builder
        self._solver_config_builder = solver_config_builder
        self._solver = solver
        self._solved_model_translator = solved_model_translator
        self._formulation_config = formulation_config

    @override
    async def solve(
        self,
        requests_by_flow: dict[FlowId, OptimizationRequest],
        context: OptimizationContext,
    ) -> OptimizationSolution:

        formulation_data = self._formulation_data_builder.build_formulation_data(
            requests_by_flow, context, self._formulation_config
        )
        async with self._formulation_model_builder.build_formulation_model(
            formulation_data
        ) as formulation_model:
            solver_config = self._solver_config_builder.build_solver_config()

            solver_output = await self._solver.solve_formulation_model(
                formulation_model, solver_config
            )
            if solver_output.solved_model is not None:
                return await self._solved_model_translator.translate_solved_formulation_model(
                    solver_output.solved_model
                )

            raise SolveError()
