from typing import override

from controller.application.optimization.abc.formulation_data_builder import (
    FormulationDataBuilder,
)
from controller.application.optimization.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from controller.application.optimization.abc.warm_start_alternated_formulation_config_builder import (
    WarmStartFormulationConfigBuilder,
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
from controller.domain.optimization.solving.strategy.alternated_strategy_config import (
    AlternatedStrategyConfig,
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
        warm_start_formulation_config_builder: WarmStartFormulationConfigBuilder[
            FormulationConfigT
        ],
        formulation_config: FormulationConfigT,
        alternated_strategy_config: AlternatedStrategyConfig,
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
        self._alternated_strategy_config = alternated_strategy_config

    @override
    async def solve(
        self,
        requests_by_flow: dict[FlowId, OptimizationRequest],
        context: OptimizationContext,
    ) -> OptimizationSolution:

        ## Build warm start model
        model_formulation_data = self._formulation_data_builder.build_formulation_data(
            requests_by_flow, context, self._formulation_config
        )

        raise SolveError()
