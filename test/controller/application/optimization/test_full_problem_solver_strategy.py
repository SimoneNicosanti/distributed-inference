from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast, override

import pytest

from controller.application.optimization.abc.formulation_data_builder import (
    FormulationDataBuilder,
)
from controller.application.optimization.strategies.full_solver_strategy import (
    FullSolverStrategy,
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
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.optimization.solving.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)


class _StageError(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "build", "solve", "translate"])
async def test_prepares_data_and_keeps_model_alive_through_translation(
    failure: str | None,
) -> None:
    events: list[str] = []
    data = object()
    model = object()
    result = object()
    solution = cast(OptimizationSolution, object())
    requests: OptimizationRequestsByFlow = {}
    context = cast(OptimizationContext, object())
    formulation_config = object()
    solver_config = object()

    class DataBuilder(FormulationDataBuilder[object, object]):
        @override
        def build_formulation_data(
            self,
            requests_by_flow: OptimizationRequestsByFlow,
            context: OptimizationContext,
            formulation_config: object,
        ) -> object:
            assert requests_by_flow is requests
            assert context is expected_context
            assert formulation_config is expected_formulation_config
            events.append("prepare")
            return data

    class ModelBuilder(FormulationModelBuilder[object, object]):
        @override
        @asynccontextmanager
        async def build_formulation_model(self, data: object) -> AsyncIterator[object]:
            assert data is expected_data
            events.append("open")
            try:
                if failure == "build":
                    raise _StageError
                yield model
            finally:
                events.append("close")

    class ConfigBuilder(SolverConfigBuilder[object]):
        @override
        def build_solver_config(self) -> object:
            assert "close" not in events
            return solver_config

    class ModelSolver(Solver[object, object, object, None]):
        @override
        async def solve_formulation_model(
            self, model: object, config: object
        ) -> SolverOutput[object, None]:
            assert model is expected_model
            assert config is solver_config
            assert "close" not in events
            events.append("solve")
            if failure == "solve":
                raise _StageError
            return SolverOutput(
                termination=SolverTerminationKind.OPTIMAL,
                solved_model=result,
                objective_value=0.0,
                objective_bound=0.0,
                runtime_s=0.0,
                statistics=None,
            )

    class Translator(SolvedFormulationModelTranslator[object]):
        @override
        async def translate_solved_formulation_model(
            self, solved_formulation_model: object
        ) -> OptimizationSolution:
            assert solved_formulation_model is result
            assert "close" not in events
            events.append("translate")
            if failure == "translate":
                raise _StageError
            return solution

    expected_context = context
    expected_data = data
    expected_model = model
    expected_formulation_config = formulation_config
    strategy = FullSolverStrategy(
        formulation_data_builder=DataBuilder(),
        formulation_model_builder=ModelBuilder(),
        solver_config_builder=ConfigBuilder(),
        solver=ModelSolver(),
        solved_model_translator=Translator(),
        formulation_config=formulation_config,
    )
    if failure is None:
        assert await strategy.solve(requests, context) is solution
    else:
        with pytest.raises(_StageError):
            await strategy.solve(requests, context)

    expected = ["prepare", "open"]
    if failure != "build":
        expected.append("solve")
    if failure not in {"build", "solve"}:
        expected.append("translate")
    assert events == [*expected, "close"]
