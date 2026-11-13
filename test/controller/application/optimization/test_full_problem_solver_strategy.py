from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast, override

import pytest

from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from optimizer.application.abc.formulation_builder import (
    FormulationBuilder,
)
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedFormulationModelTranslator,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_config_builder import (
    SolverConfigBuilder,
)
from optimizer.application.ports.outbound.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)
from optimizer.application.strategies.full_solver_strategy import (
    FullSolverStrategy,
)
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
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

    class DataBuilder(FormulationBuilder[object, object]):
        @override
        def build_formulation(
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

    class ModelBuilder(FormulationCompiler[object, object]):
        @override
        @asynccontextmanager
        async def compile_formulation(self, data: object) -> AsyncIterator[object]:
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
        async def solve_compiled_formulation(
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
        formulation_builder=DataBuilder(),
        formulation_compiler=ModelBuilder(),
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
