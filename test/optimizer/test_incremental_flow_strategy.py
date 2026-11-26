"""State-machine tests using real prefix formulation and controlled solver outputs."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from test.optimizer.test_cumulative_formulation import LAYERS, dag_problem

from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedCompiledFormulationTranslator,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_result import SolverOutput
from optimizer.application.ports.outbound.solver_result import (
    SolverTerminationKind as Termination,
)
from optimizer.application.strategies.incremental.compact_problem_prefix import (
    compact_problem_prefix,
)
from optimizer.application.strategies.incremental.incremental_flow_diagnostics import (
    IncrementalStage,
)
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolveError,
    IncrementalFlowSolverStrategy,
    IncrementalFlowStrategyConfig,
)
from optimizer.application.strategies.incremental.incremental_model_adapter import (
    IncrementalModelAdapter,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from optimizer.domain.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    FlowOptimizationSolution,
    OptimizationSolution,
)


def solution_for(problem, objective=1.0):
    flows = []
    for flow, request in problem.flows.items():
        key = request.configurations[0]
        worker = request.workers[0]
        component = ComponentSolution(
            1,
            worker,
            problem.workers[worker].backend,
            LAYERS,
            ("input",),
            ("output", "output2"),
            ComponentTensorContract((), ()),
            1,
            0.01,
            0,
            0,
            0,
        )
        flows.append(
            FlowOptimizationSolution(
                flow,
                worker,
                worker,
                key.model_variant_id,
                (component,),
                (),
                0,
                0.01,
                key.input_shapes.input_shapes,
            )
        )
    return OptimizationSolution(tuple(flows), objective, True)


@dataclass
class Outcome:
    termination: Termination = Termination.OPTIMAL
    incumbent: bool = True
    objective: float = 1.0
    valid: bool = True
    error: Exception | None = None


class FakeCompiler(FormulationCompiler):
    def __init__(self):
        self.models = []

    @asynccontextmanager
    async def compile_formulation(self, formulation) -> AsyncGenerator:
        model = SimpleNamespace(
            formulation=formulation, alive=True, seed=None, fixed=False
        )
        self.models.append(model)
        try:
            yield model
        finally:
            model.alive = False


class FakeSolver(Solver):
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    async def solve_compiled_formulation(self, model, config):
        assert model.alive
        self.calls.append((model, config))
        outcome = next(self.outcomes)
        if outcome.error:
            raise outcome.error
        model.valid = outcome.valid
        model.solution = replace(
            solution_for(model.formulation.problem, outcome.objective),
            is_optimal=outcome.termination == Termination.OPTIMAL,
        )
        return SolverOutput(
            outcome.termination,
            model if outcome.incumbent else None,
            outcome.objective if outcome.incumbent else None,
            None,
            0.2,
            SimpleNamespace(
                time_to_first_incumbent_s=0.03 if outcome.incumbent else None
            ),
        )


class FakeAdapter(IncrementalModelAdapter):
    async def initialize_model(self, model, solution):
        assert model.alive
        model.seed = solution

    def is_valid_solution(self, model):
        assert model.alive
        return model.valid


class FakeTranslator(SolvedCompiledFormulationTranslator):
    async def translate_solved_formulation_model(self, model):
        assert model.alive
        return model.solution


def strategy(outcomes):
    compiler = FakeCompiler()
    solver = FakeSolver(outcomes)
    observer = []
    result = IncrementalFlowSolverStrategy(
        formulation_builder=MisocpCompactFormulationBuilder(),
        formulation_config=MisocpCompactFormulationConfig(0.5),
        formulation_compiler=compiler,
        config=IncrementalFlowStrategyConfig("insertion"),
        solver=solver,
        solved_model_translator=FakeTranslator(),
        model_adapter=FakeAdapter(),
        observer=observer.append,
    )
    return result, compiler, solver, observer


@pytest.mark.asyncio
async def test_each_prefix_is_unrestricted_and_receives_only_previous_start():
    problem = dag_problem(variants=1, flows=3)
    runner, compiler, solver, observed = strategy([Outcome() for _ in range(3)])
    result = await runner.solve(problem)
    assert len(result.flow_solutions) == 3
    assert result.is_optimal
    assert [len(m.formulation.problem.flows) for m in compiler.models] == [1, 2, 3]
    assert not any(m.fixed for m in compiler.models)
    assert [c for _, c in solver.calls] == ["insertion"] * 3
    assert [len(m.seed.flow_solutions) if m.seed else 0 for m in compiler.models] == [
        0,
        1,
        2,
    ]
    assert compiler.models[1].seed is compiler.models[0].solution
    assert compiler.models[2].seed is compiler.models[1].solution
    assert result is compiler.models[-1].solution
    assert all(not m.alive for m in compiler.models)
    diagnostics = runner.diagnostics
    assert diagnostics.successful_insertions == 2
    assert diagnostics.as_record()["native_solve_count"] == 3
    assert "prefix_wall_time_s" in diagnostics.as_record()
    assert "incremental_construction_time_s" not in diagnostics.as_record()
    assert diagnostics.returned_source == "last_prefix"
    assert observed == [diagnostics]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "termination",
    [Termination.INFEASIBLE, Termination.TIME_LIMIT, Termination.NUMERICAL_FAILURE],
)
async def test_failed_prefix_stops_without_retry_or_smaller_solution(termination):
    runner, compiler, _, _ = strategy([Outcome(), Outcome(termination, False)])
    with pytest.raises(IncrementalFlowSolveError) as failure:
        await runner.solve(dag_problem(variants=1, flows=3))
    assert failure.value.termination == termination
    assert failure.value.attempt.stage == IncrementalStage.INSERTION
    assert runner.diagnostics.completed_flow_count == 1
    assert runner.diagnostics.returned_source is None
    assert len(compiler.models) == 2
    assert not any(m.alive for m in compiler.models)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "objective,before,after",
    [
        (ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME, 1, 2),
        (ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY, 2, 1),
    ],
)
async def test_last_prefix_is_returned_without_comparing_different_problems(
    objective, before, after
):
    base = dag_problem(variants=1, flows=2)
    problem = replace(base, policy=replace(base.policy, objective=objective))
    runner, compiler, _, _ = strategy(
        [Outcome(objective=before), Outcome(objective=after)]
    )
    result = await runner.solve(problem)
    assert result.objective_value == after
    assert len(result.flow_solutions) == 2
    assert len(compiler.models) == 2
    assert runner.diagnostics.returned_source == "last_prefix"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "termination",
    [Termination.TIME_LIMIT, Termination.INFEASIBLE, Termination.NUMERICAL_FAILURE],
)
async def test_first_flow_without_incumbent_is_explicit_failure(termination):
    runner, compiler, _, _ = strategy([Outcome(termination, False)])
    with pytest.raises(IncrementalFlowSolveError) as failure:
        await runner.solve(dag_problem(variants=1))
    assert failure.value.termination == termination
    assert len(compiler.models) == 1
    assert runner.diagnostics.completed_flow_count == 0


@pytest.mark.asyncio
async def test_invalid_insertion_incumbent_is_not_accepted():
    runner, _, _, _ = strategy(
        [Outcome(), Outcome(Termination.SOLUTION_LIMIT, True, valid=False)]
    )
    with pytest.raises(IncrementalFlowSolveError):
        await runner.solve(dag_problem(variants=1, flows=2))
    assert runner.diagnostics.attempts[-1].error_type == "ValueError"


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_model():
    import asyncio

    runner, compiler, _, _ = strategy([Outcome(error=asyncio.CancelledError())])
    with pytest.raises(asyncio.CancelledError):
        await runner.solve(dag_problem(variants=1))
    assert not compiler.models[0].alive
    assert runner.diagnostics.attempts[0].error_type == "CancelledError"


@pytest.mark.asyncio
async def test_empty_problem_rejected_before_any_solve():
    runner, compiler, _, _ = strategy([])
    with pytest.raises(ValueError, match="at least one flow"):
        await runner.solve(replace(dag_problem(variants=1), flows={}))
    assert not compiler.models


def test_prefix_preserves_identity_of_profiles_and_original_flow_parameters():
    problem = dag_problem(variants=1, flows=3)
    for count in (1, 2, 3):
        prefix = compact_problem_prefix(problem, count)
        assert tuple(prefix.flows) == tuple(problem.flows)[:count]
        assert all(prefix.flows[f] is problem.flows[f] for f in prefix.flows)
        for name in (
            "variants",
            "configurations",
            "execution_times_s",
            "workers",
            "connections",
            "policy",
        ):
            assert getattr(prefix, name) is getattr(problem, name)
    assert compact_problem_prefix(problem, 3) is problem
    for count in (0, 4, True):
        with pytest.raises(ValueError):
            compact_problem_prefix(problem, count)


@pytest.mark.asyncio
async def test_timeout_with_valid_incumbent_continues_to_next_unrestricted_prefix():
    runner, _, _, _ = strategy(
        [
            Outcome(Termination.TIME_LIMIT, True),
            Outcome(Termination.TIME_LIMIT, True),
        ]
    )
    result = await runner.solve(dag_problem(variants=1, flows=2))
    assert len(result.flow_solutions) == 2
    assert len(runner.diagnostics.attempts) == 2
    assert runner.diagnostics.successful_insertions == 1
    assert not result.is_optimal


@pytest.mark.asyncio
async def test_invalid_flow_reports_formulation_error_before_solver_call():
    problem = dag_problem(variants=1)
    flow, request = next(iter(problem.flows.items()))
    problem = replace(problem, flows={flow: replace(request, workers=())})
    runner, _, solver, _ = strategy([])
    with pytest.raises(IncrementalFlowSolveError) as failure:
        await runner.solve(problem)
    assert failure.value.attempt.error_type == "MisocpCompactFormulationBuildError"
    assert not solver.calls
    assert runner.diagnostics.as_record()["native_solve_count"] == 0
