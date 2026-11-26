"""Native persistent-prefix equivalence, retained objects, bounds and lifecycle."""

import asyncio
from dataclasses import replace
from types import MappingProxyType

import gurobipy as gp
import pytest
from test.optimizer.test_component_priority_regression import build
from test.optimizer.test_cumulative_formulation import dag_problem
from test.optimizer.test_full_optimizer_runtime import compiler

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.persistent_compact_formulation_compiler import (
    GurobiPersistentCompactFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.adapters.outbound.gurobi.strategy_factory import (
    build_compact_gurobi_solver_strategy,
)
from optimizer.application.strategies.incremental.compact_problem_prefix import (
    compact_problem_prefix,
)
from optimizer.application.strategies.incremental.persistent_incremental_flow_solver_strategy import (
    PersistentIncrementalFlowSolverStrategy,
)
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    FlowEndpointPlacement,
)


def structure(model):
    def linear(expr):
        return tuple(
            sorted(
                (expr.getVar(i).VarName, expr.getCoeff(i)) for i in range(expr.size())
            )
        )

    variables = tuple(
        sorted((v.VarName, v.VType, v.LB, v.UB, v.Obj) for v in model.getVars())
    )
    rows = tuple(
        sorted(
            (c.ConstrName, c.Sense, c.RHS, linear(model.getRow(c)))
            for c in model.getConstrs()
        )
    )
    quadratics = []
    for c in model.getQConstrs():
        expr = model.getQCRow(c)
        terms = tuple(
            sorted(
                (
                    tuple(sorted((expr.getVar1(i).VarName, expr.getVar2(i).VarName))),
                    expr.getCoeff(i),
                )
                for i in range(expr.size())
            )
        )
        quadratics.append(
            (c.QCName, c.QCSense, c.QCRHS, linear(expr.getLinExpr()), terms)
        )
    indicators = []
    for c in model.getGenConstrs():
        assert c.GenConstrType == gp.GRB.GENCONSTR_INDICATOR
        variable, value, expr, sense, rhs = model.getGenConstrIndicator(c)
        indicators.append(
            (c.GenConstrName, variable.VarName, value, linear(expr), sense, rhs)
        )
    assert len({v[0] for v in variables}) == len(variables)
    assert len({r[0] for r in rows}) == len(rows)
    return (
        variables,
        rows,
        tuple(sorted(quadratics)),
        tuple(sorted(indicators)),
        model.ObjCon,
        model.ModelSense,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("objective", list(ObjectiveKind))
@pytest.mark.parametrize("read_only_layer_mapping", [False, True])
async def test_every_extended_prefix_matches_fresh_model_and_retains_native_objects(
    kind, objective, read_only_layer_mapping
):
    base = dag_problem(variants=2, workers=2, components=2, flows=3, priorities=2)
    # Force index reordering in the standard prepared formulation at every arrival.
    problem = replace(
        base,
        flows={f: base.flows[f] for f in sorted(base.flows, key=str, reverse=True)},
        policy=replace(base.policy, objective=objective),
    )
    persistent = GurobiPersistentCompactFormulationCompiler()
    previous_variables, previous_row, native = [], None, None
    async with persistent.open_session():
        assert persistent._session is None
        for count in (1, 2, 3):
            data = build(compact_problem_prefix(problem, count), kind)
            async with persistent.compile_formulation(data) as current:
                if read_only_layer_mapping:
                    assignment = current.variables.assignment
                    assignment.layer_component = MappingProxyType(
                        assignment.layer_component
                    )
                if native is not None:
                    assert current.native_model is native
                    assert all(
                        current.native_model.getVarByName(v.VarName).sameAs(v)
                        for v in previous_variables
                    )
                    assert current.native_model.getConstrByName(
                        previous_row.ConstrName
                    ).sameAs(previous_row)
                native = current.native_model
                previous_variables = native.getVars()
                previous_row = native.getConstrs()[0]
                async with compiler().compile_formulation(
                    replace(data, flow_index=current.formulation.flow_index)
                ) as fresh:
                    assert structure(native) == structure(fresh.native_model)
                assert len(current.formulation.flows) == count
    assert persistent._session is None


class RecordingSolver:
    def __init__(self, cancel=False):
        self.models = []
        self.queue_bounds = []
        self.cancel = cancel
        self.native = GurobiSolver()

    async def solve_compiled_formulation(self, model, config):
        self.models.append(model.native_model)
        self.queue_bounds.append(
            {
                key: variable.UB
                for key, variable in model.variables.compute_queue.queue_wait_s.items()
            }
        )
        if self.cancel and len(self.models) == 2:
            raise asyncio.CancelledError
        return await self.native.solve_compiled_formulation(model, config)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_native_persistent_strategy_solves_all_prefixes_and_matches_full(
    kind, objective
):
    base = dag_problem(variants=2, workers=1, components=1, flows=3, priorities=2)
    problem = replace(base, policy=replace(base.policy, objective=objective))
    config = GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0, solution_limit=1)
    recorder = RecordingSolver()
    runner = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3, kind),
        config,
        strategy_config=GurobiStrategyConfig("incremental-persistent"),
        solver=recorder,
    )
    assert isinstance(runner, PersistentIncrementalFlowSolverStrategy)
    result = await runner.solve(problem)
    reference = await build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3, kind), replace(config, solution_limit=None)
    ).solve(problem)
    assert result.objective_value == pytest.approx(reference.objective_value, abs=1e-6)
    assert {f.flow_id for f in result.flow_solutions} == set(problem.flows)
    assert result.is_optimal
    assert len(recorder.models) == 3 and all(
        m is recorder.models[0] for m in recorder.models
    )
    assert [a.flow_count for a in runner.diagnostics.attempts] == [1, 2, 3]
    assert runner.diagnostics.as_record()["native_solve_count"] == 3
    assert runner._compiler._session is None


@pytest.mark.asyncio
async def test_session_closes_on_cancellation_and_strategy_can_be_reused():
    problem = dag_problem(variants=1, flows=2, workers=1, components=1)
    recorder = RecordingSolver(cancel=True)
    runner = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3),
        GurobiSolverConfig(time_limit_s=5, threads=1),
        strategy_config=GurobiStrategyConfig("incremental-persistent"),
        solver=recorder,
    )
    with pytest.raises(asyncio.CancelledError):
        await runner.solve(problem)
    assert runner._compiler._session is None
    assert runner.diagnostics.completed_flow_count == 1
    recorder.cancel = False
    result = await runner.solve(problem)
    assert len(result.flow_solutions) == 2
    assert recorder.models[-1] is not recorder.models[0]
    assert runner._compiler._session is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_extension_updates_shared_network_with_actual_transmissions(kind):
    base = dag_problem(variants=1, workers=2, components=2, flows=2, priorities=2)
    source, target = tuple(base.workers)
    problem = replace(
        base,
        flows={
            flow: replace(
                request,
                endpoint_placement=FlowEndpointPlacement(
                    FixedPlacement(source), FixedPlacement(target)
                ),
            )
            for flow, request in base.flows.items()
        },
    )
    config = GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0)
    recorder = RecordingSolver()
    persistent = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3, kind),
        config,
        strategy_config=GurobiStrategyConfig("incremental-persistent"),
        solver=recorder,
    )
    result = await persistent.solve(problem)
    reference = await build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3, kind), config
    ).solve(problem)
    assert result.objective_value == pytest.approx(reference.objective_value, abs=1e-6)
    assert len(recorder.models) == 2 and recorder.models[0] is recorder.models[1]
    for flow in result.flow_solutions:
        assert flow.components[0].worker_id == source
        assert flow.components[-1].worker_id == target
        assert flow.components[0].transmission_latency_s > 0
        assert flow.components[0].mean_transmission_time_s > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_previously_unused_worker_bounds_increase_when_new_flow_arrives(kind):
    base = dag_problem(variants=1, workers=2, components=1, flows=2, priorities=2)
    workers = tuple(base.workers)
    problem = replace(
        base,
        flows={
            flow: replace(
                request,
                workers=(workers[index],),
                endpoint_placement=FlowEndpointPlacement(
                    AnyPlacement(), AnyPlacement()
                ),
            )
            for index, (flow, request) in enumerate(base.flows.items())
        },
    )
    config = GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0)
    recorder = RecordingSolver()
    runner = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3, kind),
        config,
        strategy_config=GurobiStrategyConfig("incremental-persistent"),
        solver=recorder,
    )
    result = await runner.solve(problem)
    reference = await build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.3, kind), config
    ).solve(problem)
    assert recorder.queue_bounds[0][workers[1], 1] == 0
    assert recorder.queue_bounds[1][workers[1], 1] > 0
    assert result.objective_value == pytest.approx(reference.objective_value, abs=1e-6)
    assert result.is_optimal


@pytest.mark.asyncio
async def test_compiler_requires_session_and_rejects_changes_to_existing_flows():
    problem = dag_problem(variants=1, flows=2, workers=1, components=1)
    persistent = GurobiPersistentCompactFormulationCompiler()
    first = build(compact_problem_prefix(problem, 1), Kind.CUMULATIVE)
    with pytest.raises(RuntimeError, match="active session"):
        async with persistent.compile_formulation(first):
            pass
    async with persistent.open_session():
        async with persistent.compile_formulation(first) as current:
            original = structure(current.native_model)
        flow = next(iter(first.problem.flows))
        changed = replace(
            problem,
            flows={
                f: replace(r, arrival_rate_per_s=r.arrival_rate_per_s * 2)
                if f == flow
                else r
                for f, r in problem.flows.items()
            },
        )
        with pytest.raises(ValueError, match="requests must remain unchanged"):
            async with persistent.compile_formulation(build(changed, Kind.CUMULATIVE)):
                pass
        assert structure(current.native_model) == original
