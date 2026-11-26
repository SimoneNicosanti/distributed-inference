"""Exact source latency without pair variables, including inactive and empty cuts."""

from dataclasses import replace
from itertools import pairwise, product

import gurobipy as gp
import pytest

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FlowEndpointPlacement,
)

from .compact_fixtures import problem


@pytest.fixture
def licensed_gurobi():
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as error:
        if error.errno in (10009, 10010):
            pytest.skip("Gurobi license unavailable")
        raise


def latency_problem():
    p = problem(
        variants=2,
        workers=3,
        components=4,
        priorities=1,
        zero_service=True,
        objective=ObjectiveKind.CONSTANT,
    )
    return replace(
        p,
        configurations={
            key: replace(value, tensor_bytes=dict.fromkeys(value.tensor_bytes, 0))
            for key, value in p.configurations.items()
        },
        flows={
            flow: replace(
                request,
                endpoint_placement=FlowEndpointPlacement(
                    AnyPlacement(), AnyPlacement()
                ),
            )
            for flow, request in p.flows.items()
        },
        connections=tuple(
            replace(link, round_trip_time_s=0.0 if i == 0 else 0.02 + 0.007 * i)
            for i, link in enumerate(p.connections)
        ),
    )


def prepare(p, kind):
    return MisocpCompactFormulationBuilder().build_formulation(
        p, MisocpCompactFormulationConfig(0.1, kind)
    )


def pin(model, variables, data, selected, path):
    count = len(path)
    rows = []
    for (_f, key), value in variables.assignment.configuration_selected.items():
        rows.append(model.addConstr(value == int(key == selected)))
    for (_f, key, layer, k), value in variables.assignment.layer_component.items():
        graph = data.problem.variants[key.model_variant_id]
        label = (
            1
            if count == 1 or layer in graph.input_layers
            else count
            if layer in graph.output_layers
            else 2
        )
        rows.append(model.addConstr(value == int(key == selected and label == k)))
    for (_f, k, worker), value in variables.assignment.component_worker.items():
        rows.append(model.addConstr(value == int(k <= count and worker == path[k - 1])))
    return rows


@pytest.mark.parametrize("kind", list(CompactFormulationKind))
def test_latency_exact_for_every_path_under_minimization_and_maximization(
    licensed_gurobi, kind
):
    p = latency_problem()
    data = prepare(p, kind)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            variables = GurobiMisocpCompactFormulationModelBuilder().build_model_from_formulation(
                model, data
            )
            model.Params.Threads = 1
            model.Params.TimeLimit = 3
            model.Params.MIPGap = 0
            model.update()
            assert len(variables.network.source_latency_s) == (
                p.policy.max_components - 1
            ) * len(p.workers)
            assert not any(v.VarName.startswith("q[") for v in model.getVars())
            total = gp.quicksum(variables.network.source_latency_s.values())
            old_rows = []
            for count in (1, 2, 3):
                for path in product(data.workers, repeat=count):
                    if any(a == b for a, b in pairwise(path)):
                        continue
                    for selected in data.configurations_by_flow[data.flows[0]]:
                        model.remove(old_rows)
                        model.reset()
                        old_rows = pin(model, variables, data, selected, path)
                        expected = sum(
                            data.connections_by_pair[s, t].round_trip_time_s / 2
                            for s, t in pairwise(path)
                        )
                        for sense in (gp.GRB.MINIMIZE, gp.GRB.MAXIMIZE):
                            model.setObjective(total, sense)
                            model.optimize()
                            assert model.Status == gp.GRB.OPTIMAL
                            assert model.ObjVal == pytest.approx(expected, abs=1e-9)
                            for (
                                _flow,
                                k,
                                s,
                            ), value in variables.network.source_latency_s.items():
                                target = path[k] if k < count else None
                                latency = (
                                    data.connections_by_pair[
                                        s, target
                                    ].round_trip_time_s
                                    / 2
                                    if k < count and s == path[k - 1]
                                    else 0.0
                                )
                                assert pytest.approx(latency, abs=1e-9) == value.X


@pytest.mark.parametrize("kind", list(CompactFormulationKind))
@pytest.mark.parametrize(
    "case", ("local_without_links", "missing_forward", "reverse", "missing_second")
)
def test_direct_connectivity_constraints_preserve_missing_link_semantics(
    licensed_gurobi, kind, case
):
    p = latency_problem()
    a, b, c = tuple(p.workers)
    reverse = tuple(
        link for link in p.connections if (link.source, link.target) == (b, a)
    )
    paths = {
        "local_without_links": (a,),
        "missing_forward": (a, b),
        "reverse": (b, a),
        "missing_second": (b, a, c),
    }
    p = replace(p, connections=() if case == "local_without_links" else reverse)
    data = prepare(p, kind)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            variables = GurobiMisocpCompactFormulationModelBuilder().build_model_from_formulation(
                model, data
            )
            pin(
                model,
                variables,
                data,
                data.configurations_by_flow[data.flows[0]][0],
                paths[case],
            )
            model.Params.Threads = 1
            model.Params.TimeLimit = 3
            model.optimize()
            expected = (
                gp.GRB.OPTIMAL
                if case in ("local_without_links", "reverse")
                else gp.GRB.INFEASIBLE
            )
            assert model.Status == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(CompactFormulationKind))
@pytest.mark.parametrize(
    "objective", (ObjectiveKind.CONSTANT, ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY)
)
async def test_translated_latency_is_exact_and_independent_of_variant_and_volume(
    licensed_gurobi, kind, objective
):
    from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
        GurobiCompiledFormulation,
    )
    from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
        GurobiMisocpCompactSolvedCompiledFormulationTranslator,
    )

    p = latency_problem()
    p = replace(p, policy=replace(p.policy, objective=objective))
    data = prepare(p, kind)
    path = (data.workers[0], data.workers[1], data.workers[2])
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            variables = GurobiMisocpCompactFormulationModelBuilder().build_model_from_formulation(
                model, data
            )
            pin(
                model,
                variables,
                data,
                data.configurations_by_flow[data.flows[0]][1],
                path,
            )
            model.Params.Threads = 1
            model.Params.TimeLimit = 3
            model.optimize()
            assert model.Status == gp.GRB.OPTIMAL
            compiled = GurobiCompiledFormulation(
                native_model=model,
                formulation=data,
                variables=variables,
                construction_time_s=0.0,
            )
            result = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                compiled
            )
            actual = result.flow_solutions[0].components
            assert [v.transmission_latency_s for v in actual] == pytest.approx(
                [
                    data.connections_by_pair[path[0], path[1]].round_trip_time_s / 2,
                    data.connections_by_pair[path[1], path[2]].round_trip_time_s / 2,
                    0.0,
                ]
            )
