"""Component priorities retain independent choices with exact source latency."""

import hashlib

import pytest
from gurobipy import GRB
from test.optimizer.test_cumulative_formulation import dag_problem
from test.optimizer.test_full_optimizer_runtime import compiler

from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)


def build(problem, kind):
    return MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.3, kind)
    )


def matrix(model):
    def linear(expression):
        return tuple(
            sorted(
                (expression.getVar(i).VarName, expression.getCoeff(i))
                for i in range(expression.size())
            )
        )

    variables = tuple((v.VarName, v.VType, v.LB, v.UB, v.Obj) for v in model.getVars())
    rows = tuple(
        (c.ConstrName, c.Sense, c.RHS, linear(model.getRow(c)))
        for c in model.getConstrs()
    )
    quadratics = []
    for c in model.getQConstrs():
        expression = model.getQCRow(c)
        terms = tuple(
            sorted(
                (
                    tuple(
                        sorted(
                            (
                                expression.getVar1(i).VarName,
                                expression.getVar2(i).VarName,
                            )
                        )
                    ),
                    expression.getCoeff(i),
                )
                for i in range(expression.size())
            )
        )
        quadratics.append(
            (c.QCName, c.QCSense, c.QCRHS, linear(expression.getLinExpr()), terms)
        )
    assert model.NumGenConstrs == 0 and model.NumQNZs == 0
    return variables, rows, tuple(quadratics), model.ObjCon, model.ModelSense


def pin_reference(compiled, *, mixed=False):
    data, a, queue = (
        compiled.formulation,
        compiled.variables.assignment,
        compiled.variables.compute_queue,
    )
    source, target = data.workers
    for (_flow, config, layer, component), term in a.layer_component.items():
        variant = data.problem.variants[config.model_variant_id]
        index = (
            1
            if layer in variant.input_layers
            else 3
            if layer in variant.output_layers
            else 2
        )
        compiled.native_model.addConstr(term == int(component == index))
    for (_, component, worker), term in a.component_worker.items():
        compiled.native_model.addConstr(
            term
            == int(component <= 3 and worker == (target if component == 2 else source))
        )
    for (
        flow,
        component,
        worker,
        priority,
    ), variable in queue.component_priority.items():
        selected = 2 if mixed and flow == data.flows[0] and component == 2 else 1
        variable.LB = variable.UB = int(
            component <= 3
            and priority == selected
            and worker == (target if component == 2 else source)
        )
    compiled.native_model.update()


def optimize(compiled):
    m = compiled.native_model
    m.Params.Threads = 1
    m.Params.Seed = 0
    m.Params.MIPGap = 0
    m.Params.TimeLimit = 15
    m.optimize()
    return m.Status


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_component_model_matches_source_latency_snapshot(kind):
    # Snapshot updated for the requested q-to-xi latency reformulation.
    expected = {
        "assignment": "3d6b08f4b83570ef01d2a9fbe17f89b1eda75437042a801875e83dbf7a5a5d8c",
        "cumulative": "812672ba5f865d28b6099f667eab26083700d6c3b84026b7b7706f62562a3536",
    }
    problem = dag_problem(variants=1, flows=2, components=4, priorities=2)
    async with compiler().compile_formulation(build(problem, kind)) as c:
        assert (
            hashlib.sha256(repr(matrix(c.native_model)).encode()).hexdigest()
            == expected[kind.value]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_components_of_one_flow_can_use_different_priorities(kind):
    problem = dag_problem(variants=1, flows=2, components=4, priorities=2)
    async with compiler().compile_formulation(build(problem, kind)) as c:
        pin_reference(c, mixed=True)
        assert optimize(c) == GRB.OPTIMAL
        solution = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
            c
        )
        assert {p.priority for p in solution.flow_solutions[0].components} == {1, 2}


@pytest.mark.parametrize("option", ["priority_assignment", "priority_formulation"])
def test_removed_priority_selectors_are_rejected(option):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        MisocpCompactFormulationConfig(0.3, **{option: "flow"})
