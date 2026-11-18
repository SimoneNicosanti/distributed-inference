"""Integer equivalence, native optima, representation and MIP-start contracts."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast
from uuid import UUID

import gurobipy as gp
import pytest
from gurobipy import GRB
from test.optimizer.test_full_optimizer_runtime import assert_solution, compiler
from test.optimizer.test_single_configuration_contract import (
    single_configuration_problem,
)

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.abc.gurobi_model_builder import (
    GurobiModelBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_builder import (
    CompactAssignmentBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
    CumulativeAssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.cumulative_assignment_builder import (
    CumulativeAssignmentBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.expressions import (
    destination_crossing_expression,
    tensor_crossing_term,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_builder import (
    CompactNetworkBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_warm_start_applier import (
    CompactGurobiWarmStartApplier,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    ObjectiveKind,
    VariantData,
)
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from shared.flow.flow import FlowId
from shared.service.service import ServiceId

LAYERS = ("input", "a", "b", "c", "e", "join", "output", "output2")
EDGES = (
    EdgeDescription("input", "a", frozenset({"input-tensor"})),
    EdgeDescription("input", "b", frozenset({"input-tensor"})),
    EdgeDescription("a", "c", frozenset({"skip"})),
    EdgeDescription("a", "e", frozenset({"skip"})),
    EdgeDescription("b", "c", frozenset({"b-tensor"})),
    EdgeDescription("c", "join", frozenset({"c-tensor"})),
    EdgeDescription("e", "join", frozenset({"e-tensor"})),
    EdgeDescription("a", "join", frozenset({"residual"})),
    EdgeDescription("join", "output", frozenset({"result"})),
    EdgeDescription("join", "output2", frozenset({"result"})),
)


def dag_problem(
    *,
    variants: int = 2,
    workers: int = 2,
    components: int = 3,
    flows: int = 1,
    priorities: int = 1,
) -> CompactOptimizationProblem:
    base = cast(
        Callable[[int], CompactOptimizationProblem], single_configuration_problem
    )(variants)
    source = next(iter(base.workers))
    target = ServiceId(UUID(int=4))
    worker_ids = (source, target)[:workers]
    graph = VariantData(
        next(iter(base.variants.values())).accuracies,
        LAYERS,
        ("input",),
        ("output", "output2"),
        EDGES,
        {
            layer: 0 if layer in {"input", "output", "output2"} else 64
            for layer in LAYERS
        },
    )
    tensors = frozenset(tensor for edge in EDGES for tensor in edge.tensors)
    configurations = {
        key: ConfigurationData(
            data.batch_size,
            dict.fromkeys(tensors, 32 * data.batch_size),
            32 * data.batch_size * len(tensors),
        )
        for key, data in base.configurations.items()
    }
    first_flow, request = next(iter(base.flows.items()))
    return replace(
        base,
        variants=dict.fromkeys(base.variants, graph),
        configurations=configurations,
        workers={
            worker: replace(
                base.workers[source],
                backend=Backend.CPU if worker == source else Backend.GPU,
            )
            for worker in worker_ids
        },
        execution_times_s={
            (key, worker): {
                layer: 0.0
                if layer in {"input", "output", "output2"}
                else 0.001 * value.batch_size * (1 if worker == target else 2)
                for layer in LAYERS
            }
            for key, value in configurations.items()
            for worker in worker_ids
        },
        flows={
            FlowId(first_flow.user_id, UUID(int=20 + index)): replace(
                request,
                arrival_rate_per_s=0.1,
                workers=worker_ids,
                endpoint_placement=FlowEndpointPlacement(
                    FixedPlacement(source), FixedPlacement(source)
                ),
            )
            for index in range(flows)
        },
        connections=tuple(
            NetworkConnection(left, right, 1_000_000, 0.002)
            for left in worker_ids
            for right in worker_ids
            if left != right
        ),
        policy=replace(
            base.policy, max_components=components, priority_count=priorities
        ),
    )


def formulation(
    problem: CompactOptimizationProblem, kind: Kind
) -> MisocpCompactFormulation:
    return MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5, kind)
    )


def value(term):
    return term.X if isinstance(term, gp.Var) else term.getValue()


@dataclass
class PartitionVariables:
    assignment: AssignmentVariables
    network: NetworkVariables


class PartitionBuilder(
    GurobiModelBuilder[MisocpCompactFormulation, PartitionVariables]
):
    def build_model_from_formulation(self, model, data):
        builder = (
            CumulativeAssignmentBuilder()
            if data.representation is Kind.CUMULATIVE
            else CompactAssignmentBuilder()
        )
        network_builder = CompactNetworkBuilder()
        assignment = builder.create_variables(model, data)
        network = network_builder.create_variables(model, data)
        builder.add_constraints(model, data, assignment=assignment)
        network_builder.add_constraints(
            model, data, assignment=assignment, network=network
        )
        return PartitionVariables(assignment, network)


def partition_compiler():
    return GurobiFormulationCompiler(PartitionBuilder(), non_convex=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("components", [1, 2, 3, 7])
async def test_structure_and_direct_prefix_coefficients(components):
    data = formulation(dag_problem(components=components), Kind.CUMULATIVE)
    async with partition_compiler().compile_formulation(data) as compiled:
        model = compiled.native_model
        variables = model.getVars()
        assert not any(var.VarName.startswith("x[") for var in variables)
        ys = [var for var in variables if var.VarName.startswith("y[")]
        assert len(ys) == 2 * len(LAYERS) * (components - 1)
        assert all(var.VType == GRB.BINARY for var in ys)
        assignment = compiled.variables.assignment
        assert all(
            var.VType == GRB.BINARY
            for var in assignment.configuration_selected.values()
        )
        assert all(
            isinstance(term, gp.LinExpr) for term in assignment.layer_component.values()
        )
        for constraint in model.getConstrs():
            if constraint.ConstrName.startswith("precedence["):
                row = model.getRow(constraint)
                assert row.size() == 2
                assert all(
                    row.getVar(index).VarName.startswith("y[")
                    for index in range(row.size())
                )
        flow = data.flows[0]
        key = data.configurations_by_flow[flow][0]
        for cut in range(1, components):
            term = destination_crossing_expression(
                assignment,
                flow=flow,
                model_variant=key,
                source="input",
                destination="b",
                cut=cut,
            )
            assert term.size() == 2
            assert all(
                term.getVar(index).VarName.startswith("y[")
                for index in range(term.size())
            )


@pytest.mark.asyncio
async def test_exhaustive_integer_partitions_and_tensor_cuts_match():
    problem = dag_problem()
    problem = replace(
        problem,
        flows={
            flow: replace(
                request,
                endpoint_placement=FlowEndpointPlacement(
                    AnyPlacement(), AnyPlacement()
                ),
            )
            for flow, request in problem.flows.items()
        },
    )
    feasible_sets = []
    for kind in Kind:
        data = formulation(problem, kind)
        async with partition_compiler().compile_formulation(data) as compiled:
            native = compiled.native_model
            assignment = compiled.variables.assignment
            flow = data.flows[0]
            selected, excluded = data.configurations_by_flow[flow]
            assignment.configuration_selected[flow, selected].LB = 1
            assignment.configuration_selected[flow, excluded].UB = 0
            feasible = set()
            for last in (1, 2, 3):
                for internal in itertools.product(range(1, last + 1), repeat=5):
                    positions = dict(
                        zip(LAYERS, (1, *internal, last, last), strict=True)
                    )
                    expected = set(positions.values()) == set(
                        range(1, last + 1)
                    ) and all(
                        positions[edge.source] <= positions[edge.target]
                        for edge in EDGES
                    )
                    pins = [
                        native.addConstr(
                            assignment.layer_component[flow, selected, layer, component]
                            == int(positions[layer] == component)
                        )
                        for layer in LAYERS
                        for component in range(1, 4)
                    ]
                    native.optimize()
                    assert (native.Status == GRB.OPTIMAL) is expected, (kind, positions)
                    if expected:
                        feasible.add(tuple(positions[layer] for layer in LAYERS))
                        for layer in LAYERS:
                            assert sum(
                                value(
                                    assignment.layer_component[flow, selected, layer, k]
                                )
                                for k in range(1, 4)
                            ) == pytest.approx(1)
                            assert all(
                                value(
                                    assignment.layer_component[flow, excluded, layer, k]
                                )
                                == pytest.approx(0)
                                for k in range(1, 4)
                            )
                        if isinstance(assignment, CumulativeAssignmentVariables):
                            for (
                                _f,
                                key,
                                layer,
                                cut,
                            ), y in assignment.layer_cumulative.items():
                                assert (
                                    pytest.approx(
                                        int(key == selected and positions[layer] <= cut)
                                    )
                                    == y.X
                                )
                        for cut in (1, 2):
                            cut_bytes = 0
                            expected_bytes = 0
                            for tensor, route in data.tensor_routes_by_variant[
                                selected.model_variant_id
                            ].items():
                                crosses = value(
                                    tensor_crossing_term(
                                        assignment,
                                        compiled.variables.network,
                                        flow=flow,
                                        model_variant=selected,
                                        tensor=tensor,
                                        route=route,
                                        cut=cut,
                                    )
                                )
                                expected_crossing = positions[
                                    route.source
                                ] <= cut and any(
                                    positions[destination] > cut
                                    for destination in route.destinations
                                )
                                assert crosses == pytest.approx(int(expected_crossing))
                                size = data.problem.configurations[
                                    selected
                                ].tensor_bytes[tensor]
                                cut_bytes += size * crosses
                                expected_bytes += size * expected_crossing
                            assert cut_bytes == pytest.approx(expected_bytes)
                    native.remove(pins)
                    native.reset()
            feasible_sets.append(feasible)
    assert feasible_sets[0] == feasible_sets[1]
    assert len(feasible_sets[0]) > 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        {"variants": 2, "workers": 1, "components": 1},
        {"variants": 1, "workers": 2, "components": 3, "priorities": 2},
        {"variants": 1, "workers": 1, "components": 1, "flows": 2, "priorities": 2},
        {"variants": 2, "workers": 2, "components": 2},
    ],
)
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_native_full_optima_and_solutions_match(settings, objective):
    problem = dag_problem(**settings)
    problem = replace(problem, policy=replace(problem.policy, objective=objective))
    optima = []
    for kind in Kind:
        data = formulation(problem, kind)
        async with compiler().compile_formulation(data) as compiled:
            output = await GurobiSolver().solve_compiled_formulation(
                compiled,
                GurobiSolverConfig(
                    time_limit_s=10,
                    threads=1,
                    seed=0,
                    mip_gap=0,
                    record_incumbent_history=True,
                ),
            )
            assert output.termination is SolverTerminationKind.OPTIMAL
            assert compiled.native_model.ConstrVio < 1e-5
            assert compiled.native_model.BoundVio < 1e-5
            assert compiled.native_model.IntVio < 1e-5
            solution = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                compiled
            )
            assert_solution(problem, solution)
            optima.append(output.objective_value)
            assert output.statistics.original_variables == compiled.native_model.NumVars
            assert output.statistics.linear_nonzeros == compiled.native_model.NumNZs
            assert output.statistics.bound_history
    assert optima[0] == pytest.approx(optima[1], abs=1e-5, rel=1e-5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deadline,expected",
    [(1.0, SolverTerminationKind.OPTIMAL), (1e-7, SolverTerminationKind.INFEASIBLE)],
)
async def test_hard_deadline_equivalence(deadline, expected):
    problem = dag_problem(variants=1, workers=1, components=1)
    problem = replace(
        problem,
        flows={
            flow: replace(request, max_response_time_s=deadline)
            for flow, request in problem.flows.items()
        },
    )
    for kind in Kind:
        async with compiler().compile_formulation(
            formulation(problem, kind)
        ) as compiled:
            output = await GurobiSolver().solve_compiled_formulation(
                compiled, GurobiSolverConfig(time_limit_s=5, threads=1)
            )
            assert output.termination is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("components", [1, 3])
async def test_mip_starts_work_between_representations(components):
    problem = dag_problem(variants=2, workers=1, components=components)
    incumbent = None
    for kind in (Kind.ASSIGNMENT, Kind.CUMULATIVE, Kind.ASSIGNMENT):
        async with compiler().compile_formulation(
            formulation(problem, kind)
        ) as compiled:
            if incumbent is not None:
                await CompactGurobiWarmStartApplier().apply_warm_start(
                    compiled, incumbent
                )
                assignment = compiled.variables.assignment
                assert all(
                    var.Start in (0.0, 1.0)
                    for var in assignment.layer_decision_variables()
                )
                if isinstance(assignment, CumulativeAssignmentVariables):
                    seed = incumbent.flow_solutions[0]
                    positions = {
                        layer: component.component_index
                        for component in seed.components
                        for layer in component.layer_keys
                    }
                    for (
                        _,
                        key,
                        layer,
                        cut,
                    ), var in assignment.layer_cumulative.items():
                        assert var.Start == int(
                            key.model_variant_id == seed.model_variant_id
                            and positions[layer] <= cut
                        )
                assert all(
                    var.Start == GRB.UNDEFINED
                    for var in compiled.variables.execution.execution_time_s.values()
                )
            output = await GurobiSolver().solve_compiled_formulation(
                compiled, GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0)
            )
            assert output.termination is SolverTerminationKind.OPTIMAL
            incumbent = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                compiled
            )
            assert_solution(problem, incumbent)
