"""Execution and nominal transmission must be exact for every objective."""

from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pytest
from test.optimizer.test_full_optimizer_runtime import compiler as full_compiler
from test.optimizer.test_single_configuration_contract import (
    single_configuration_problem,
)

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_builder import (
    CompactAssignmentBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_builder import (
    CompactExecutionBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_builder import (
    CompactNetworkBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from optimizer.domain.problem.other.network_connection import NetworkConnection
from shared.service.service import ServiceId


def _network_formulation(objective):
    problem = single_configuration_problem(2)
    flow = next(iter(problem.flows))
    source = next(iter(problem.workers))
    fast, slow, ghost, isolated, outside = (
        ServiceId(UUID(int=index)) for index in range(4, 9)
    )
    workers = (source, fast, slow, ghost, isolated)
    problem = replace(
        problem,
        workers=dict.fromkeys((*workers, outside), problem.workers[source]),
        configurations={
            key: replace(
                data,
                tensor_bytes={
                    "x": data.batch_size * 1_000_000,
                    "y": data.batch_size * 250_000,
                },
                max_cut_size_bytes=data.batch_size * 1_000_000,
            )
            for key, data in problem.configurations.items()
        },
        execution_times_s={
            (key, worker): {**times, "compute": times["compute"] * (index + 1)}
            for (key, _), times in problem.execution_times_s.items()
            for index, worker in enumerate(workers)
        },
        flows={
            flow: replace(
                problem.flows[flow],
                workers=workers,
                arrival_rate_per_s=0.01,
                max_response_time_s=1000,
            )
        },
        policy=replace(problem.policy, max_components=3, objective=objective),
        connections=(
            NetworkConnection(source, fast, 10_000_000, 0.02),
            NetworkConnection(source, slow, 1_000_000, 0.02),
            NetworkConnection(ghost, fast, 2_000_000, 0.02),
            NetworkConnection(ghost, slow, 20_000_000, 0.02),
            NetworkConnection(fast, ghost, 5_000_000, 0.02),
            NetworkConnection(slow, ghost, 3_000_000, 0.02),
            # A link outside the flow must not determine its slowest bandwidth.
            NetworkConnection(source, outside, 1, 0.02),
        ),
    )
    return MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5)
    )


def _fix_plan(compiled, *, selected_index, target_index, compute_component):
    data = compiled.formulation
    flow = data.flows[0]
    workers = data.problem.flows[flow].workers
    source, target = workers[0], workers[target_index]
    selected = data.configurations_by_flow[flow][selected_index]
    assignment = compiled.variables.assignment
    components = range(1, data.problem.policy.max_components + 1)
    for key in data.configurations_by_flow[flow]:
        variable = assignment.configuration_selected[flow, key]
        variable.LB = variable.UB = int(key == selected)
        for layer in data.problem.variants[key.model_variant_id].layers:
            layer_component = (
                compute_component
                if layer == "compute"
                else {"input": 1, "output": 2}[layer]
            )
            for component in components:
                variable = assignment.layer_component[flow, key, layer, component]
                variable.LB = variable.UB = int(
                    key == selected and component == layer_component
                )
    for component in components:
        for worker in workers:
            variable = assignment.component_worker[flow, component, worker]
            variable.LB = variable.UB = int(
                (component == 1 and worker == source)
                or (component == 2 and worker == target)
            )
    return flow, selected, source, target


async def _solve(compiled):
    return await GurobiSolver().solve_compiled_formulation(
        compiled, GurobiSolverConfig(time_limit_s=10, threads=1, seed=0, mip_gap=0)
    )


class _NominalTimeModelBuilder:
    """Isolate linear cost constraints from queue SOCs and their license limit."""

    def build_model_from_formulation(self, model, data):
        assignment_builder = CompactAssignmentBuilder()
        execution_builder = CompactExecutionBuilder()
        network_builder = CompactNetworkBuilder()
        assignment = assignment_builder.create_variables(model, data)
        execution = execution_builder.create_variables(model, data)
        network = network_builder.create_variables(model, data)
        assignment_builder.add_constraints(model, data, assignment=assignment)
        execution_builder.add_constraints(
            model, data, assignment=assignment, execution=execution
        )
        network_builder.add_constraints(
            model, data, assignment=assignment, network=network
        )
        return SimpleNamespace(
            assignment=assignment, execution=execution, network=network
        )


def _compiler():
    return GurobiFormulationCompiler(_NominalTimeModelBuilder(), non_convex=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_index", [1, 2])
@pytest.mark.parametrize("selected_index,compute_component", [(0, 1), (1, 2)])
async def test_native_execution_and_transmission_are_exact(
    target_index, selected_index, compute_component
):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        _flow, selected, source, target = _fix_plan(
            compiled,
            selected_index=selected_index,
            target_index=target_index,
            compute_component=compute_component,
        )
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.OPTIMAL
        assignment = compiled.variables.assignment
        for (
            f,
            key,
            component,
            worker,
        ), variable in compiled.variables.execution.execution_time_s.items():
            times = data.problem.execution_times_s[key, worker]
            expected = (
                sum(
                    time
                    for layer, time in times.items()
                    if assignment.layer_component[f, key, layer, component].X > 0.5
                )
                if assignment.component_worker[f, component, worker].X > 0.5
                else 0
            )
            assert pytest.approx(expected, abs=1e-7) == variable.X
        tensor = "x" if compute_component == 2 else "y"
        expected_transmission = (
            data.problem.configurations[selected].tensor_bytes[tensor]
            / data.connections_by_pair[source, target].throughput_bytes_per_sec
        )
        for (
            _,
            key,
            cut,
            worker,
        ), variable in compiled.variables.network.transmission_time_s.items():
            expected = (
                expected_transmission
                if (key == selected and cut == 1 and worker == source)
                else 0
            )
            assert pytest.approx(expected, abs=1e-7) == variable.X


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "witness",
    [
        "excluded_execution",
        "unplaced_execution",
        "empty_execution",
        "execution_below_true",
        "transmission_above_true",
        "transmission_below_true",
        "excluded_transmission",
        "unplaced_transmission",
        "empty_cut_transmission",
        "disconnected_transmission",
    ],
)
async def test_incorrect_nominal_times_are_infeasible(witness):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, selected, source, target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        excluded = data.configurations_by_flow[flow][1]
        ghost, isolated = data.problem.flows[flow].workers[3:]
        execution = compiled.variables.execution.execution_time_s
        transmission = compiled.variables.network.transmission_time_s
        variable, incorrect = {
            "excluded_execution": (execution[flow, excluded, 2, target], 0.05),
            "unplaced_execution": (execution[flow, selected, 2, source], 0.05),
            "empty_execution": (execution[flow, selected, 1, source], 0.05),
            "execution_below_true": (execution[flow, selected, 2, target], 0.15),
            "transmission_above_true": (transmission[flow, selected, 1, source], 0.15),
            "transmission_below_true": (transmission[flow, selected, 1, source], 0.05),
            "excluded_transmission": (transmission[flow, excluded, 1, source], 0.05),
            "unplaced_transmission": (transmission[flow, selected, 1, ghost], 0.05),
            "empty_cut_transmission": (transmission[flow, selected, 2, target], 0.05),
            "disconnected_transmission": (
                transmission[flow, selected, 1, isolated],
                0.05,
            ),
        }[witness]
        compiled.native_model.addConstr(variable == incorrect)
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.INFEASIBLE


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_full_network_has_exact_nominal_times_for_every_objective(objective):
    problem = _network_formulation(objective).problem
    flow = next(iter(problem.flows))
    workers = problem.flows[flow].workers[:2]
    problem = replace(
        problem,
        workers={worker: problem.workers[worker] for worker in workers},
        flows={flow: replace(problem.flows[flow], workers=workers)},
        execution_times_s={
            (key, worker): times
            for (key, worker), times in problem.execution_times_s.items()
            if worker in workers
        },
        connections=tuple(
            link
            for link in problem.connections
            if link.source in workers and link.target in workers
        ),
        policy=replace(problem.policy, max_components=2),
    )
    data = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5)
    )
    async with full_compiler().compile_formulation(data) as compiled:
        flow, selected, source, target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.OPTIMAL
        excluded = data.configurations_by_flow[flow][1]
        times = compiled.variables.execution.execution_time_s
        assert pytest.approx(0.2) == times[flow, selected, 2, target].X
        assert pytest.approx(0) == times[flow, selected, 1, source].X
        assert pytest.approx(0) == times[flow, excluded, 2, target].X
        times = compiled.variables.network.transmission_time_s
        assert pytest.approx(0.1) == times[flow, selected, 1, source].X
        assert pytest.approx(0) == times[flow, selected, 1, target].X
        assert pytest.approx(0) == times[flow, excluded, 1, source].X
