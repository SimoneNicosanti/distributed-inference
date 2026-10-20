from collections.abc import Mapping
from itertools import product
from types import SimpleNamespace
from typing import Literal, cast
from uuid import UUID

import gurobipy as gp
import pytest
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    _add_transition_marginal_constraints,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    NetworkVariables,
    _add_network_variables,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.profile.network_topology import (
    NetworkConnection,
    NetworkTopology,
)
from shared.flow.flow import FlowId
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId

type Arc = tuple[WorkerId, WorkerId]
type Formulation = Literal["legacy", "marginal"]

_USER = UserId(id=UUID(int=1))
_FLOW_A = FlowId(user_id=_USER, flow_id=UUID(int=2))
_FLOW_B = FlowId(user_id=_USER, flow_id=UUID(int=3))
_WORKER_A = ServiceId(service_id=UUID(int=4))
_WORKER_B = ServiceId(service_id=UUID(int=5))
_WORKER_C = ServiceId(service_id=UUID(int=6))


def _problem(
    workers_by_flow: Mapping[FlowId, tuple[WorkerId, ...]],
) -> GurobiProblem:
    flows = tuple(workers_by_flow)
    workers = tuple(
        dict.fromkeys(
            worker
            for flow_workers in workers_by_flow.values()
            for worker in flow_workers
        )
    )
    return GurobiProblem(
        flows=flows,
        workers=workers,
        workers_by_flow=dict(workers_by_flow),
        models_by_flow=dict.fromkeys(flows, ()),
        service_arrival_rate={},
        batch_fill_time={},
        compute_bin_limits=(0.0, 0.9),
        network_bin_limits=(0.0, 0.9),
        max_queue_wait_by_worker={},
        max_queue_wait_by_flow={},
        flow_index={flow: index for index, flow in enumerate(flows)},
        worker_index={worker: index for index, worker in enumerate(workers)},
        model_index={},
        layer_index={},
        tensor_index={},
    )


def _topology(
    workers: tuple[WorkerId, ...], arcs: frozenset[Arc]
) -> NetworkTopology:
    return NetworkTopology(
        workers=workers,
        connections=(
            NetworkConnection(
                source=source,
                target=target,
                throughput_bytes_per_sec=1_000,
                round_trip_time_s=0.01,
            )
            for source, target in arcs
            if source != target
        ),
    )


def _transition_variables(
    model: gp.Model,
    problem: GurobiProblem,
    arcs_by_flow: dict[FlowId, frozenset[Arc]],
    *,
    max_components: int,
    variable_type: str = GRB.BINARY,
) -> tuple[AssignmentVariables, NetworkVariables]:
    component_worker = {
        (flow, component, worker): model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=variable_type,
            name=(
                f"test_e[{problem.flow_index[flow]},{component},"
                f"{problem.worker_index[worker]}]"
            ),
        )
        for flow in problem.flows
        for component in range(1, max_components + 1)
        for worker in problem.workers_by_flow[flow]
    }
    adjacent_component_workers = {
        (flow, cut, source, target): model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=variable_type,
            name=(
                f"test_q[{problem.flow_index[flow]},{cut},"
                f"{problem.worker_index[source]},{problem.worker_index[target]}]"
            ),
        )
        for flow in problem.flows
        for cut in range(1, max_components)
        for source in problem.workers_by_flow[flow]
        for target in problem.workers_by_flow[flow]
        if (source, target) in arcs_by_flow[flow]
    }
    return (
        AssignmentVariables(
            max_components=max_components,
            model_selected={},
            layer_component={},
            component_worker=component_worker,
            component_active={},
            component_last={},
        ),
        NetworkVariables(
            tensor_crossing_auxiliary={},
            adjacent_component_workers=adjacent_component_workers,
            transmission_time={},
            network_utilization={},
            network_bin_selected={},
            component_transmission_response={},
        ),
    )


def _fix_worker(
    assignment: AssignmentVariables,
    flow: FlowId,
    component: int,
    workers: tuple[WorkerId, ...],
    selected: WorkerId | None,
) -> None:
    for worker in workers:
        value = float(worker == selected)
        variable = assignment.component_worker[flow, component, worker]
        variable.LB = value
        variable.UB = value


def _add_legacy_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    flow: FlowId,
    cut: int,
) -> None:
    workers = problem.workers_by_flow[flow]
    for target in workers:
        allowed_sources = tuple(
            source
            for source in workers
            if (flow, cut, source, target)
            in network.adjacent_component_workers
        )
        if len(allowed_sources) < len(workers):
            model.addConstr(
                assignment.component_worker[flow, cut + 1, target]
                <= gp.quicksum(
                    assignment.component_worker[flow, cut, source]
                    for source in allowed_sources
                )
            )

    for source in workers:
        for target in workers:
            q = network.adjacent_component_workers.get(
                (flow, cut, source, target)
            )
            if q is None:
                continue
            current = assignment.component_worker[flow, cut, source]
            following = assignment.component_worker[flow, cut + 1, target]
            model.addConstr(q <= current)
            model.addConstr(q <= following)
            model.addConstr(q >= current + following - 1)


def _solve_fixed_transition(
    formulation: Formulation,
    *,
    workers: tuple[WorkerId, ...],
    arcs: frozenset[Arc],
    current: WorkerId | None,
    following: WorkerId | None,
) -> tuple[int, frozenset[Arc], int]:
    problem = _problem({_FLOW_A: workers})
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        model.Params.DualReductions = 0
        assignment, network = _transition_variables(
            model,
            problem,
            {_FLOW_A: arcs},
            max_components=2,
        )
        _fix_worker(assignment, _FLOW_A, 1, workers, current)
        _fix_worker(assignment, _FLOW_A, 2, workers, following)
        if formulation == "legacy":
            _add_legacy_constraints(
                model,
                problem,
                assignment,
                network,
                flow=_FLOW_A,
                cut=1,
            )
        else:
            _add_transition_marginal_constraints(
                model,
                problem,
                assignment,
                network,
                topology=_topology(workers, arcs),
                flow=_FLOW_A,
                cut=1,
            )
        model.optimize()
        selected = (
            frozenset(
                (source, target)
                for source, target in arcs
                if network.adjacent_component_workers[
                    _FLOW_A, 1, source, target
                ].X
                > 0.5
            )
            if model.Status == GRB.OPTIMAL
            else frozenset()
        )
        return model.Status, selected, model.NumConstrs


def test_production_q_indices_are_sparse_binary_and_keep_self_transitions() -> None:
    workers = (_WORKER_A, _WORKER_B, _WORKER_C)
    problem = _problem({_FLOW_A: workers})
    topology = NetworkTopology(
        workers=workers,
        connections=(
            NetworkConnection(
                source=_WORKER_A,
                target=_WORKER_B,
                throughput_bytes_per_sec=1_000,
                round_trip_time_s=0.01,
            ),
        ),
    )
    context = cast(
        OptimizationContext,
        SimpleNamespace(network_topology=topology, static_profiles_by_variant={}),
    )
    requests = cast(
        OptimizationRequestsByFlow,
        {
            _FLOW_A: SimpleNamespace(
                qos_requirements=SimpleNamespace(max_mean_response_time=1.0)
            )
        },
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        network = _add_network_variables(
            model,
            problem,
            context,
            requests,
            max_components=3,
            network_utilization_limit=0.9,
        )
        model.update()

        expected_arcs = {
            (_WORKER_A, _WORKER_A),
            (_WORKER_B, _WORKER_B),
            (_WORKER_C, _WORKER_C),
            (_WORKER_A, _WORKER_B),
        }
        actual_arcs = {
            (source, target)
            for flow, _cut, source, target in (
                network.adjacent_component_workers
            )
            if flow == _FLOW_A
        }
        assert actual_arcs == expected_arcs
        assert len(network.adjacent_component_workers) == 2 * len(expected_arcs)
        assert all(
            variable.VType == GRB.BINARY
            for variable in network.adjacent_component_workers.values()
        )


def test_marginals_are_integer_equivalent_to_sparse_legacy_formulation() -> None:
    workers = (_WORKER_A, _WORKER_B)
    arcs = frozenset(
        {
            (_WORKER_A, _WORKER_A),
            (_WORKER_B, _WORKER_B),
            (_WORKER_A, _WORKER_B),
        }
    )
    placements = (None, *workers)

    for current, following in product(placements, repeat=2):
        if current is None and following is not None:
            continue
        legacy = _solve_fixed_transition(
            "legacy",
            workers=workers,
            arcs=arcs,
            current=current,
            following=following,
        )
        marginal = _solve_fixed_transition(
            "marginal",
            workers=workers,
            arcs=arcs,
            current=current,
            following=following,
        )
        assert marginal[:2] == legacy[:2]

        if marginal[0] == GRB.OPTIMAL:
            expected = (
                frozenset({(current, following)})
                if current is not None and following is not None
                else frozenset()
            )
            assert marginal[1] == expected


def test_missing_remote_predecessor_forbids_transition() -> None:
    workers = (_WORKER_A, _WORKER_B)
    arcs = frozenset(
        {
            (_WORKER_A, _WORKER_A),
            (_WORKER_B, _WORKER_B),
        }
    )

    infeasible, _, _ = _solve_fixed_transition(
        "marginal",
        workers=workers,
        arcs=arcs,
        current=_WORKER_A,
        following=_WORKER_B,
    )
    feasible, selected, constraint_count = _solve_fixed_transition(
        "marginal",
        workers=workers,
        arcs=arcs,
        current=_WORKER_A,
        following=None,
    )

    assert infeasible == GRB.INFEASIBLE
    assert feasible == GRB.OPTIMAL
    assert selected == frozenset()
    assert constraint_count == 4


def test_marginals_cover_multiple_cuts_and_flow_specific_arc_sets() -> None:
    workers_by_flow = {
        _FLOW_A: (_WORKER_A, _WORKER_B),
        _FLOW_B: (_WORKER_A, _WORKER_C),
    }
    arcs_by_flow = {
        _FLOW_A: frozenset(
            {
                (_WORKER_A, _WORKER_A),
                (_WORKER_B, _WORKER_B),
                (_WORKER_A, _WORKER_B),
            }
        ),
        _FLOW_B: frozenset(
            {
                (_WORKER_A, _WORKER_A),
                (_WORKER_C, _WORKER_C),
            }
        ),
    }
    problem = _problem(workers_by_flow)
    topology = _topology(
        problem.workers,
        frozenset(arc for flow_arcs in arcs_by_flow.values() for arc in flow_arcs),
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment, network = _transition_variables(
            model,
            problem,
            arcs_by_flow,
            max_components=4,
        )
        for component, selected in enumerate(
            (_WORKER_A, _WORKER_B, _WORKER_B, None), start=1
        ):
            _fix_worker(
                assignment,
                _FLOW_A,
                component,
                workers_by_flow[_FLOW_A],
                selected,
            )
        for component, selected in enumerate(
            (_WORKER_A, _WORKER_A, None, None), start=1
        ):
            _fix_worker(
                assignment,
                _FLOW_B,
                component,
                workers_by_flow[_FLOW_B],
                selected,
            )
        for flow in problem.flows:
            for cut in range(1, 4):
                _add_transition_marginal_constraints(
                    model,
                    problem,
                    assignment,
                    network,
                    topology=topology,
                    flow=flow,
                    cut=cut,
                )

        model.optimize()

        assert model.Status == GRB.OPTIMAL
        selected_transitions = {
            (flow, cut, source, target)
            for (flow, cut, source, target), variable in (
                network.adjacent_component_workers.items()
            )
            if variable.X > 0.5
        }
        assert selected_transitions == {
            (_FLOW_A, 1, _WORKER_A, _WORKER_B),
            (_FLOW_A, 2, _WORKER_B, _WORKER_B),
            (_FLOW_B, 1, _WORKER_A, _WORKER_A),
        }


def test_dense_marginals_reduce_constraint_count_without_changing_q_count() -> None:
    workers = (_WORKER_A, _WORKER_B, _WORKER_C)
    arcs = frozenset(
        (source, target) for source in workers for target in workers
    )
    cuts = 2
    problem = _problem({_FLOW_A: workers})
    topology = _topology(workers, arcs)

    counts: dict[Formulation, tuple[int, int]] = {}
    for formulation in ("legacy", "marginal"):
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            assignment, network = _transition_variables(
                model,
                problem,
                {_FLOW_A: arcs},
                max_components=cuts + 1,
            )
            for cut in range(1, cuts + 1):
                if formulation == "legacy":
                    _add_legacy_constraints(
                        model,
                        problem,
                        assignment,
                        network,
                        flow=_FLOW_A,
                        cut=cut,
                    )
                else:
                    _add_transition_marginal_constraints(
                        model,
                        problem,
                        assignment,
                        network,
                        topology=topology,
                        flow=_FLOW_A,
                        cut=cut,
                    )
            model.update()
            counts[formulation] = (
                len(network.adjacent_component_workers),
                model.NumConstrs,
            )

    assert counts["legacy"] == (cuts * len(arcs), 3 * cuts * len(arcs))
    assert counts["marginal"] == (cuts * len(arcs), 2 * cuts * len(workers))


def test_marginal_continuous_relaxation_dominates_mccormick() -> None:
    workers = (_WORKER_A, _WORKER_B)
    arcs = frozenset(
        (source, target) for source in workers for target in workers
    )
    problem = _problem({_FLOW_A: workers})
    topology = _topology(workers, arcs)
    objectives: dict[Formulation, float] = {}

    for formulation in ("legacy", "marginal"):
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            assignment, network = _transition_variables(
                model,
                problem,
                {_FLOW_A: arcs},
                max_components=2,
                variable_type=GRB.CONTINUOUS,
            )
            for component in (1, 2):
                for worker in workers:
                    variable = assignment.component_worker[
                        _FLOW_A, component, worker
                    ]
                    variable.LB = 0.5
                    variable.UB = 0.5
            if formulation == "legacy":
                _add_legacy_constraints(
                    model,
                    problem,
                    assignment,
                    network,
                    flow=_FLOW_A,
                    cut=1,
                )
            else:
                _add_transition_marginal_constraints(
                    model,
                    problem,
                    assignment,
                    network,
                    topology=topology,
                    flow=_FLOW_A,
                    cut=1,
                )
            model.setObjective(
                gp.quicksum(network.adjacent_component_workers.values()),
                GRB.MINIMIZE,
            )
            model.optimize()
            assert model.Status == GRB.OPTIMAL
            objectives[formulation] = model.ObjVal

    assert objectives["legacy"] == pytest.approx(0.0)
    assert objectives["marginal"] == pytest.approx(1.0)
