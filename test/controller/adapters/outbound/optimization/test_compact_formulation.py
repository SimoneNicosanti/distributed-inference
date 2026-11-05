"""Focused checks for the sole supported compact formulation."""

from dataclasses import replace

import gurobipy as gp
import pytest
from test.controller.adapters.outbound.optimization.test_gurobi_solver import (
    _context,
    _flow,
    _problem_config,
    _request,
    _worker,
)

from controller.adapters.outbound.optimization.gurobi_solver import GurobiSolver
from controller.domain.optimization.solving.solver_config import (
    FormulationKind,
    ObjectiveKind,
    SolverConfig,
)


def solve(formulation=FormulationKind.COMPACT, *, components=1, priorities=1):
    observations = []
    worker = _worker(1)
    request = _request(worker, arrival_rate=1, max_response_time_s=10, flow_id=_flow(1))
    solution = GurobiSolver(
        SolverConfig(
            ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            formulation=formulation,
            observer=observations.append,
            threads=1,
            seed=0,
        )
    )._solve(
        {request.flow_id: request},
        _context({worker: 0.1}),
        _problem_config(components, priority_count=priorities),
    )
    return solution, observations[-1]


def test_formulation_configuration():
    config = SolverConfig(ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
    assert config.formulation is FormulationKind.COMPACT
    assert list(FormulationKind) == [FormulationKind.COMPACT]
    assert replace(config, formulation="compact").formulation is FormulationKind.COMPACT
    for unsupported in ("legacy", "unknown"):
        with pytest.raises(ValueError):
            replace(config, formulation=unsupported)


@pytest.mark.parametrize("components", [1, 2])
def test_compact_matches_analytical_single_worker_response(components):
    compact, statistics = solve(components=components)
    assert compact.objective_value == pytest.approx(0.10555555555555556)
    assert compact.flow_solutions[0].mean_response_time_s == pytest.approx(
        0.10555555555555556
    )
    assert statistics.formulation == "compact"


def test_implicit_integrality_from_binary_layer_and_priority_assignments(monkeypatch):
    original = gp.Model.optimize

    def optimize(model):
        model.update()
        assert model.NumGenConstrs == 0
        for variable in model.getVars():
            if variable.VarName.startswith(
                ("u[", "e[", "active[", "last[", "n[", "c[", "q[")
            ):
                assert variable.VType == gp.GRB.CONTINUOUS
        original(model)
        for variable in model.getVars():
            if variable.VarName.startswith(
                ("u[", "e[", "active[", "last[", "n[", "c[", "q[")
            ):
                assert pytest.approx(round(variable.X), abs=1e-7) == variable.X

    monkeypatch.setattr(gp.Model, "optimize", optimize)
    solve(FormulationKind.COMPACT, components=2)


@pytest.mark.parametrize("selected", [False, True])
def test_execution_epigraph_cannot_underestimate_physical_time(selected):
    from controller.adapters.outbound.optimization.compact_execution import (
        add_compact_execution_constraints,
    )
    from controller.adapters.outbound.optimization.gurobi_problem_builder import (
        DefaultGurobiProblemBuilder,
    )
    from controller.adapters.outbound.optimization.gurobi_variables import (
        _add_assignment_variables,
        _add_execution_variables,
    )

    workers = (_worker(1), _worker(2))
    request = _request(
        workers[0], arrival_rate=1, max_response_time_s=10, flow_id=_flow(1)
    )
    context = _context(dict.fromkeys(workers, 0.1))
    requests = {request.flow_id: request}
    problem = DefaultGurobiProblemBuilder().build(requests, context, _problem_config(1))
    variant = problem.models_by_flow[request.flow_id][0]
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment = _add_assignment_variables(
            model, problem, context, max_components=1
        )
        execution = _add_execution_variables(model, problem, max_components=1)
        for x in assignment.layer_component.values():
            x.LB = x.UB = 1
        for worker, chosen in zip(workers, (selected, not selected), strict=True):
            e = assignment.component_worker[request.flow_id, 1, worker]
            e.LB = e.UB = int(chosen)
        add_compact_execution_constraints(
            model,
            problem,
            assignment,
            execution,
            context=context,
            requests_by_flow=requests,
            max_components=1,
        )
        time = execution.execution_time_s[request.flow_id, variant, 1, workers[0]]
        model.setObjective(time)
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        assert pytest.approx(0.1 if selected else 0) == time.X
        assert model.NumGenConstrs == 0


@pytest.mark.parametrize("missing_link", [False, True])
def test_transmission_link_latency_and_sparse_topology(missing_link):
    from controller.adapters.outbound.optimization.compact_transmission import (
        add_compact_transmission_constraints,
    )
    from controller.adapters.outbound.optimization.compact_variables import (
        add_compact_variables,
    )
    from controller.adapters.outbound.optimization.gurobi_problem_builder import (
        DefaultGurobiProblemBuilder,
    )
    from controller.domain.profile.network_topology import NetworkTopology

    workers = (_worker(1), _worker(2))
    request = _request(
        workers[0], arrival_rate=1, max_response_time_s=10, flow_id=_flow(1)
    )
    requests = {request.flow_id: request}
    context = _context(dict.fromkeys(workers, 0.1), throughput_bytes_per_s=4)
    if missing_link:
        from controller.domain.optimization.solving.optimization_context import (
            OptimizationContext,
        )

        context = OptimizationContext(
            static_profiles_by_variant=context.static_profiles_by_variant,
            network_topology=NetworkTopology(workers=workers, connections=()),
            execution_profiles_by_worker=context.execution_profiles_by_worker,
            resource_profiles_by_worker=context.resource_profiles_by_worker,
        )
    config = _problem_config(2)
    problem = DefaultGurobiProblemBuilder().build(requests, context, config)
    variant = problem.models_by_flow[request.flow_id][0]
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        variables = add_compact_variables(
            model,
            problem,
            context=context,
            requests_by_flow=requests,
            max_components=2,
            priority_count=1,
            compute_utilization_limit=0.9,
            network_utilization_limit=0.9,
        )
        for (
            _flow_id,
            component,
            worker,
        ), e in variables.assignment.component_worker.items():
            e.LB = e.UB = int(worker == workers[component - 1])
        for (
            _flow_id,
            _variant_id,
            layer,
            component,
        ), x in variables.assignment.layer_component.items():
            x.LB = x.UB = int(component == (1 if layer == "input" else 2))
        add_compact_transmission_constraints(
            model,
            problem,
            variables.assignment,
            variables.network,
            context=context,
            requests_by_flow=requests,
            max_components=2,
        )
        times = variables.network.transmission_time_s
        model.setObjective(gp.quicksum(times.values()))
        model.optimize()
        assert model.Status == (gp.GRB.INFEASIBLE if missing_link else gp.GRB.OPTIMAL)
        if not missing_link:
            assert pytest.approx(1) == times[request.flow_id, variant, 1, workers[0]].X
            assert pytest.approx(0) == times[request.flow_id, variant, 1, workers[1]].X
            assert (
                pytest.approx(1)
                == variables.network.adjacent_component_workers[
                    request.flow_id, 1, *workers
                ].X
            )


@pytest.mark.parametrize("selected", [False, True])
def test_memory_uses_profiled_maximum_cut_and_one_server_activation(selected):
    from controller.adapters.outbound.optimization.compact_memory import (
        add_compact_memory_constraints,
        add_compact_memory_variables,
    )
    from controller.adapters.outbound.optimization.gurobi_problem_builder import (
        DefaultGurobiProblemBuilder,
    )
    from controller.adapters.outbound.optimization.gurobi_variables import (
        _add_assignment_variables,
    )

    worker = _worker(1)
    request = _request(worker, arrival_rate=1, max_response_time_s=10, flow_id=_flow(1))
    requests = {request.flow_id: request}
    context = _context({worker: 0.1})
    problem = DefaultGurobiProblemBuilder().build(requests, context, _problem_config(2))
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment = _add_assignment_variables(
            model, problem, context, max_components=2
        )
        memory = add_compact_memory_variables(model, problem, context, max_components=2)
        for e in assignment.component_worker.values():
            e.LB = e.UB = int(selected)
        for u in assignment.model_selected.values():
            u.LB = u.UB = 1
        add_compact_memory_constraints(
            model, problem, assignment, memory, context=context, max_components=2
        )
        model.setObjective(memory.max_activation_mb[worker])
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        assert not any(
            variable.VarName.startswith("mem_activation[")
            for variable in model.getVars()
        )
        assert (
            pytest.approx(4e-6 if selected else 0, abs=1e-10)
            == memory.max_activation_mb[worker].X
        )
        assert model.NumGenConstrs == 0


@pytest.mark.parametrize("priorities", [1, 2, 3])
def test_compact_priority_queue_matches_physical_first_priority(priorities):
    solution, stats = solve(
        FormulationKind.COMPACT, components=2, priorities=priorities
    )
    component = solution.flow_solutions[0].components[0]
    assert component.priority == 1
    assert component.mean_execution_queue_time_s == pytest.approx(
        0.01 / (2 * 0.9), abs=1e-6
    )
    assert stats.quadratic_constraints == priorities


def test_queue_bounds_use_full_profile_and_stability_without_deadline_clipping():
    from controller.adapters.outbound.optimization.compact_bounds import (
        prepare_compact_problem,
    )
    from controller.adapters.outbound.optimization.gurobi_problem_builder import (
        DefaultGurobiProblemBuilder,
    )

    worker = _worker(1)
    context = _context({worker: 100})
    request = _request(
        worker, arrival_rate=1, max_response_time_s=0.1, flow_id=_flow(1)
    )
    requests = {request.flow_id: request}
    config = _problem_config(1)
    problem = DefaultGurobiProblemBuilder().build(requests, context, config)
    bounded = prepare_compact_problem(
        problem, context=context, requests_by_flow=requests, problem_config=config
    )
    expected = min(1 * 100**2, config.compute_utilization_limit * 100) / (
        2 * (1 - config.compute_utilization_limit) ** 2
    )
    assert bounded.max_queue_wait_s_by_worker[worker] == pytest.approx(expected)
    assert bounded.max_queue_wait_s_by_flow[request.flow_id] == pytest.approx(expected)
    assert (
        problem.max_queue_wait_s_by_worker[worker]
        > bounded.max_queue_wait_s_by_worker[worker]
    )


def test_queue_bound_keeps_models_even_when_batch_fill_exceeds_deadline():
    from controller.adapters.outbound.optimization.compact_bounds import (
        prepare_compact_problem,
    )
    from controller.adapters.outbound.optimization.gurobi_problem_builder import (
        DefaultGurobiProblemBuilder,
    )

    worker = _worker(1)
    context = _context({worker: 0.1})
    request = _request(worker, arrival_rate=1, max_response_time_s=1, flow_id=_flow(1))
    requests = {request.flow_id: request}
    config = _problem_config(1)
    problem = DefaultGurobiProblemBuilder().build(requests, context, config)
    problem = replace(
        problem, batch_fill_time_s=dict.fromkeys(problem.batch_fill_time_s, 2.0)
    )
    bounded = prepare_compact_problem(
        problem, context=context, requests_by_flow=requests, problem_config=config
    )
    expected = 0.1**2 / (2 * (1 - config.compute_utilization_limit) ** 2)
    assert bounded.max_queue_wait_s_by_worker[worker] == pytest.approx(expected)
    assert bounded.max_queue_wait_s_by_flow[request.flow_id] == pytest.approx(expected)


def test_two_priority_queue_uses_upper_bin_endpoint_and_unscaled_wait():
    from controller.adapters.outbound.optimization.compact_bounds import (
        prepare_compact_problem,
    )
    from controller.adapters.outbound.optimization.compact_execution_queue import (
        add_compact_execution_queue_constraints,
    )
    from controller.adapters.outbound.optimization.compact_variables import (
        add_compact_variables,
    )
    from controller.adapters.outbound.optimization.gurobi_problem_builder import (
        DefaultGurobiProblemBuilder,
    )

    worker = _worker(1)
    context = _context({worker: 0.1})
    requests = {
        request.flow_id: request
        for request in (
            _request(worker, arrival_rate=1, max_response_time_s=10, flow_id=_flow(i))
            for i in (1, 2)
        )
    }
    config = _problem_config(1, priority_count=2)
    problem = prepare_compact_problem(
        DefaultGurobiProblemBuilder().build(requests, context, config),
        context=context,
        requests_by_flow=requests,
        problem_config=config,
    )
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 0
        variables = add_compact_variables(
            model,
            problem,
            context=context,
            requests_by_flow=requests,
            max_components=1,
            priority_count=2,
            compute_utilization_limit=0.9,
            network_utilization_limit=0.9,
        )
        for e in variables.assignment.component_worker.values():
            e.LB = e.UB = 1
        for active in variables.assignment.component_active.values():
            active.LB = active.UB = 1
        for (
            flow,
            _component,
            _host,
            priority,
        ), r in variables.compute_queue.component_priority.items():
            r.LB = r.UB = int(priority == (1 if flow == _flow(1) else 2))
        for time in variables.execution.execution_time_s.values():
            time.LB = time.UB = 0.1
        add_compact_execution_queue_constraints(
            model,
            problem,
            variables.assignment,
            variables.execution,
            variables.compute_queue,
            max_components=1,
            priority_count=2,
            compute_utilization_limit=0.9,
        )
        model.setObjective(
            gp.quicksum(variables.compute_queue.component_queue_wait_s.values())
        )
        model.optimize()
        assert model.Status == gp.GRB.OPTIMAL
        assert (
            pytest.approx(0.02 / (2 * 0.9), abs=1e-6)
            == variables.compute_queue.component_queue_wait_s[_flow(1), 1].X
        )
        assert (
            pytest.approx(0.02 / (2 * 0.5 * 0.8), abs=1e-6)
            == variables.compute_queue.component_queue_wait_s[_flow(2), 1].X
        )


@pytest.mark.parametrize(
    "selected", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.5, 0.3), (0.6, 0.4, 0)]
)
@pytest.mark.parametrize("nominal", [0, 0.2, 0.7, 1])
def test_projected_ps_matches_independent_disaggregated_lp(selected, nominal):
    from controller.adapters.outbound.optimization.compact_network_queue import (
        add_projected_ps_epigraph,
    )

    limits = (0, 0.2, 0.5, 0.8)
    values = []
    for projected in (True, False):
        with gp.Model() as model:
            model.Params.OutputFlag = 0
            g = tuple(model.addVar(lb=value, ub=value) for value in selected)
            time = model.addVar(lb=nominal, ub=nominal)
            response = model.addVar(lb=0)
            if projected:
                add_projected_ps_epigraph(
                    model, response, time, 1, limits, g, name="test_ps"
                )
            else:
                z = tuple(model.addVar(lb=0) for _ in selected)
                for part, bin_variable in zip(z, g, strict=True):
                    model.addConstr(part <= bin_variable)
                model.addConstr(gp.quicksum(z) == time)
                model.addConstr(
                    response
                    >= gp.quicksum(
                        part / (1 - limit)
                        for part, limit in zip(z, limits[1:], strict=True)
                    )
                )
            model.setObjective(response)
            model.optimize()
            assert model.Status == gp.GRB.OPTIMAL
            values.append(response.X)
    assert values[0] == pytest.approx(values[1], abs=1e-8)
