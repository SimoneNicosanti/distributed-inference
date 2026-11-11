"""Full-path checks after independently validating each compact subsystem."""

from dataclasses import replace
from itertools import combinations

import gurobipy as gp
import pytest
from controller.adapters.outbound.optimization.gurobi_solver import GurobiSolver
from test.controller.adapters.outbound.optimization import (
    test_gurobi_solver as fixtures,
)

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from controller.application.optimization.formulation.compact.profile_data import (
    layer_execution_profiles,
    tensor_size_bytes,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_config import (
    FormulationKind,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)


def profiled_context(context):
    profiles = {}
    for variant, profile in context.static_profiles_by_variant.items():
        layers = tuple(profile.topology.layers())
        maximum = 0
        for count in range(len(layers) + 1):
            for subset in combinations(layers, count):
                left = set(subset)
                if any(
                    edge.target in left and edge.source not in left
                    for edge in profile.topology.edges().values()
                ):
                    continue
                cut = sum(
                    tensor_size_bytes(profile, tensor)
                    for tensor, route in profile.topology.tensor_routes().items()
                    if route.source in left
                    and any(target not in left for target in route.destinations)
                )
                maximum = max(maximum, cut)
        profiles[variant] = replace(
            profile,
            shape_profiles=tuple(
                replace(shape, max_cut_size_mib=maximum / 1024**2)
                for shape in profile.shape_profiles
            ),
        )
    return OptimizationContext(
        static_profiles_by_variant=profiles,
        network_topology=context.network_topology,
        execution_profiles_by_worker=context.execution_profiles_by_worker,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "test_solves_multi_variant_two_worker_two_priority_instance",
        "test_solves_two_flows_sharing_a_worker",
        "test_solves_two_flows_two_variants_two_priorities",
        "test_solves_three_flows_two_variants_three_workers_three_priorities",
        "test_solves_branched_dag_on_sparse_asymmetric_network",
        "test_rejects_placement_across_missing_network_connection",
        "test_partitions_and_places_compute_on_faster_worker",
        "test_rejects_unstable_compute_queue",
        "test_rejects_unstable_network_queue",
    ],
)
async def test_existing_operational_cases_with_compact(monkeypatch, case):
    optimize = gp.Model.optimize

    def check_model(model):
        model.update()
        assert model.NumGenConstrs == 0
        implicit = tuple(
            variable
            for variable in model.getVars()
            if variable.VarName.startswith(
                ("u[", "e[", "active[", "last[", "c[", "n[", "q[")
            )
        )
        assert all(variable.VType == gp.GRB.CONTINUOUS for variable in implicit)
        optimize(model)
        if model.SolCount:
            assert all(
                abs(variable.X - round(variable.X)) < 1e-6 for variable in implicit
            )

    monkeypatch.setattr(gp.Model, "optimize", check_model)
    monkeypatch.setattr(
        fixtures,
        "_solver",
        lambda: GurobiSolver(
            GurobiSolverConfig(
                CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
                formulation=FormulationKind.COMPACT,
                threads=1,
                seed=0,
            )
        ),
    )
    for name in ("_context", "_multi_variant_context", "_branched_context"):
        original = getattr(fixtures, name)
        monkeypatch.setattr(
            fixtures,
            name,
            lambda *args, _original=original, **kwargs: profiled_context(
                _original(*args, **kwargs)
            ),
        )
    await getattr(fixtures, case)()


@pytest.mark.parametrize("priorities", [1, 2, 3])
def test_compact_physical_times_and_response_aggregation(priorities):
    worker, edge = fixtures._worker(1), fixtures._worker(2)
    context = profiled_context(fixtures._context({worker: 2, edge: 0.04}))
    request = fixtures._request(
        worker, arrival_rate=0.05, max_response_time_s=1, flow_id=fixtures._flow(1)
    )
    result = GurobiSolver(
        GurobiSolverConfig(
            CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            formulation=FormulationKind.COMPACT,
            threads=1,
            seed=0,
        )
    )._solve(
        {request.flow_id: request},
        context,
        fixtures._problem_config(3, priority_count=priorities),
    )
    flow = result.flow_solutions[0]
    assert flow.mean_response_time_s == pytest.approx(
        flow.mean_batch_fill_time_s
        + sum(
            c.mean_execution_time_s
            + c.mean_execution_queue_time_s
            + c.mean_transmission_time_s
            + c.transmission_latency_s
            for c in flow.components
        ),
        abs=1e-6,
    )
    for component in flow.components:
        profiles = layer_execution_profiles(
            context, flow.model_variant_id, component.worker_id
        )
        physical = sum(
            profiles[layer].execution_time_s for layer in component.layer_keys
        )
        assert component.mean_execution_time_s == pytest.approx(physical, abs=1e-6)
    assert sum(
        component.transmission_latency_s for component in flow.components
    ) == pytest.approx(0.002)


def test_physical_network_queue_and_objective_at_well_scaled_tensor_sizes():
    from controller.domain.profile.model_variant_static_profile import (
        TensorShapeProperty,
    )

    worker, edge = fixtures._worker(1), fixtures._worker(2)
    original = fixtures._context(
        {worker: 2, edge: 0.04}, throughput_bytes_per_s=100_000_000
    )
    context = OptimizationContext(
        static_profiles_by_variant={
            variant: replace(
                profile,
                shape_profiles=tuple(
                    replace(
                        shape,
                        max_cut_size_mib=1e6 / 1024**2,
                        tensor_properties={
                            tensor: TensorShapeProperty((250000,))
                            for tensor in shape.tensor_properties
                        },
                    )
                    for shape in profile.shape_profiles
                ),
            )
            for variant, profile in original.static_profiles_by_variant.items()
        },
        network_topology=original.network_topology,
        execution_profiles_by_worker=original.execution_profiles_by_worker,
        resource_profiles_by_worker={
            host: replace(
                resource,
                ram_profile=replace(
                    resource.ram_profile,
                    capacity_bytes=10_000_000,
                    available_bytes=10_000_000,
                ),
            )
            for host, resource in original.resource_profiles_by_worker.items()
        },
    )
    request = fixtures._request(
        worker, arrival_rate=0.05, max_response_time_s=2, flow_id=fixtures._flow(1)
    )
    expected_wait = 0.05 * 0.04**2 / (2 * (1 - 0.05 * 0.04))
    expected = 0.04 + expected_wait + 2 * 0.01 / (1 - 0.5) + 0.002
    objective = CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
    result = GurobiSolver(
        GurobiSolverConfig(
            objective, formulation=FormulationKind.COMPACT, threads=1, seed=0, mip_gap=0
        )
    )._solve({request.flow_id: request}, context, fixtures._problem_config(3))
    flow = result.flow_solutions[0]
    assert flow.mean_response_time_s == pytest.approx(expected, abs=1e-6)
    assert sum(c.mean_execution_queue_time_s for c in flow.components) == pytest.approx(
        expected_wait, abs=1e-6
    )
    assert tuple(c.mean_transmission_time_s for c in flow.components) == pytest.approx(
        (0.02, 0.02, 0), abs=1e-7
    )
    assert result.objective_value == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("inference_runner", [False, True])
def test_relay_only_worker_keeps_endpoints_and_excludes_computation(inference_runner):
    from controller.domain.optimization.request.local_worker_config import (
        ValidLocalWorkerConfig,
    )

    worker, edge = fixtures._worker(1), fixtures._worker(2)
    context = profiled_context(fixtures._context({worker: 0.01, edge: 0.1}))
    request = fixtures._request(
        worker, arrival_rate=0.1, max_response_time_s=2, flow_id=fixtures._flow(1)
    )
    request = replace(
        request,
        local_worker_config=ValidLocalWorkerConfig(
            worker_id=worker, is_inference_runner=inference_runner
        ),
    )
    result = GurobiSolver(
        GurobiSolverConfig(
            CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            formulation=FormulationKind.COMPACT,
            threads=1,
            seed=0,
        )
    )._solve({request.flow_id: request}, context, fixtures._problem_config(3))
    flow = result.flow_solutions[0]
    assert flow.ingress_worker_id == flow.sink_worker_id == worker
    compute = next(c for c in flow.components if "compute" in c.layer_keys)
    assert compute.worker_id == (worker if inference_runner else edge)
