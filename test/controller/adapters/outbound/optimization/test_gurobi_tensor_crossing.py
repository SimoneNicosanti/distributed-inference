from types import SimpleNamespace
from typing import cast

import gurobipy as gp
import pytest
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_expressions import (
    GurobiAffineTerm,
    tensor_crossing_term,
    transmitted_data_size_mb_expression,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    tensor_size_bytes,
)
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    _tensor_crossing_cut,
)
from controller.adapters.outbound.optimization.gurobi_units import bytes_to_megabytes
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    NetworkVariables,
    _add_network_variables,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
    TensorRoute,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId

_FLOW = cast(FlowId, "flow")
_MODEL = cast(ModelVariantId, "model")
_MAX_COMPONENTS = 3


def _profile(
    edge_specs: tuple[tuple[LayerKey, LayerKey, frozenset[TensorKey]], ...],
    tensor_sizes: dict[TensorKey, int],
) -> ModelVariantStaticProfile:
    layer_keys = {
        layer for source, target, _ in edge_specs for layer in (source, target)
    }
    topology = ModelVariantTopology(
        layers={
            layer: LayerDescription(
                name=layer,
                type="Compute",
                is_input=False,
                is_output=False,
            )
            for layer in layer_keys
        },
        edges=(
            EdgeDescription(source=source, target=target, tensors=tensors)
            for source, target, tensors in edge_specs
        ),
    )
    return cast(
        ModelVariantStaticProfile,
        SimpleNamespace(
            topology=topology,
            invariant_tensor_properties={
                tensor: SimpleNamespace(data_type_size_in_bytes=1)
                for tensor in tensor_sizes
            },
            shape_profiles=(
                SimpleNamespace(
                    tensor_properties={
                        tensor: SimpleNamespace(shape=(size,))
                        for tensor, size in tensor_sizes.items()
                    }
                ),
            ),
        ),
    )


def _problem(profile: ModelVariantStaticProfile) -> GurobiProblem:
    layers = profile.topology.layers_in_topological_order()
    tensors = tuple(profile.topology.tensor_routes())
    return GurobiProblem(
        flows=(_FLOW,),
        workers=(),
        workers_by_flow={_FLOW: ()},
        models_by_flow={_FLOW: (_MODEL,)},
        service_arrival_rate_per_s={},
        batch_fill_time_s={},
        compute_bin_limits=(0.0, 0.9),
        network_bin_limits=(0.0, 0.9),
        max_queue_wait_s_by_worker={},
        max_queue_wait_s_by_flow={},
        flow_index={_FLOW: 0},
        worker_index={},
        model_index={(_FLOW, _MODEL): 0},
        layer_index={(_MODEL, layer): index for index, layer in enumerate(layers)},
        tensor_index={(_MODEL, tensor): index for index, tensor in enumerate(tensors)},
    )


def _assignment(
    model: gp.Model,
    profile: ModelVariantStaticProfile,
    placements: dict[LayerKey, int],
    *,
    selected: bool = True,
) -> AssignmentVariables:
    layer_component: dict[tuple[FlowId, ModelVariantId, LayerKey, int], gp.Var] = {}
    for layer in profile.topology.layers():
        for component in range(1, _MAX_COMPONENTS + 1):
            value = int(selected and placements[layer] == component)
            layer_component[_FLOW, _MODEL, layer, component] = model.addVar(
                lb=value,
                ub=value,
                vtype=GRB.BINARY,
                name=f"x[{layer},{component}]",
            )
    return AssignmentVariables(
        max_components=_MAX_COMPONENTS,
        model_selected={},
        layer_component=layer_component,
        component_worker={},
        component_active={},
        component_last={},
    )


def _network(
    model: gp.Model,
    profile: ModelVariantStaticProfile,
) -> NetworkVariables:
    auxiliary = {
        (_FLOW, _MODEL, tensor, cut): model.addVar(
            lb=0.0,
            ub=1.0,
            name=f"c[{tensor},{cut}]",
        )
        for tensor, route in profile.topology.tensor_routes().items()
        if len(route.maximal_destinations) > 1
        for cut in range(1, _MAX_COMPONENTS)
    }
    return NetworkVariables(
        tensor_crossing_auxiliary=auxiliary,
        adjacent_component_workers={},
        transmission_time_s={},
        network_utilization={},
        network_bin_selected={},
        component_transmission_response_s={},
    )


def _add_current_crossing_constraints(
    model: gp.Model,
    profile: ModelVariantStaticProfile,
    assignment: AssignmentVariables,
    network: NetworkVariables,
) -> None:
    _tensor_crossing_cut(
        model,
        _problem(profile),
        assignment,
        network,
        flow=_FLOW,
        model_variant=_MODEL,
        static_profile=profile,
        flow_index=0,
        model_index=0,
        cuts=range(1, _MAX_COMPONENTS),
    )


def _legacy_crossing_variable(
    model: gp.Model,
    assignment: AssignmentVariables,
    tensor: TensorKey,
    route: TensorRoute,
    cut: int,
) -> gp.Var:
    crosses = model.addVar(lb=0.0, ub=1.0, name=f"legacy_c[{tensor},{cut}]")
    source_before = gp.quicksum(
        assignment.layer_component[_FLOW, _MODEL, route.source, component]
        for component in range(1, cut + 1)
    )
    destinations_after = gp.quicksum(
        assignment.layer_component[_FLOW, _MODEL, destination, component]
        for destination in route.destinations
        for component in range(cut + 1, _MAX_COMPONENTS + 1)
    )
    model.addConstr(crosses <= source_before)
    model.addConstr(crosses <= destinations_after)
    for destination in route.destinations:
        destination_after = gp.quicksum(
            assignment.layer_component[_FLOW, _MODEL, destination, component]
            for component in range(cut + 1, _MAX_COMPONENTS + 1)
        )
        model.addConstr(crosses >= source_before + destination_after - 1)
    return crosses


def _value(term: GurobiAffineTerm) -> float:
    return float(term.X if isinstance(term, gp.Var) else term.getValue())


def _single_consumer_profile() -> ModelVariantStaticProfile:
    return _profile(
        (("source", "destination", frozenset({"tensor"})),),
        {"tensor": 4},
    )


@pytest.mark.parametrize(
    ("placements", "cut", "expected"),
    [
        ({"source": 1, "destination": 1}, 1, 0.0),
        ({"source": 1, "destination": 2}, 1, 1.0),
        ({"source": 2, "destination": 3}, 1, 0.0),
        ({"source": 2, "destination": 3}, 2, 1.0),
    ],
)
def test_single_consumer_expression_matches_legacy_formulation(
    placements: dict[LayerKey, int],
    cut: int,
    expected: float,
) -> None:
    profile = _single_consumer_profile()
    route = profile.topology.tensor_routes()["tensor"]
    model = gp.Model()
    model.Params.OutputFlag = 0
    assignment = _assignment(model, profile, placements)
    network = _network(model, profile)
    _add_current_crossing_constraints(model, profile, assignment, network)
    current = tensor_crossing_term(
        assignment,
        network,
        flow=_FLOW,
        model_variant=_MODEL,
        tensor="tensor",
        route=route,
        cut=cut,
    )
    legacy = _legacy_crossing_variable(model, assignment, "tensor", route, cut)

    model.optimize()

    assert model.Status == GRB.OPTIMAL
    assert not network.tensor_crossing_auxiliary
    assert _value(current) == pytest.approx(expected)
    assert _value(current) == pytest.approx(legacy.X)


def _mixed_profile() -> ModelVariantStaticProfile:
    return _profile(
        (
            ("source", "single", frozenset({"single"})),
            ("source", "a", frozenset({"comparable", "transitive"})),
            ("source", "b", frozenset({"comparable"})),
            ("source", "c", frozenset({"comparable", "transitive"})),
            ("a", "b", frozenset()),
            ("b", "c", frozenset()),
            ("source", "left", frozenset({"parallel"})),
            ("source", "right", frozenset({"parallel"})),
            ("single", "output", frozenset()),
            ("c", "output", frozenset()),
            ("left", "output", frozenset()),
            ("right", "output", frozenset()),
        ),
        {
            "single": 2,
            "comparable": 3,
            "transitive": 5,
            "parallel": 7,
            "unused": 11,
        },
    )


def _mixed_placements() -> dict[LayerKey, int]:
    return {
        "source": 1,
        "single": 2,
        "a": 1,
        "b": 2,
        "c": 3,
        "left": 2,
        "right": 3,
        "output": 3,
    }


@pytest.mark.parametrize("selected", [True, False])
def test_compact_crossings_and_transmitted_volume_match_legacy(
    selected: bool,
) -> None:
    profile = _mixed_profile()
    routes = profile.topology.tensor_routes()
    model = gp.Model()
    model.Params.OutputFlag = 0
    assignment = _assignment(
        model,
        profile,
        _mixed_placements(),
        selected=selected,
    )
    network = _network(model, profile)
    _add_current_crossing_constraints(model, profile, assignment, network)
    model.update()

    assert routes["comparable"].maximal_destinations == ("c",)
    assert routes["transitive"].maximal_destinations == ("c",)
    assert routes["parallel"].maximal_destinations == ("left", "right")
    assert "unused" not in routes
    assert len(network.tensor_crossing_auxiliary) == 2
    assert model.NumConstrs == 6

    legacy = {
        (tensor, cut): _legacy_crossing_variable(model, assignment, tensor, route, cut)
        for tensor, route in routes.items()
        for cut in range(1, _MAX_COMPONENTS)
    }
    model.update()
    assert len(legacy) == 8
    assert model.NumConstrs - 6 == 32

    current_volumes = {
        cut: transmitted_data_size_mb_expression(
            assignment,
            network,
            profile,
            flow=_FLOW,
            model_variant=_MODEL,
            cut=cut,
        )
        for cut in range(1, _MAX_COMPONENTS)
    }
    legacy_volumes = {
        cut: gp.quicksum(
            bytes_to_megabytes(tensor_size_bytes(profile, tensor))
            * legacy[tensor, cut]
            for tensor in routes
        )
        for cut in range(1, _MAX_COMPONENTS)
    }

    model.optimize()

    assert model.Status == GRB.OPTIMAL
    for tensor, route in routes.items():
        for cut in range(1, _MAX_COMPONENTS):
            current = tensor_crossing_term(
                assignment,
                network,
                flow=_FLOW,
                model_variant=_MODEL,
                tensor=tensor,
                route=route,
                cut=cut,
            )
            assert _value(current) == pytest.approx(legacy[tensor, cut].X)

    expected_volumes = (17e-6, 15e-6) if selected else (0.0, 0.0)
    for cut, expected in enumerate(expected_volumes, start=1):
        assert current_volumes[cut].getValue() == pytest.approx(expected)
        assert current_volumes[cut].getValue() == pytest.approx(
            legacy_volumes[cut].getValue()
        )


def test_network_variable_builder_creates_only_parallel_crossing_auxiliaries() -> None:
    profile = _mixed_profile()
    model = gp.Model()
    model.Params.OutputFlag = 0
    context = cast(
        OptimizationContext,
        SimpleNamespace(
            static_profiles_by_variant={_MODEL: profile},
            network_topology=SimpleNamespace(connections=lambda: {}),
        ),
    )
    requests = cast(
        OptimizationRequestsByFlow,
        {
            _FLOW: SimpleNamespace(
                qos_requirements=SimpleNamespace(max_mean_response_time=10.0)
            )
        },
    )

    network = _add_network_variables(
        model,
        _problem(profile),
        context,
        requests,
        max_components=_MAX_COMPONENTS,
        network_utilization_limit=0.9,
    )
    model.update()

    assert set(network.tensor_crossing_auxiliary) == {
        (_FLOW, _MODEL, "parallel", 1),
        (_FLOW, _MODEL, "parallel", 2),
    }
    assert sum(variable.VarName.startswith("c[") for variable in model.getVars()) == 2


def test_tensor_without_consumers_is_constant_zero_without_model_artifacts() -> None:
    profile = _single_consumer_profile()
    model = gp.Model()
    model.Params.OutputFlag = 0
    assignment = _assignment(
        model,
        profile,
        {"source": 1, "destination": 2},
    )
    network = _network(model, profile)
    model.update()
    variable_count = model.NumVars
    constraint_count = model.NumConstrs

    term = tensor_crossing_term(
        assignment,
        network,
        flow=_FLOW,
        model_variant=_MODEL,
        tensor="unused",
        route=TensorRoute(
            source="source",
            destinations=(),
            maximal_destinations=(),
        ),
        cut=1,
    )
    model.update()

    assert isinstance(term, gp.LinExpr)
    assert term.size() == 0
    assert term.getConstant() == 0.0
    assert model.NumVars == variable_count
    assert model.NumConstrs == constraint_count
