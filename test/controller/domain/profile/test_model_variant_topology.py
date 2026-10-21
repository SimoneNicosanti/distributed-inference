from controller.domain.profile.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from shared.model.keys import LayerKey, TensorKey


def _topology(
    edge_specs: tuple[tuple[LayerKey, LayerKey, frozenset[TensorKey]], ...],
) -> ModelVariantTopology:
    layer_keys = {layer for source, target, _ in edge_specs for layer in (source, target)}
    return ModelVariantTopology(
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


def test_tensor_route_with_one_consumer_has_one_maximal_destination() -> None:
    route = _topology(
        (("source", "consumer", frozenset({"t"})),)
    ).tensor_routes()["t"]

    assert route.destinations == ("consumer",)
    assert route.maximal_destinations == ("consumer",)


def test_tensor_route_keeps_only_last_of_comparable_consumers() -> None:
    topology = _topology(
        (
            ("source", "a", frozenset({"t"})),
            ("source", "b", frozenset({"t"})),
            ("source", "c", frozenset({"t"})),
            ("a", "b", frozenset()),
            ("b", "c", frozenset()),
        )
    )

    assert topology.tensor_routes()["t"].maximal_destinations == ("c",)


def test_tensor_route_uses_transitive_reachability() -> None:
    topology = _topology(
        (
            ("source", "a", frozenset({"t"})),
            ("source", "c", frozenset({"t"})),
            ("a", "middle", frozenset()),
            ("middle", "c", frozenset()),
        )
    )

    assert topology.is_reachable("a", "c")
    assert topology.tensor_routes()["t"].maximal_destinations == ("c",)


def test_tensor_route_preserves_parallel_maximal_consumers() -> None:
    topology = _topology(
        (
            ("source", "left", frozenset({"t"})),
            ("source", "right", frozenset({"t"})),
            ("left", "output", frozenset()),
            ("right", "output", frozenset()),
        )
    )

    assert topology.tensor_routes()["t"].maximal_destinations == ("left", "right")


def test_tensor_without_consumers_has_no_route() -> None:
    routes = _topology(
        (("source", "consumer", frozenset({"used"})),)
    ).tensor_routes()

    assert "unused" not in routes


def _topology_with_model_boundaries() -> ModelVariantTopology:
    return ModelVariantTopology(
        layers={
            "input": LayerDescription(
                name="input",
                type="Input",
                is_input=True,
                is_output=False,
            ),
            "first": LayerDescription(
                name="first",
                type="Compute",
                is_input=False,
                is_output=False,
            ),
            "second": LayerDescription(
                name="second",
                type="Compute",
                is_input=False,
                is_output=False,
            ),
            "output": LayerDescription(
                name="output",
                type="Output",
                is_input=False,
                is_output=True,
            ),
        },
        edges=(
            EdgeDescription(
                source="input",
                target="first",
                tensors=frozenset({"model_input"}),
            ),
            EdgeDescription(
                source="first",
                target="second",
                tensors=frozenset({"intermediate"}),
            ),
            EdgeDescription(
                source="second",
                target="output",
                tensors=frozenset({"model_output"}),
            ),
        ),
    )


def test_component_contract_includes_input_carried_by_input_layer() -> None:
    contract = _topology_with_model_boundaries().component_tensor_contract(
        ("input", "first")
    )

    assert contract.input_tensor_keys == ("model_input",)
    assert contract.output_tensor_keys == ("intermediate",)


def test_component_contract_includes_output_carried_by_output_layer() -> None:
    contract = _topology_with_model_boundaries().component_tensor_contract(
        ("second", "output")
    )

    assert contract.input_tensor_keys == ("intermediate",)
    assert contract.output_tensor_keys == ("model_output",)


def test_synthetic_boundary_layer_is_a_pass_through_component() -> None:
    topology = _topology_with_model_boundaries()

    input_contract = topology.component_tensor_contract(("input",))
    output_contract = topology.component_tensor_contract(("output",))

    assert input_contract.input_tensor_keys == ("model_input",)
    assert input_contract.output_tensor_keys == ("model_input",)
    assert output_contract.input_tensor_keys == ("model_output",)
    assert output_contract.output_tensor_keys == ("model_output",)
