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
