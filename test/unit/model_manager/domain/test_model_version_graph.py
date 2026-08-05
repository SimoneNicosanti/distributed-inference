import pytest

from distributed_inference.model_manager.domain.model_version_graph import (
    AGGREGATED_LAYER_TYPE,
    EdgeInfo,
    FlopsInfo,
    LayerInfo,
    ModelVersionGraph,
    ShapePoint,
    TensorInfo,
)

BATCH_1 = ShapePoint(dims=(("batch_size", 1),))
BATCH_8 = ShapePoint(dims=(("batch_size", 8),))


def _layer(
    name: str,
    *,
    inputs: set[str],
    outputs: set[str],
    flops: tuple[float, float] = (10.0, 80.0),
    weights_size: float = 2.0,
    is_input: bool = False,
    is_output: bool = False,
) -> LayerInfo:
    return LayerInfo(
        name=name,
        type="TestOp",
        flops=FlopsInfo(flops={BATCH_1: flops[0], BATCH_8: flops[1]}),
        weights_size=weights_size,
        inputs=inputs,
        outputs=outputs,
        is_input=is_input,
        is_output=is_output,
        is_aggregated=False,
        aggregated_layers=[],
    )


def _linear_graph() -> ModelVersionGraph:
    graph = ModelVersionGraph(shape_points=[BATCH_1, BATCH_8])
    for layer in (
        _layer(
            "input",
            inputs={"input_tensor"},
            outputs={"input_tensor"},
            is_input=True,
        ),
        _layer("first", inputs={"input_tensor"}, outputs={"hidden"}),
        _layer(
            "second",
            inputs={"hidden"},
            outputs={"output_tensor"},
            flops=(20.0, 160.0),
            weights_size=3.0,
        ),
        _layer(
            "output",
            inputs={"output_tensor"},
            outputs={"output_tensor"},
            is_output=True,
        ),
    ):
        graph.add_layer(layer)

    for edge in (
        EdgeInfo(source="input", target="first", tensors={"input_tensor"}),
        EdgeInfo(source="first", target="second", tensors={"hidden"}),
        EdgeInfo(source="second", target="output", tensors={"output_tensor"}),
    ):
        graph.add_edge(edge)

    graph.set_tensors_map(
        {
            "input_tensor": TensorInfo(
                name="input_tensor",
                shapes={BATCH_1: [1, 3], BATCH_8: [8, 3]},
                sizes={BATCH_1: 12.0, BATCH_8: 96.0},
            ),
            "output_tensor": TensorInfo(
                name="output_tensor",
                shapes={BATCH_1: [1, 4], BATCH_8: [8, 4]},
                sizes={BATCH_1: 16.0, BATCH_8: 128.0},
            ),
        }
    )
    return graph


@pytest.mark.unit
def test_flops_addition_combines_matching_shape_points() -> None:
    total = FlopsInfo(flops={BATCH_1: 10.0, BATCH_8: 80.0}) + FlopsInfo(
        flops={BATCH_1: 5.0, BATCH_8: 40.0}
    )

    assert total.flops == {BATCH_1: 15.0, BATCH_8: 120.0}

    with pytest.raises(ValueError, match="same shape points"):
        FlopsInfo(flops={BATCH_1: 10.0}) + FlopsInfo(flops={BATCH_8: 80.0})


@pytest.mark.unit
def test_flops_addition_treats_an_empty_operand_as_neutral() -> None:
    populated = FlopsInfo(flops={BATCH_1: 10.0})

    assert FlopsInfo() + populated == populated
    assert populated + FlopsInfo() == populated


@pytest.mark.unit
def test_graph_queries_report_reachability_and_boundaries() -> None:
    graph = _linear_graph()

    assert graph.is_dag()
    assert graph.has_path("input", "output")
    assert set(graph.get_reachable_from_layer("first")) == {
        "first",
        "second",
        "output",
    }
    assert graph.get_in_out_layer_degree("first") == (1, 1)
    assert graph.find_tensor_producer("hidden") == "first"
    assert graph.find_tensor_consumer_set("hidden") == {"second"}
    assert graph.get_topological_sort() == ["input", "first", "second", "output"]
    assert graph.extract_incoming_outgoing_tensors_of_sub_model(
        {"first", "second"}
    ) == ({"input_tensor"}, {"output_tensor"})


@pytest.mark.unit
def test_sub_model_boundaries_include_model_input_and_output_tensors() -> None:
    graph = _linear_graph()

    incoming, outgoing = graph.extract_incoming_outgoing_tensors_of_sub_model(
        {"input", "first"}
    )

    assert incoming == {"input_tensor"}
    assert outgoing == {"hidden"}


@pytest.mark.unit
def test_tensors_map_round_trips_the_registered_tensor_info() -> None:
    graph = _linear_graph()

    tensors_map = graph.get_tensors_map()

    assert set(tensors_map) == {"input_tensor", "output_tensor"}
    assert tensors_map["input_tensor"].sizes[BATCH_8] == 96.0
    assert tensors_map["output_tensor"].shapes[BATCH_1] == [1, 4]


@pytest.mark.unit
def test_contracting_an_internal_edge_rewires_graph_and_combines_metadata() -> None:
    graph = _linear_graph()

    graph.contract_edge_layers(("first", "second"))

    aggregated = graph.get_layer_info("first∘second")
    assert aggregated.type == AGGREGATED_LAYER_TYPE
    assert aggregated.is_aggregated
    assert aggregated.flops == FlopsInfo(flops={BATCH_1: 30.0, BATCH_8: 240.0})
    assert aggregated.weights_size == 5.0
    assert aggregated.inputs == {"input_tensor"}
    assert aggregated.outputs == {"output_tensor"}
    assert [layer.name for layer in aggregated.aggregated_layers] == [
        "first",
        "second",
    ]
    assert set(graph.get_all_layers()) == {"input", "first∘second", "output"}
    assert set(graph.get_all_edges()) == {
        ("input", "first∘second"),
        ("first∘second", "output"),
    }
    assert graph.get_topological_sort() == ["input", "first∘second", "output"]


@pytest.mark.unit
def test_edge_with_an_alternative_path_is_not_contractible() -> None:
    graph = _linear_graph()
    graph.add_edge(EdgeInfo(source="first", target="output", tensors={"hidden"}))

    assert not graph.is_edge_contractible(("first", "output"))
    assert not graph.is_edge_contractible(("input", "first"))
    assert not graph.is_edge_contractible(("input", "output"))

    with pytest.raises(ValueError, match="cannot be contracted"):
        graph.contract_edge_layers(("input", "first"))
