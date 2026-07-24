from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from distributed_inference.adapters.outbound.model_profile.onnx.onnx_model_graph_extractor import (
    OnnxGraphExtractor,
)
from distributed_inference.domain.model_graph_info import (
    INPUT_LAYER_NAME,
    OUTPUT_LAYER_NAME,
    DynamicShapeType,
    EdgeInfo,
    FlopsInfo,
    LayerInfo,
    ModelGraph,
    ModelInfo,
    TensorInfo,
)
from test.contracts.model_graph_extractor_contract import (
    AggregationCase,
    ExtractedGraphExpectation,
    ModelGraphExtractorContract,
)


def _model_info(
    *,
    dynamic_shapes: dict[str, DynamicShapeType] | None = None,
    sequence_sizes: list[int] | None = None,
) -> ModelInfo:
    # model_construct keeps this fixture focused on fields consumed by the
    # extractor and avoids coupling these adapter tests to unrelated metadata.
    return ModelInfo.model_construct(
        name="test-model",
        accuracy=1.0,
        task="classification",
        type="bert",
        dynamic_shapes=dynamic_shapes or {},
        sequence_sizes=sequence_sizes or [1],
        num_heads=1,
        hidden_size=4,
    )


def _flops(values: dict[int, float] | None = None) -> FlopsInfo:
    return FlopsInfo(flops=values or {})


def _layer(
    name: str,
    *,
    inputs: Iterable[str] = (),
    outputs: Iterable[str] = (),
    flops: dict[int, float] | None = None,
    is_input: bool = False,
    is_output: bool = False,
) -> LayerInfo:
    return LayerInfo(
        name=name,
        type=name,
        flops=_flops(flops),
        weights_size=0,
        inputs=set(inputs),
        outputs=set(outputs),
        is_input=is_input,
        is_output=is_output,
        is_aggregated=False,
        aggregated_layers=[],
    )


def _add_edge(
    graph: ModelGraph,
    source: str,
    target: str,
    tensors: Iterable[str],
) -> None:
    graph.add_edge(
        EdgeInfo(
            source=source,
            target=target,
            tensors=set(tensors),
        )
    )


def _make_model(
    *,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "test-graph",
        inputs,
        outputs,
        initializer=initializers or [],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="distributed-inference-tests",
    )


def _write_representative_model(path: Path) -> np.ndarray:
    weight = np.arange(12, dtype=np.float32).reshape(4, 3)

    model = _make_model(
        nodes=[
            helper.make_node(
                "MatMul",
                ["input", "weight"],
                ["hidden"],
                name="matmul",
            ),
            helper.make_node(
                "Relu",
                ["hidden"],
                ["output"],
                name="relu",
            ),
        ],
        inputs=[
            helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                ["batch_size", "sequence_size", 4],
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                ["batch_size", "sequence_size", 3],
            )
        ],
        initializers=[numpy_helper.from_array(weight, name="weight")],
    )
    onnx.save(model, path)
    return weight


def _build_aggregation_case(model_info: ModelInfo) -> AggregationCase:
    level_1 = ModelGraph(model_info=model_info)
    for layer in (
        _layer(
            INPUT_LAYER_NAME,
            inputs={"input"},
            outputs={"input"},
            flops={1: 0, 4: 0},
            is_input=True,
        ),
        _layer(
            "a",
            inputs={"input"},
            outputs={"a_out"},
            flops={1: 10, 4: 40},
        ),
        _layer(
            "b",
            inputs={"a_out"},
            outputs={"b_out"},
            flops={1: 20, 4: 80},
        ),
        _layer(
            "c",
            inputs={"b_out"},
            outputs={"output"},
            flops={1: 30, 4: 120},
        ),
        _layer(
            OUTPUT_LAYER_NAME,
            inputs={"output"},
            outputs={"output"},
            flops={1: 0, 4: 0},
            is_output=True,
        ),
    ):
        level_1.add_layer(layer)

    _add_edge(level_1, INPUT_LAYER_NAME, "a", {"input"})
    _add_edge(level_1, "a", "b", {"a_out"})
    _add_edge(level_1, "b", "c", {"b_out"})
    _add_edge(level_1, "c", OUTPUT_LAYER_NAME, {"output"})

    level_1.set_tensors_map(
        {
            "input": TensorInfo(
                name="input",
                shapes={1: [1, 1, 4], 4: [1, 4, 4]},
                sizes={1: 16, 4: 64},
            ),
            "a_out": TensorInfo(
                name="a_out",
                shapes={1: [1, 1, 4], 4: [1, 4, 4]},
                sizes={1: 16, 4: 64},
            ),
            "b_out": TensorInfo(
                name="b_out",
                shapes={1: [1, 1, 4], 4: [1, 4, 4]},
                sizes={1: 16, 4: 64},
            ),
            "output": TensorInfo(
                name="output",
                shapes={1: [1, 1, 3], 4: [1, 4, 3]},
                sizes={1: 12, 4: 48},
            ),
        }
    )

    level_2 = ModelGraph(model_info=model_info)
    for layer in (
        _layer(
            INPUT_LAYER_NAME,
            inputs={"input"},
            outputs={"input"},
            flops={1: 0, 4: 0},
            is_input=True,
        ),
        _layer(
            "a",
            inputs={"input"},
            outputs={"a_out"},
            flops={1: 0, 4: 0},
        ),
        _layer(
            "fused_bc",
            inputs={"a_out"},
            outputs={"output"},
        ),
        _layer(
            OUTPUT_LAYER_NAME,
            inputs={"output"},
            outputs={"output"},
            flops={1: 0, 4: 0},
            is_output=True,
        ),
    ):
        level_2.add_layer(layer)

    _add_edge(level_2, INPUT_LAYER_NAME, "a", {"input"})
    _add_edge(level_2, "a", "fused_bc", {"a_out"})
    _add_edge(level_2, "fused_bc", OUTPUT_LAYER_NAME, {"output"})

    return AggregationCase(
        level_1_graph=level_1,
        level_2_graph=level_2,
        unchanged_layer="a",
        fused_layer="fused_bc",
        fused_members=frozenset({"b", "c"}),
    )


class TestOnnxGraphExtractorContract(ModelGraphExtractorContract):
    @pytest.fixture
    def extractor(self) -> OnnxGraphExtractor:
        return OnnxGraphExtractor()

    @pytest.fixture
    def representative_model_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "representative.onnx"
        _write_representative_model(path)
        return path

    @pytest.fixture
    def model_info(self) -> ModelInfo:
        return _model_info(
            dynamic_shapes={
                "batch_size": DynamicShapeType.BATCH,
                "sequence_size": DynamicShapeType.SEQUENCE,
            },
            sequence_sizes=[1, 4],
        )

    @pytest.fixture
    def extracted_graph_expectation(self) -> ExtractedGraphExpectation:
        return ExtractedGraphExpectation(
            layers=frozenset(
                {
                    INPUT_LAYER_NAME,
                    "matmul",
                    "relu",
                    OUTPUT_LAYER_NAME,
                }
            ),
            edges={
                (INPUT_LAYER_NAME, "matmul"): frozenset({"input"}),
                ("matmul", "relu"): frozenset({"hidden"}),
                ("relu", OUTPUT_LAYER_NAME): frozenset({"output"}),
            },
            internal_layers=frozenset({"matmul", "relu"}),
            tensor_names=frozenset({"input", "hidden", "output"}),
            positive_flop_layers=frozenset({"matmul"}),
        )

    @pytest.fixture
    def aggregation_case(self, model_info: ModelInfo) -> AggregationCase:
        return _build_aggregation_case(model_info)


class TestOnnxGraphExtractor:
    def test_excludes_initializers_from_layer_inputs_and_counts_weight_bytes(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "weighted.onnx"
        weight = _write_representative_model(path)

        graph = OnnxGraphExtractor().extract_model_graph(
            path,
            _model_info(
                dynamic_shapes={
                    "batch_size": DynamicShapeType.BATCH,
                    "sequence_size": DynamicShapeType.SEQUENCE,
                },
                sequence_sizes=[1],
            ),
            profile_flops=False,
            profile_tensors=False,
        )

        matmul = graph.get_layer_info("matmul")
        assert matmul.inputs == {"input"}
        assert "weight" not in matmul.inputs
        assert matmul.weights_size == weight.nbytes

    def test_ignores_omitted_optional_onnx_inputs_and_outputs(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "optional-inputs.onnx"
        model = _make_model(
            nodes=[
                helper.make_node(
                    "Dropout",
                    ["input", "", ""],
                    ["output", ""],
                    name="dropout",
                )
            ],
            inputs=[
                helper.make_tensor_value_info(
                    "input",
                    TensorProto.FLOAT,
                    [1, 4],
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output",
                    TensorProto.FLOAT,
                    [1, 4],
                )
            ],
        )
        onnx.save(model, path)

        graph = OnnxGraphExtractor().extract_model_graph(
            path,
            _model_info(),
            profile_flops=False,
            profile_tensors=False,
        )

        dropout = graph.get_layer_info("dropout")
        assert dropout.inputs == {"input"}
        assert dropout.outputs == {"output"}

    def test_removes_nodes_not_reachable_from_the_synthetic_input_layer(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "dead-branch.onnx"
        dead_value = numpy_helper.from_array(
            np.asarray([1.0], dtype=np.float32),
            name="dead_value",
        )
        model = _make_model(
            nodes=[
                helper.make_node(
                    "Relu",
                    ["input"],
                    ["output"],
                    name="live_relu",
                ),
                helper.make_node(
                    "Constant",
                    [],
                    ["dead_output"],
                    name="dead_constant",
                    value=dead_value,
                ),
            ],
            inputs=[
                helper.make_tensor_value_info(
                    "input",
                    TensorProto.FLOAT,
                    [1, 4],
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output",
                    TensorProto.FLOAT,
                    [1, 4],
                )
            ],
        )
        onnx.save(model, path)

        graph = OnnxGraphExtractor().extract_model_graph(
            path,
            _model_info(),
            profile_flops=False,
            profile_tensors=False,
        )

        assert graph.has_layer("live_relu")
        assert not graph.has_layer("dead_constant")

    def test_wraps_onnx_validation_errors_with_adapter_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        model = _make_model(nodes=[], inputs=[], outputs=[])

        monkeypatch.setattr(onnx, "load", lambda _: model)
        monkeypatch.setattr(
            OnnxGraphExtractor,
            "_OnnxGraphExtractor__infer_model_shape",
            staticmethod(lambda value: value),
        )

        def raise_validation_error(*_: Any, **__: Any) -> None:
            raise onnx.checker.ValidationError("broken graph")

        monkeypatch.setattr(onnx.checker, "check_model", raise_validation_error)

        with pytest.raises(Exception, match=r"Invalid ONNX model: broken graph"):
            OnnxGraphExtractor().extract_model_graph(
                tmp_path / "ignored.onnx",
                _model_info(),
                profile_flops=False,
                profile_tensors=False,
            )

    @pytest.mark.parametrize(
        ("dimension", "bindings", "expected"),
        [
            (
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [7]
                ).type.tensor_type.shape.dim[0],
                {},
                7,
            ),
            (
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, ["2 * sequence_size + 1"]
                ).type.tensor_type.shape.dim[0],
                {"sequence_size": 4},
                9,
            ),
        ],
    )
    def test_resolves_static_and_symbolic_dimensions(
        self,
        dimension: onnx.TensorShapeProto.Dimension,
        bindings: dict[str, int],
        expected: int,
    ) -> None:
        assert OnnxGraphExtractor._resolve_dimension(dimension, bindings) == expected

    def test_rejects_anonymous_dynamic_dimensions(self) -> None:
        dimension = onnx.TensorShapeProto.Dimension()

        with pytest.raises(
            ValueError,
            match="Anonymous dynamic dimension cannot be resolved",
        ):
            OnnxGraphExtractor._resolve_dimension(dimension, {})

    def test_rejects_unresolved_symbolic_dimensions(self) -> None:
        dimension = onnx.TensorShapeProto.Dimension(dim_param="unknown + 1")

        with pytest.raises(
            ValueError,
            match=r"Unresolved symbolic dimension: unknown \+ 1",
        ):
            OnnxGraphExtractor._resolve_dimension(dimension, {})

    def test_rejects_non_integer_symbolic_dimensions(self) -> None:
        dimension = onnx.TensorShapeProto.Dimension(dim_param="sequence_size / 2")

        with pytest.raises(
            ValueError,
            match="Non-integer symbolic dimension",
        ):
            OnnxGraphExtractor._resolve_dimension(
                dimension,
                {"sequence_size": 3},
            )

    def test_infers_a_branched_fused_group_from_its_boundary_tensors(self) -> None:
        graph = ModelGraph(model_info=_model_info())
        for layer in (
            _layer(
                INPUT_LAYER_NAME,
                inputs={"left_input", "right_input"},
                outputs={"left_input", "right_input"},
                is_input=True,
            ),
            _layer("left", inputs={"left_input"}, outputs={"left_out"}),
            _layer("right", inputs={"right_input"}, outputs={"right_out"}),
            _layer(
                "merge",
                inputs={"left_out", "right_out"},
                outputs={"merged"},
            ),
        ):
            graph.add_layer(layer)

        fused = _layer(
            "fused",
            inputs={"left_input", "right_input"},
            outputs={"merged"},
        )

        assert OnnxGraphExtractor._infer_fused_group(graph, fused) == {
            "left",
            "right",
            "merge",
        }

    def test_returns_none_when_a_fused_output_has_no_basic_graph_producer(
        self,
    ) -> None:
        graph = ModelGraph(model_info=_model_info())
        graph.add_layer(_layer("node", inputs={"input"}, outputs={"known"}))

        fused = _layer(
            "fused",
            inputs={"input"},
            outputs={"unknown"},
        )

        assert OnnxGraphExtractor._infer_fused_group(graph, fused) is None

    def test_excludes_input_layer_from_inferred_fused_group(
        self,
    ) -> None:
        graph = ModelGraph(model_info=_model_info())
        for layer in (
            _layer(
                INPUT_LAYER_NAME,
                inputs={"input"},
                outputs={"input"},
                is_input=True,
            ),
            _layer("first", inputs={"input"}, outputs={"first_out"}),
            _layer("second", inputs={"first_out"}, outputs={"output"}),
        ):
            graph.add_layer(layer)

        fused = _layer(
            "fused",
            inputs=set(),
            outputs={"output"},
        )

        assert OnnxGraphExtractor._infer_fused_group(graph, fused) == {
            "first",
            "second",
        }

    def test_detects_fusion_when_onnx_optimizer_reuses_an_existing_node_name(
        self,
    ) -> None:
        case = _build_aggregation_case(_model_info(sequence_sizes=[1, 4]))
        level_2 = case.level_2_graph

        fused = level_2.get_layer_info("fused_bc")
        level_2.remove_layers_from_iterable({"fused_bc"})
        level_2.add_layer(
            _layer(
                "c",
                inputs=fused.inputs,
                outputs=fused.outputs,
            )
        )
        _add_edge(level_2, "a", "c", {"a_out"})
        _add_edge(level_2, "c", OUTPUT_LAYER_NAME, {"output"})

        aggregated = OnnxGraphExtractor().aggregate_model_graphs(
            case.level_1_graph,
            level_2,
        )

        restored = aggregated.get_layer_info("c")
        assert {layer.name for layer in restored.aggregated_layers} == {"b", "c"}
        assert restored.flops == _flops({1: 50, 4: 200})

    def test_leaves_unknown_fusions_unannotated(self) -> None:
        case = _build_aggregation_case(_model_info(sequence_sizes=[1, 4]))
        fused = case.level_2_graph.get_layer_info("fused_bc")
        fused.outputs = {"unknown_output"}

        aggregated = OnnxGraphExtractor().aggregate_model_graphs(
            case.level_1_graph,
            case.level_2_graph,
        )

        unknown = aggregated.get_layer_info("fused_bc")
        assert unknown.aggregated_layers == []
        assert unknown.flops == FlopsInfo()
