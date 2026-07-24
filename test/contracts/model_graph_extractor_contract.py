from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from distributed_inference.application.model_profile.contracts.model_graph_extractor import (
    ModelGraphExtractor,
)
from distributed_inference.domain.model_graph_info import (
    FlopsInfo,
    ModelGraph,
    ModelInfo,
)


@dataclass(frozen=True)
class ExtractedGraphExpectation:
    layers: frozenset[str]
    edges: Mapping[tuple[str, str], frozenset[str]]
    internal_layers: frozenset[str]
    tensor_names: frozenset[str]
    positive_flop_layers: frozenset[str]


@dataclass(frozen=True)
class AggregationCase:
    level_1_graph: ModelGraph
    level_2_graph: ModelGraph
    unchanged_layer: str
    fused_layer: str
    fused_members: frozenset[str]


def _normalise(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalise(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {key: _normalise(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_normalise(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalise(item) for item in value)
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def _graph_snapshot(graph: ModelGraph) -> dict[str, Any]:
    return {
        "model_info": _normalise(graph.model_info),
        "layers": {
            name: _normalise(layer_info)
            for name, layer_info in graph.get_all_layers().items()
        },
        "edges": {
            edge: _normalise(edge_info)
            for edge, edge_info in graph.get_all_edges().items()
        },
        "tensors": _normalise(graph.get_tensors_map()),
    }


class ModelGraphExtractorContract:
    """Reusable behavioural contract for every ModelGraphExtractor adapter."""

    @pytest.fixture
    def extractor(self) -> ModelGraphExtractor:
        raise NotImplementedError

    @pytest.fixture
    def representative_model_path(self) -> Path:
        raise NotImplementedError

    @pytest.fixture
    def model_info(self) -> ModelInfo:
        raise NotImplementedError

    @pytest.fixture
    def extracted_graph_expectation(self) -> ExtractedGraphExpectation:
        raise NotImplementedError

    @pytest.fixture
    def aggregation_case(self) -> AggregationCase:
        raise NotImplementedError

    def test_extract_model_graph_preserves_metadata_and_topology(
        self,
        extractor: ModelGraphExtractor,
        representative_model_path: Path,
        model_info: ModelInfo,
        extracted_graph_expectation: ExtractedGraphExpectation,
    ) -> None:
        graph = extractor.extract_model_graph(
            representative_model_path,
            model_info,
            profile_flops=False,
            profile_tensors=False,
        )

        assert graph.model_info == model_info
        assert frozenset(graph.get_all_layers()) == extracted_graph_expectation.layers

        actual_edges = {
            edge: frozenset(edge_info.tensors)
            for edge, edge_info in graph.get_all_edges().items()
        }
        assert actual_edges == extracted_graph_expectation.edges

    def test_extract_model_graph_honours_flops_profiling_flag(
        self,
        extractor: ModelGraphExtractor,
        representative_model_path: Path,
        model_info: ModelInfo,
        extracted_graph_expectation: ExtractedGraphExpectation,
    ) -> None:
        unprofiled_graph = extractor.extract_model_graph(
            representative_model_path,
            model_info,
            profile_flops=False,
            profile_tensors=False,
        )
        profiled_graph = extractor.extract_model_graph(
            representative_model_path,
            model_info,
            profile_flops=True,
            profile_tensors=False,
        )

        for layer_name in extracted_graph_expectation.internal_layers:
            assert unprofiled_graph.get_layer_info(layer_name).flops.flops == {}

            profiled_flops = profiled_graph.get_layer_info(layer_name).flops.flops
            assert set(profiled_flops) == set(model_info.sequence_sizes)
            assert all(value >= 0 for value in profiled_flops.values())

        for layer_name in extracted_graph_expectation.positive_flop_layers:
            assert all(
                value > 0
                for value in profiled_graph.get_layer_info(
                    layer_name
                ).flops.flops.values()
            )

    def test_extract_model_graph_honours_tensor_profiling_flag(
        self,
        extractor: ModelGraphExtractor,
        representative_model_path: Path,
        model_info: ModelInfo,
        extracted_graph_expectation: ExtractedGraphExpectation,
    ) -> None:
        unprofiled_graph = extractor.extract_model_graph(
            representative_model_path,
            model_info,
            profile_flops=False,
            profile_tensors=False,
        )
        profiled_graph = extractor.extract_model_graph(
            representative_model_path,
            model_info,
            profile_flops=False,
            profile_tensors=True,
        )

        assert unprofiled_graph.get_tensors_map() == {}

        tensors_map = profiled_graph.get_tensors_map()
        assert frozenset(tensors_map) == extracted_graph_expectation.tensor_names

        for tensor_info in tensors_map.values():
            assert set(tensor_info.shapes) == set(model_info.sequence_sizes)
            assert set(tensor_info.sizes) == set(model_info.sequence_sizes)
            assert all(size > 0 for size in tensor_info.sizes.values())

    def test_aggregate_model_graphs_combines_metadata_without_mutating_inputs(
        self,
        extractor: ModelGraphExtractor,
        aggregation_case: AggregationCase,
    ) -> None:
        level_1_before = _graph_snapshot(aggregation_case.level_1_graph)
        level_2_before = _graph_snapshot(aggregation_case.level_2_graph)

        aggregated = extractor.aggregate_model_graphs(
            aggregation_case.level_1_graph,
            aggregation_case.level_2_graph,
        )

        assert aggregated is not aggregation_case.level_1_graph
        assert aggregated is not aggregation_case.level_2_graph
        assert _graph_snapshot(aggregation_case.level_1_graph) == level_1_before
        assert _graph_snapshot(aggregation_case.level_2_graph) == level_2_before

        assert set(aggregated.get_all_layers()) == set(
            aggregation_case.level_2_graph.get_all_layers()
        )
        assert set(aggregated.get_all_edges()) == set(
            aggregation_case.level_2_graph.get_all_edges()
        )
        assert aggregated.model_info == aggregation_case.level_2_graph.model_info

        assert (
            aggregated.get_layer_info(aggregation_case.unchanged_layer).flops
            == aggregation_case.level_1_graph.get_layer_info(
                aggregation_case.unchanged_layer
            ).flops
        )

        fused_layer = aggregated.get_layer_info(aggregation_case.fused_layer)
        assert {layer.name for layer in fused_layer.aggregated_layers} == set(
            aggregation_case.fused_members
        )
        assert fused_layer.is_aggregated is False

        expected_flops = sum(
            (
                aggregation_case.level_1_graph.get_layer_info(layer_name).flops
                for layer_name in aggregation_case.fused_members
            ),
            start=FlopsInfo(),
        )
        assert fused_layer.flops == expected_flops
        assert (
            aggregated.get_tensors_map()
            == aggregation_case.level_1_graph.get_tensors_map()
        )
