from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from distributed_inference.model_manager.domain.model_version_graph import (
    ModelInfo,
    ModelType,
    ModelVersionGraph,
    TaskType,
)
from distributed_inference.model_optimizer.application.ports.outbound.model_optimizer import (
    ModelOptimizer,
)
from distributed_inference.model_optimizer.domain.optimization_level import (
    OptimizationLevel,
)
from distributed_inference.model_profiler.application.ports.outbound.model_graph_extractor import (
    ModelGraphExtractor,
)
from distributed_inference.model_profiler.application.services.default_model_profiler import (
    DefaultModelProfiler,
)
from test.support.artifact_materializer.materialized_artifact_test_utils import (
    build_test_materialized_artifact,
)


def _model_info() -> ModelInfo:
    return ModelInfo(
        name="vision-model",
        accuracy=0.92,
        task=TaskType.CLASSIFICATION,
        type=ModelType.VIT,
        dynamic_shapes={},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_profile_model_runs_both_optimization_levels_then_aggregates(
    tmp_path: Path,
) -> None:
    optimizer = MagicMock(spec=ModelOptimizer)
    extractor = MagicMock(spec=ModelGraphExtractor)
    basic_graph = MagicMock(spec=ModelVersionGraph)
    extended_graph = MagicMock(spec=ModelVersionGraph)
    aggregated_graph = MagicMock(spec=ModelVersionGraph)
    extractor.extract_model_graph.side_effect = [basic_graph, extended_graph]
    extractor.aggregate_model_graphs.return_value = aggregated_graph
    profiler = DefaultModelProfiler(optimizer, extractor)
    (tmp_path / "model.onnx").write_bytes(b"model")
    source_paths = build_test_materialized_artifact(
        tmp_path / "model.onnx", root_path=tmp_path
    )
    model_info = _model_info()

    result = await profiler.profile_model_version(source_paths, model_info)

    assert result is aggregated_graph
    assert optimizer.optimize_model.call_count == 2
    basic_call, extended_call = optimizer.optimize_model.call_args_list
    assert basic_call.args[0] is source_paths
    assert basic_call.args[2:] == (model_info, OptimizationLevel.BASIC)
    assert extended_call.args[0] is source_paths
    assert extended_call.args[2:] == (model_info, OptimizationLevel.EXTENDED)
    assert isinstance(basic_call.args[1].root_path, Path)
    assert basic_call.args[1].root_path == extended_call.args[1].root_path
    assert extractor.extract_model_graph.call_args_list == [
        call(
            basic_call.args[1],
            model_info,
            profile_flops=True,
            profile_tensors=True,
        ),
        call(
            extended_call.args[1],
            model_info,
            profile_flops=False,
            profile_tensors=False,
        ),
    ]
    extractor.aggregate_model_graphs.assert_called_once_with(
        basic_graph,
        extended_graph,
    )
