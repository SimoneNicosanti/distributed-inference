from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.model_manager.domain.model_version_graph import (
    ModelVersionGraph,
)
from distributed_inference.model_splitter.adapters.outbound.onnx.onnx_model_splitter import (
    OnnxModelSplitter,
)
from test.support.artifact_materializer.materialized_artifact_test_utils import (
    build_test_materialized_artifact,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_split_model_extracts_boundary_tensors_and_sets_output_entrypoint(
    tmp_path: Path,
) -> None:
    input_model = tmp_path / "input" / "model.onnx"
    input_model.parent.mkdir()
    input_model.write_bytes(b"model")
    output_root = tmp_path / "output"
    output_root.mkdir()
    input_paths = build_test_materialized_artifact(input_model)
    output_paths = ArtifactWorkspace(root_path=output_root)
    model_graph = MagicMock(spec=ModelVersionGraph)
    model_graph.extract_incoming_outgoing_tensors_of_sub_model.return_value = (
        {"input_ids", "attention_mask"},
        {"logits"},
    )

    def create_output(*, output_path: Path, **_kwargs: object) -> None:
        output_path.write_bytes(b"split-model")

    with patch(
        "distributed_inference.model_splitter.adapters.outbound.onnx."
        "onnx_model_splitter.extract_model",
        side_effect=create_output,
    ) as extract_model:
        await OnnxModelSplitter().split_model(
            model_graph,
            ["encoder.1", "encoder.2", "encoder.1"],
            input_paths,
            output_paths,
        )

    model_graph.extract_incoming_outgoing_tensors_of_sub_model.assert_called_once_with(
        {"encoder.1", "encoder.2"}
    )
    assert output_paths.entrypoint_path == output_root / "split_model.onnx"
    kwargs = extract_model.call_args.kwargs
    assert kwargs["input_path"] == input_model
    assert kwargs["output_path"] == output_root / "split_model.onnx"
    assert set(kwargs["input_names"]) == {"input_ids", "attention_mask"}
    assert kwargs["output_names"] == ["logits"]
    assert kwargs["check_model"] is True
    assert kwargs["infer_shapes"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_split_model_rejects_empty_components_before_touching_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        await OnnxModelSplitter().split_model(
            MagicMock(spec=ModelVersionGraph),
            [],
            MagicMock(),
            ArtifactWorkspace(root_path=tmp_path),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ((set(), {"output"}), "no inputs"),
        (({"input"}, set()), "no outputs"),
    ],
)
@pytest.mark.asyncio
async def test_split_model_rejects_components_without_complete_boundaries(
    tmp_path: Path,
    boundaries: tuple[set[str], set[str]],
    message: str,
) -> None:
    input_model = tmp_path / "model.onnx"
    input_model.write_bytes(b"model")
    output_root = tmp_path / "output"
    output_root.mkdir()
    graph = MagicMock(spec=ModelVersionGraph)
    graph.extract_incoming_outgoing_tensors_of_sub_model.return_value = boundaries

    with pytest.raises(ValueError, match=message):
        await OnnxModelSplitter().split_model(
            graph,
            ["layer"],
            build_test_materialized_artifact(input_model, root_path=tmp_path),
            ArtifactWorkspace(root_path=output_root),
        )
