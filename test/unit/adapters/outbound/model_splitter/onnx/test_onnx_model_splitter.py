from unittest.mock import MagicMock, patch

import pytest

from distributed_inference.adapters.outbound.model_splitter.onnx.onnx_model_splitter import (
    OnnxModelSplitter,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.domain.model_graph_info import ModelGraph


@pytest.mark.unit
def test_split_model_extracts_boundary_tensors_and_sets_output_entrypoint(
    tmp_path,
) -> None:
    input_model = tmp_path / "input" / "model.onnx"
    input_model.parent.mkdir()
    input_model.write_bytes(b"model")
    output_root = tmp_path / "output"
    output_root.mkdir()
    input_paths = ArtifactConcretePaths(
        root_path=input_model.parent,
        entrypoint_path=input_model,
    )
    output_paths = ArtifactConcretePaths(root_path=output_root)
    model_graph = MagicMock(spec=ModelGraph)
    model_graph.extract_incoming_outgoing_tensors_of_sub_model.return_value = (
        {"input_ids", "attention_mask"},
        {"logits"},
    )

    def create_output(*, output_path, **_kwargs) -> None:
        output_path.write_bytes(b"split-model")

    with patch(
        "distributed_inference.adapters.outbound.model_splitter.onnx."
        "onnx_model_splitter.extract_model",
        side_effect=create_output,
    ) as extract_model:
        OnnxModelSplitter().split_model(
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
def test_split_model_rejects_empty_components_before_touching_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        OnnxModelSplitter().split_model(
            MagicMock(spec=ModelGraph),
            [],
            ArtifactConcretePaths(root_path=tmp_path),
            ArtifactConcretePaths(root_path=tmp_path),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ((set(), {"output"}), "no inputs"),
        (({"input"}, set()), "no outputs"),
    ],
)
def test_split_model_rejects_components_without_complete_boundaries(
    tmp_path,
    boundaries,
    message: str,
) -> None:
    input_model = tmp_path / "model.onnx"
    input_model.write_bytes(b"model")
    output_root = tmp_path / "output"
    output_root.mkdir()
    graph = MagicMock(spec=ModelGraph)
    graph.extract_incoming_outgoing_tensors_of_sub_model.return_value = boundaries

    with pytest.raises(ValueError, match=message):
        OnnxModelSplitter().split_model(
            graph,
            ["layer"],
            ArtifactConcretePaths(
                root_path=tmp_path,
                entrypoint_path=input_model,
            ),
            ArtifactConcretePaths(root_path=output_root),
        )
