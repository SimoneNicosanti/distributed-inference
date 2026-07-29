from pathlib import Path
from unittest.mock import patch

import pytest

from distributed_inference.domain.model_graph_info import (
    ModelInfo,
    ModelType,
    TaskType,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.model_optimize.adapters.outbound.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from distributed_inference.model_optimize.domain.optimization_level import (
    OptimizationLevel,
)


def _model_info(model_type: ModelType) -> ModelInfo:
    return ModelInfo(
        name="test-model",
        accuracy=0.9,
        task=TaskType.CLASSIFICATION,
        type=model_type,
        dynamic_shapes={},
        num_heads=12,
        hidden_size=768,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("level", "model_type", "method_name"),
    [
        (OptimizationLevel.NONE, ModelType.CNN, "_optimize_with_ort_standard"),
        (OptimizationLevel.BASIC, ModelType.VIT, "_optimize_with_ort_standard"),
        (OptimizationLevel.EXTENDED, ModelType.CNN, "_optimize_with_ort_standard"),
        (
            OptimizationLevel.EXTENDED,
            ModelType.BERT,
            "_optimize_with_ort_transformer",
        ),
    ],
)
def test_optimize_model_dispatches_to_expected_backend(
    tmp_path: Path,
    level: OptimizationLevel,
    model_type: ModelType,
    method_name: str,
) -> None:
    input_model = tmp_path / "input.onnx"
    input_model.write_bytes(b"model")
    output_root = tmp_path / "optimized"
    output_root.mkdir()
    optimizer = OnnxModelOptimizer()

    def create_output(*, output_path, **_kwargs) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"optimized")

    with (
        patch.object(
            optimizer,
            "_optimize_with_ort_standard",
            side_effect=create_output,
        ) as standard,
        patch.object(
            optimizer,
            "_optimize_with_ort_transformer",
            side_effect=create_output,
        ) as transformer,
    ):
        optimizer.optimize_model(
            ArtifactConcretePaths(
                root_path=tmp_path,
                entrypoint_path=input_model,
            ),
            ArtifactConcretePaths(root_path=output_root),
            _model_info(model_type),
            level,
        )

    selected = standard if method_name.endswith("standard") else transformer
    rejected = transformer if selected is standard else standard
    selected.assert_called_once()
    rejected.assert_not_called()
    assert selected.call_args.kwargs["opt_level"] is level
    assert selected.call_args.kwargs["output_path"].is_file()


@pytest.mark.unit
def test_optimizer_maps_public_levels_and_transformer_types() -> None:
    optimizer = OnnxModelOptimizer()

    assert optimizer._get_ort_transformer_model_type(ModelType.BERT) == "bert"
    assert optimizer._get_ort_transformer_model_type(ModelType.VIT) == "vit"

    with pytest.raises(ValueError, match="not a supported transformer"):
        optimizer._get_ort_transformer_model_type(ModelType.CNN)

    assert (
        optimizer._get_ort_opt_level(OptimizationLevel.NONE).name == "ORT_DISABLE_ALL"
    )
    assert (
        optimizer._get_ort_opt_level(OptimizationLevel.BASIC).name == "ORT_ENABLE_BASIC"
    )
    assert (
        optimizer._get_ort_opt_level(OptimizationLevel.EXTENDED).name
        == "ORT_ENABLE_EXTENDED"
    )
