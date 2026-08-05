from pathlib import Path
from unittest.mock import patch

import pytest

from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.model_manager.domain.model import ModelType
from distributed_inference.model_manager.domain.model_version import (
    ArchitectureInfo,
    BERTArchitectureInfo,
    CNNArchitectureInfo,
    ModelVersionInfo,
    VITArchitectureInfo,
)
from distributed_inference.model_optimizer.adapters.outbound.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from distributed_inference.model_optimizer.domain.optimization_level import (
    OptimizationLevel,
)
from test.support.artifact_materializer.materialized_artifact_test_utils import (
    build_test_materialized_artifact,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model_info,
    build_model_version_info,
)


def _model_version_info(model_type: ModelType) -> ModelVersionInfo:
    architecture_info: dict[ModelType, ArchitectureInfo] = {
        ModelType.CNN: CNNArchitectureInfo(),
        ModelType.VIT: VITArchitectureInfo(num_heads=12, hidden_size=768),
        ModelType.BERT: BERTArchitectureInfo(num_heads=12, hidden_size=768),
    }
    return build_model_version_info(architecture_info=architecture_info[model_type])


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
    output_paths = ArtifactWorkspace(root_path=output_root)

    def create_output(*, output_path: Path, **_kwargs: object) -> None:
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
            build_test_materialized_artifact(input_model, root_path=tmp_path),
            output_paths,
            build_model_info(model_type=model_type),
            _model_version_info(model_type),
            level,
        )

    selected = standard if method_name.endswith("standard") else transformer
    rejected = transformer if selected is standard else standard
    selected.assert_called_once()
    rejected.assert_not_called()
    assert selected.call_args.kwargs["opt_level"] is level
    assert selected.call_args.kwargs["output_path"].is_file()
    assert output_paths.entrypoint_path == selected.call_args.kwargs["output_path"]


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
