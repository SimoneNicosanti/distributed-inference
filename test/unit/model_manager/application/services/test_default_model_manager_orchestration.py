from contextlib import nullcontext
from io import BytesIO
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import (
    ModelGraph,
    ModelInfo,
    ModelType,
    TaskType,
)
from distributed_inference.model_artifact.application.ports.outbound.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.model_artifact.application.ports.outbound.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
)
from distributed_inference.model_manager.application.services.default_model_manager import (
    DefaultModelManager,
)
from distributed_inference.model_materializer.application.ports.outbound.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.model_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from distributed_inference.model_profile.application.ports.inbound.model_profiler import (
    ModelProfiler,
)
from distributed_inference.model_splitter.application.ports.outbound.model_splitter import (
    ModelSplitter,
)


def _ids() -> tuple[ModelId, ModelVersionId, SubModelId]:
    model_id = ModelId(
        user_id=UserId(user_id=uuid4()),
        model_name="vision-model",
    )
    version_id = ModelVersionId(model_id=model_id, version_number=1)
    sub_model_id = SubModelId(
        model_version_id=version_id,
        layers=("encoder.0", "encoder.1"),
    )
    return model_id, version_id, sub_model_id


def _model_info() -> ModelInfo:
    return ModelInfo(
        name="vision-model",
        accuracy=0.9,
        task=TaskType.CLASSIFICATION,
        type=ModelType.VIT,
        dynamic_shapes={},
    )


def _bundle() -> ArtifactBundle:
    entrypoint = PurePosixPath("model.onnx")
    return ArtifactBundle(
        manifest=ArtifactManifest(
            rel_entrypoint_path=entrypoint,
            rel_file_paths=(entrypoint,),
        ),
        artifact_files=(
            ArtifactFile(
                rel_path=entrypoint,
                content=BytesIO(b"model"),
            ),
        ),
    )


@pytest.mark.unit
def test_put_model_version_stores_profiles_and_registers_graph(tmp_path: Path) -> None:
    profiler = MagicMock(spec=ModelProfiler)
    version_store = MagicMock(spec=ModelVersionArtifactStore)
    metadata_store = MagicMock(spec=ModelMetadataStore)
    materializer = MagicMock(spec=ModelVersionMaterializer)
    manager = DefaultModelManager(
        profiler,
        version_store,
        MagicMock(spec=SubModelArtifactStore),
        metadata_store,
        materializer,
        MagicMock(spec=ModelSplitter),
    )
    model_id, version_id, _ = _ids()
    model_info = _model_info()
    bundle = _bundle()
    (tmp_path / "model.onnx").write_bytes(b"model")
    paths = MaterializedArtifact(
        root_path=tmp_path,
        entrypoint_path=tmp_path / "model.onnx",
    )
    graph = MagicMock(spec=ModelGraph)
    metadata_store.register_model_version.return_value = version_id
    materializer.materialize_model_version.return_value = nullcontext(paths)
    profiler.profile_model.return_value = graph

    result = manager.put_model_version(model_id, model_info, bundle)

    assert result == version_id
    version_store.put_model_version.assert_called_once_with(version_id, bundle)
    profiler.profile_model.assert_called_once_with(paths, model_info)
    metadata_store.register_model_version_graph.assert_called_once_with(
        version_id, graph
    )


@pytest.mark.unit
def test_generate_sub_model_splits_and_stores_component(tmp_path: Path) -> None:
    sub_model_store = MagicMock(spec=SubModelArtifactStore)
    metadata_store = MagicMock(spec=ModelMetadataStore)
    materializer = MagicMock(spec=ModelVersionMaterializer)
    splitter = MagicMock(spec=ModelSplitter)
    manager = DefaultModelManager(
        MagicMock(spec=ModelProfiler),
        MagicMock(spec=ModelVersionArtifactStore),
        sub_model_store,
        metadata_store,
        materializer,
        splitter,
    )
    _, version_id, sub_model_id = _ids()
    layers = ["encoder.0", "encoder.1"]
    graph = MagicMock(spec=ModelGraph)
    (tmp_path / "model.onnx").write_bytes(b"model")
    model_paths = MaterializedArtifact(
        root_path=tmp_path,
        entrypoint_path=tmp_path / "model.onnx",
    )
    split_bundle = _bundle()
    metadata_store.get_model_graph.return_value = graph
    metadata_store.register_sub_model.return_value = sub_model_id
    materializer.materialize_model_version.return_value = nullcontext(model_paths)

    with patch(
        "distributed_inference.model_manager.application.services.default_model_manager."
        "artifact_bundle_builder.build_artifact_bundle_from_bundle_paths",
        return_value=nullcontext(split_bundle),
    ) as build_bundle:
        result = manager.generate_sub_model(version_id, layers)

    assert result == sub_model_id
    normalized_layers = tuple(layers)
    split_output_paths = splitter.split_model.call_args.args[3]
    metadata_store.register_sub_model.assert_called_once_with(
        version_id, normalized_layers
    )
    splitter.split_model.assert_called_once_with(
        graph, normalized_layers, model_paths, split_output_paths
    )
    build_bundle.assert_called_once_with(split_output_paths)
    sub_model_store.put_sub_model.assert_called_once_with(sub_model_id, split_bundle)


@pytest.mark.unit
def test_generate_sub_model_rejects_string_layers_before_dependencies() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    manager = DefaultModelManager(
        MagicMock(spec=ModelProfiler),
        MagicMock(spec=ModelVersionArtifactStore),
        MagicMock(spec=SubModelArtifactStore),
        metadata_store,
        MagicMock(spec=ModelVersionMaterializer),
        MagicMock(spec=ModelSplitter),
    )
    _, version_id, _ = _ids()

    with pytest.raises(ValueError, match="Layers must contain layer names"):
        manager.generate_sub_model(version_id, "encoder.0")

    metadata_store.get_model_graph.assert_not_called()
