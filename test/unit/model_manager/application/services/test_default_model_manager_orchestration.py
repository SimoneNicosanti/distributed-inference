from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ModelVersionArtifactKey,
    SubModelArtifactKey,
)
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
from distributed_inference.model_manager.application.services.default_model_manager import (
    DefaultModelManager,
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
from test.support.artifact_materializer.materialized_artifact_test_utils import (
    build_test_materialized_artifact,
)
from test.support.artifact_store.artifact_bundle_test_utils import build_test_bundle


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


@asynccontextmanager
async def _async_context[T](value: T) -> AsyncGenerator[T]:
    yield value


def _manager(
    *,
    profiler: ModelProfiler | None = None,
    artifact_store: ArtifactStore | None = None,
    metadata_store: ModelMetadataStore | None = None,
    materializer: ArtifactMaterializer | None = None,
    splitter: ModelSplitter | None = None,
) -> DefaultModelManager:
    return DefaultModelManager(
        profiler or MagicMock(spec=ModelProfiler),
        artifact_store or MagicMock(spec=ArtifactStore),
        metadata_store or MagicMock(spec=ModelMetadataStore),
        materializer or MagicMock(spec=ArtifactMaterializer),
        splitter or MagicMock(spec=ModelSplitter),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_put_model_version_stores_profiles_and_registers_graph(
    tmp_path: Path,
) -> None:
    profiler = MagicMock(spec=ModelProfiler)
    artifact_store = MagicMock(spec=ArtifactStore)
    metadata_store = MagicMock(spec=ModelMetadataStore)
    materializer = MagicMock(spec=ArtifactMaterializer)
    manager = _manager(
        profiler=profiler,
        artifact_store=artifact_store,
        metadata_store=metadata_store,
        materializer=materializer,
    )
    model_id, version_id, _ = _ids()
    model_info = _model_info()
    bundle = build_test_bundle(tmp_path / "input")
    entrypoint = tmp_path / "materialized" / "model.onnx"
    entrypoint.parent.mkdir()
    entrypoint.write_bytes(b"model")
    materialized = build_test_materialized_artifact(entrypoint)
    graph = MagicMock(spec=ModelGraph)
    metadata_store.register_model_version.return_value = version_id
    materializer.materialize_artifact.return_value = _async_context(materialized)
    profiler.profile_model.return_value = graph

    result = await manager.put_model_version(model_id, model_info, bundle)

    assert result == version_id
    metadata_store.register_model_version.assert_awaited_once_with(
        model_id, model_info
    )
    artifact_store.put_artifact.assert_awaited_once_with(
        ModelVersionArtifactKey(id=version_id),
        bundle,
    )
    materializer.materialize_artifact.assert_called_once_with(
        ModelVersionArtifactKey(id=version_id)
    )
    profiler.profile_model.assert_awaited_once_with(materialized, model_info)
    metadata_store.register_model_version_graph.assert_awaited_once_with(
        version_id, graph
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_sub_model_splits_and_stores_component(
    tmp_path: Path,
) -> None:
    artifact_store = MagicMock(spec=ArtifactStore)
    metadata_store = MagicMock(spec=ModelMetadataStore)
    materializer = MagicMock(spec=ArtifactMaterializer)
    splitter = MagicMock(spec=ModelSplitter)
    manager = _manager(
        artifact_store=artifact_store,
        metadata_store=metadata_store,
        materializer=materializer,
        splitter=splitter,
    )
    _, version_id, sub_model_id = _ids()
    layers = ["encoder.0", "encoder.1"]
    graph = MagicMock(spec=ModelGraph)
    input_entrypoint = tmp_path / "input" / "model.onnx"
    input_entrypoint.parent.mkdir()
    input_entrypoint.write_bytes(b"model")
    materialized = build_test_materialized_artifact(input_entrypoint)
    split_bundle = build_test_bundle(tmp_path / "split-result")
    metadata_store.get_model_graph.return_value = graph
    metadata_store.register_sub_model.return_value = sub_model_id
    materializer.materialize_artifact.return_value = _async_context(materialized)

    with patch(
        "distributed_inference.model_manager.application.services."
        "default_model_manager."
        "build_local_artifact_bundle_from_artifact_workspace",
        return_value=split_bundle,
    ) as build_bundle:
        result = await manager.generate_sub_model(version_id, layers)

    assert result == sub_model_id
    normalized_layers = tuple(layers)
    metadata_store.register_sub_model.assert_awaited_once_with(
        version_id, normalized_layers
    )
    materializer.materialize_artifact.assert_called_once_with(
        ModelVersionArtifactKey(id=version_id)
    )
    split_output = splitter.split_model.call_args.args[3]
    assert isinstance(split_output, ArtifactWorkspace)
    splitter.split_model.assert_awaited_once_with(
        graph,
        normalized_layers,
        materialized,
        split_output,
    )
    build_bundle.assert_called_once_with(split_output)
    artifact_store.put_artifact.assert_awaited_once_with(
        SubModelArtifactKey(id=sub_model_id),
        split_bundle,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_sub_model_rejects_string_layers_before_dependencies() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    manager = _manager(metadata_store=metadata_store)
    _, version_id, _ = _ids()

    with pytest.raises(ValueError, match="Layers must contain layer names"):
        await manager.generate_sub_model(version_id, "encoder.0")

    metadata_store.get_model_graph.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_sub_model_requires_ready_graph() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    metadata_store.get_model_graph.return_value = None
    manager = _manager(metadata_store=metadata_store)
    _, version_id, _ = _ids()

    with pytest.raises(ValueError, match="still not ready"):
        await manager.generate_sub_model(version_id, ["encoder.0"])

    metadata_store.register_sub_model.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_exists", "artifact_exists", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
async def test_model_version_existence_requires_metadata_and_artifact(
    metadata_exists: bool,
    artifact_exists: bool,
    expected: bool,
) -> None:
    artifact_store = MagicMock(spec=ArtifactStore)
    metadata_store = MagicMock(spec=ModelMetadataStore)
    manager = _manager(
        artifact_store=artifact_store,
        metadata_store=metadata_store,
    )
    _, version_id, _ = _ids()
    metadata_store.check_model_version_existence.return_value = metadata_exists
    artifact_store.check_artifact_existence.return_value = artifact_exists

    result = await manager.check_model_version_existence(version_id)

    assert result is expected
    artifact_store.check_artifact_existence.assert_awaited_once_with(
        ModelVersionArtifactKey(id=version_id)
    )


@pytest.mark.unit
def test_get_sub_model_opens_sub_model_artifact() -> None:
    artifact_store = MagicMock(spec=ArtifactStore)
    manager = _manager(artifact_store=artifact_store)
    _, _, sub_model_id = _ids()
    expected_context = MagicMock()
    artifact_store.open_artifact.return_value = expected_context

    result = manager.get_sub_model(sub_model_id)

    assert result is expected_context
    artifact_store.open_artifact.assert_called_once_with(
        SubModelArtifactKey(id=sub_model_id)
    )
