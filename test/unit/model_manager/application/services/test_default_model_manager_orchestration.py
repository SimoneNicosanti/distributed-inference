from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from distributed_inference.model_manager.application.services.default_model_manager import (
    DefaultModelManager,
)
from distributed_inference.model_manager.domain.sub_model import SubModelId
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from distributed_inference.model_profiler.application.ports.inbound.model_profiler import (
    ModelProfiler,
)
from distributed_inference.model_splitter.application.ports.outbound.model_splitter import (
    ModelSplitter,
)
from test.support.artifact_materializer.materialized_artifact_test_utils import (
    build_test_materialized_artifact,
)
from test.support.artifact_store.artifact_bundle_test_utils import build_test_bundle
from test.support.model_manager.model_domain_test_utils import (
    build_model,
    build_model_version,
    build_model_version_id,
    build_profiled_model_version,
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
async def test_register_model_delegates_to_the_metadata_store() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    manager = _manager(metadata_store=metadata_store)
    model = build_model()
    metadata_store.register_model.return_value = model.model_id

    result = await manager.register_model(model)

    assert result == model.model_id
    metadata_store.register_model.assert_awaited_once_with(model)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_model_version_stores_artifact_then_registers_the_profile(
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
    model = build_model()
    model_version = build_model_version(
        model_version_id=build_model_version_id(model_id=model.model_id)
    )
    version_id = model_version.model_version_id
    profiled_model_version = build_profiled_model_version(model_version=model_version)
    bundle = build_test_bundle(tmp_path / "input")
    entrypoint = tmp_path / "materialized" / "model.onnx"
    entrypoint.parent.mkdir()
    entrypoint.write_bytes(b"model")
    materialized = build_test_materialized_artifact(entrypoint)
    metadata_store.register_model_version.return_value = version_id
    metadata_store.get_model.return_value = model
    materializer.materialize_artifact.return_value = _async_context(materialized)
    profiler.profile_model_version.return_value = profiled_model_version

    result = await manager.upload_model_version(model_version, bundle)

    assert result == version_id
    metadata_store.register_model_version.assert_awaited_once_with(model_version)
    artifact_store.put_artifact.assert_awaited_once_with(
        ModelVersionArtifactKey(id=version_id),
        bundle,
    )
    materializer.materialize_artifact.assert_called_once_with(
        ModelVersionArtifactKey(id=version_id)
    )
    profiler.profile_model_version.assert_awaited_once_with(
        materialized,
        model.model_info,
        model_version,
    )
    metadata_store.register_profiled_model_version.assert_awaited_once_with(
        profiled_model_version
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_sub_model_splits_the_model_and_stores_the_sub_model(
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
    model_version = build_model_version()
    version_id = model_version.model_version_id
    profiled_model_version = build_profiled_model_version(model_version=model_version)
    layers = ["encoder.1", "encoder.0"]
    expected_sub_model_id = SubModelId(
        model_version_id=version_id,
        layers=tuple(layers),
    )
    input_entrypoint = tmp_path / "input" / "model.onnx"
    input_entrypoint.parent.mkdir()
    input_entrypoint.write_bytes(b"model")
    materialized = build_test_materialized_artifact(input_entrypoint)
    split_bundle = build_test_bundle(tmp_path / "split-result")
    metadata_store.get_profiled_model_version.return_value = profiled_model_version
    metadata_store.register_sub_model.return_value = expected_sub_model_id
    materializer.materialize_artifact.return_value = _async_context(materialized)

    with patch(
        "distributed_inference.model_manager.application.services."
        "default_model_manager."
        "build_local_artifact_bundle_from_artifact_workspace",
        return_value=split_bundle,
    ) as build_bundle:
        result = await manager.generate_sub_model(version_id, layers)

    assert result.sub_model_id == expected_sub_model_id
    metadata_store.register_sub_model.assert_awaited_once()
    assert (
        metadata_store.register_sub_model.call_args.args[0].sub_model_id
        == expected_sub_model_id
    )
    materializer.materialize_artifact.assert_called_once_with(
        ModelVersionArtifactKey(id=version_id)
    )
    splitter.split_model.assert_awaited_once()
    split_call_args = splitter.split_model.call_args.args
    assert split_call_args[0] is profiled_model_version.model_version_graph
    assert split_call_args[1] == tuple(layers)
    assert split_call_args[2] is materialized
    split_output = split_call_args[3]
    assert isinstance(split_output, ArtifactWorkspace)
    build_bundle.assert_called_once_with(split_output)
    artifact_store.put_artifact.assert_awaited_once_with(
        SubModelArtifactKey(id=expected_sub_model_id),
        split_bundle,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_sub_model_rejects_string_layers_before_dependencies() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    manager = _manager(metadata_store=metadata_store)

    with pytest.raises(ValueError, match="Layers must contain layer names"):
        await manager.generate_sub_model(build_model_version_id(), "encoder.0")

    metadata_store.get_profiled_model_version.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_sub_model_requires_a_profiled_model_version() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    metadata_store.get_profiled_model_version.return_value = None
    manager = _manager(metadata_store=metadata_store)

    with pytest.raises(ValueError, match="still not ready"):
        await manager.generate_sub_model(build_model_version_id(), ["encoder.0"])

    metadata_store.register_sub_model.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_profiled_model_version_requires_a_profiled_model_version() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    metadata_store.get_profiled_model_version.return_value = None
    manager = _manager(metadata_store=metadata_store)

    with pytest.raises(ValueError, match="still not ready"):
        await manager.get_profiled_model_version(build_model_version_id())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_profiled_model_version_returns_the_stored_profile() -> None:
    metadata_store = MagicMock(spec=ModelMetadataStore)
    manager = _manager(metadata_store=metadata_store)
    profiled_model_version = build_profiled_model_version()
    version_id = profiled_model_version.model_version_id
    metadata_store.get_profiled_model_version.return_value = profiled_model_version

    result = await manager.get_profiled_model_version(version_id)

    assert result is profiled_model_version
    metadata_store.get_profiled_model_version.assert_awaited_once_with(version_id)


@pytest.mark.unit
def test_download_sub_model_opens_the_sub_model_artifact() -> None:
    artifact_store = MagicMock(spec=ArtifactStore)
    manager = _manager(artifact_store=artifact_store)
    sub_model_id = SubModelId(
        model_version_id=build_model_version_id(),
        layers=("encoder.0", "encoder.1"),
    )
    expected_context = MagicMock()
    artifact_store.open_artifact.return_value = expected_context

    result = manager.download_sub_model(sub_model_id)

    assert result is expected_context
    artifact_store.open_artifact.assert_called_once_with(
        SubModelArtifactKey(id=sub_model_id)
    )
