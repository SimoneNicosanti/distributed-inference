from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncGenerator, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from distributed_inference.artifact_materializer.adapters.outbound.local.local_artifact_materializer import (
    LocalArtifactMaterializer,
)
from distributed_inference.artifact_store.adapters.outbound.local.local_artifact_store import (
    LocalArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ModelVersionArtifactKey,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)
from distributed_inference.domain.identifiers import (
    UserId,
)
from distributed_inference.model_manager.domain.model import ModelId
from distributed_inference.model_manager.domain.model_version import ModelVersionId


@pytest.mark.unit
@pytest.mark.asyncio
async def test_materializer_delegates_paths_to_local_store(tmp_path: Path) -> None:
    raw_store = MagicMock(spec=LocalArtifactStore)
    store = cast(LocalArtifactStore, raw_store)
    key = ModelVersionArtifactKey(
        id=ModelVersionId(
            model_id=ModelId(
                owner_id=UserId(id=uuid4()),
                model_name="resnet50",
            ),
            version_tag="v1",
        )
    )
    root_path = tmp_path / "bundle"
    root_path.mkdir()
    entrypoint_path = root_path / "model.onnx"
    entrypoint_path.write_bytes(b"model")
    entrypoint_ppp = PurePosixPath("model.onnx")
    manifest = ArtifactManifest(
        entrypoint_ppp=entrypoint_ppp,
        files_info=(ArtifactFileInfo(file_ppp=entrypoint_ppp),),
    )

    @asynccontextmanager
    async def stored_paths() -> AsyncGenerator[tuple[ArtifactManifest, Path, Path]]:
        yield manifest, root_path, entrypoint_path

    raw_store.get_artifact_manifest_root_path_entry_path.return_value = stored_paths()
    materializer = LocalArtifactMaterializer(store)

    async with materializer.materialize_artifact(key) as result:
        assert result.manifest == manifest
        assert result.root_path == root_path
        assert result.entrypoint_path == entrypoint_path

    raw_store.get_artifact_manifest_root_path_entry_path.assert_called_once_with(key)
