import asyncio
from pathlib import Path, PurePosixPath

import pytest

from artifacts.contracts.artifact_manifest import ArtifactFileInfo, ArtifactManifest
from artifacts.contracts.artifact_ref import ArtifactRef
from artifacts.storage.cached_artifact_store import CachedArtifactStore
from artifacts.storage.local_artifact_store import LocalArtifactStore
from artifacts.workspace.artifact_bundle import ArtifactBundle


def build_bundle(root_path: Path, content: bytes) -> ArtifactBundle:
    root_path.mkdir(parents=True)
    entrypoint_path = root_path / "model.onnx"
    entrypoint_path.write_bytes(content)
    entrypoint_ppp = PurePosixPath(entrypoint_path.name)
    return ArtifactBundle(
        manifest=ArtifactManifest(
            entrypoint_ppp=entrypoint_ppp,
            files_info=(ArtifactFileInfo(file_ppp=entrypoint_ppp),),
        ),
        root_path=root_path,
    )


@pytest.mark.asyncio
async def test_cached_store_yields_cached_path_without_copy(tmp_path: Path) -> None:
    authoritative_store = LocalArtifactStore(tmp_path / "authoritative")
    cache_store = LocalArtifactStore(tmp_path / "cache")
    store = CachedArtifactStore(authoritative_store, cache_store)
    artifact_ref = ArtifactRef(value="model")
    source = build_bundle(tmp_path / "source", b"model")
    await authoritative_store.put_artifact(artifact_ref, source)

    async with store.download_artifact(artifact_ref) as first_bundle:
        cached_root = first_bundle.root_path
        assert cached_root.is_relative_to(tmp_path / "cache")
        assert first_bundle.entrypoint_path.read_bytes() == b"model"

    async with store.download_artifact(artifact_ref) as second_bundle:
        assert second_bundle.root_path == cached_root


@pytest.mark.asyncio
async def test_local_store_holds_read_lease_while_bundle_is_in_use(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    artifact_ref = ArtifactRef(value="model")
    first_source = build_bundle(tmp_path / "first", b"first")
    second_source = build_bundle(tmp_path / "second", b"second")
    await store.put_artifact(artifact_ref, first_source)

    async with store.download_artifact(artifact_ref) as bundle:
        update_task = asyncio.create_task(
            store.put_artifact(artifact_ref, second_source)
        )
        await asyncio.sleep(0)
        assert not update_task.done()
        assert bundle.entrypoint_path.read_bytes() == b"first"

    await update_task
    async with store.download_artifact(artifact_ref) as bundle:
        assert bundle.entrypoint_path.read_bytes() == b"second"
