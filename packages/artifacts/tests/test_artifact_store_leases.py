import asyncio
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock

import pytest

from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.contracts.artifact_manifest import ArtifactFileInfo, ArtifactManifest
from artifacts.storage.artifact_store import (
    ArtifactCheckError,
    ArtifactDownloadError,
    ArtifactUploadError,
)
from artifacts.storage.cached_artifact_store import CachedArtifactStore
from artifacts.storage.local_artifact_store import LocalArtifactStore
from shared.artifact.artifact_ref import ArtifactRef


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
    await authoritative_store.upload_artifact(artifact_ref, source)

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
    await store.upload_artifact(artifact_ref, first_source)

    async with store.download_artifact(artifact_ref) as bundle:
        update_task = asyncio.create_task(
            store.upload_artifact(artifact_ref, second_source)
        )
        await asyncio.sleep(0)
        assert not update_task.done()
        assert bundle.entrypoint_path.read_bytes() == b"first"

    await update_task
    async with store.download_artifact(artifact_ref) as bundle:
        assert bundle.entrypoint_path.read_bytes() == b"second"


@pytest.mark.asyncio
async def test_local_store_reports_invalid_manifest(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    ref = ArtifactRef("model")
    assert not await store.check_artifact_existence(ref)
    await store.upload_artifact(ref, build_bundle(tmp_path / "source", b"model"))
    async with store.download_artifact(ref) as bundle:
        manifest_path = bundle.root_path / "manifest.json"
    manifest_path.write_text("invalid")

    with pytest.raises(ArtifactCheckError) as check:
        await store.check_artifact_existence(ref)
    assert isinstance(check.value.__cause__, ValueError)
    with pytest.raises(ArtifactDownloadError) as download:
        async with store.download_artifact(ref):
            pytest.fail("Invalid manifest must not yield a bundle")
    assert isinstance(download.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_local_store_reports_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalArtifactStore(tmp_path / "store")
    failure = PermissionError("denied")
    monkeypatch.setattr("aiofiles.os.stat", AsyncMock(side_effect=failure))
    with pytest.raises(ArtifactCheckError) as caught:
        await store.check_artifact_existence(ArtifactRef("model"))
    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_hit", [False, True])
async def test_cached_store_preserves_caller_error(
    tmp_path: Path,
    cache_hit: bool,
) -> None:
    authoritative = LocalArtifactStore(tmp_path / "authoritative")
    cache = LocalArtifactStore(tmp_path / "cache")
    store = CachedArtifactStore(authoritative, cache)
    ref = ArtifactRef("model")
    source = build_bundle(tmp_path / "source", b"model")
    await authoritative.upload_artifact(ref, source)
    if cache_hit:
        await cache.upload_artifact(ref, source)
    failure = ArtifactUploadError("caller failure")
    with pytest.raises(ArtifactUploadError) as caught:
        async with store.download_artifact(ref):
            raise failure
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["check", "upload"])
async def test_cached_download_fails_when_cache_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    authoritative = LocalArtifactStore(tmp_path / "authoritative")
    cache = LocalArtifactStore(tmp_path / "cache")
    store = CachedArtifactStore(authoritative, cache)
    ref = ArtifactRef("model")
    await authoritative.upload_artifact(
        ref, build_bundle(tmp_path / "source", b"model")
    )
    if operation == "check":
        failure = ArtifactCheckError("cache failure")
        method = "check_artifact_existence"
    else:
        failure = ArtifactUploadError("cache failure")
        method = "upload_artifact"
    monkeypatch.setattr(cache, method, AsyncMock(side_effect=failure))
    with pytest.raises(ArtifactDownloadError) as caught:
        async with store.download_artifact(ref):
            pytest.fail("Failed cache must not yield a bundle")
    assert caught.value.__cause__ is failure
