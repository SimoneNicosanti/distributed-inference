from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from advertisement.http.http_capability_life_checker import DEFAULT_HTTP_READINESS_PATH
from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.contracts.artifact_manifest import ArtifactFileInfo, ArtifactManifest
from artifacts.storage.artifact_store import ArtifactStore
from artifacts.storage.local_artifact_store import LocalArtifactStore
from integration.service_instance.service_instance import CapabilityType
from integration.worker.deployment.http.service_plan_http_paths import (
    APPLY_SERVICE_PLAN_PATH,
    PREPARE_SERVICE_PLAN_PATH,
)
from integration.worker.worker_identity.http_worker_identity_query_paths import (
    WORKER_IDENTITY_PATH,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.service.service import ServiceId
from worker.application.worker_identity.default_worker_identity_query import (
    DefaultWorkerIdentityQuery,
)
from worker.composition import default_composition_root as composition_module
from worker.composition.composition_root import WorkerComposition
from worker.composition.default_composition_root import DefaultCompositionRoot
from worker.main import build_http_app, build_worker_instance


def test_worker_http_app_keeps_runtime_routes_without_deployment() -> None:
    worker_id = ServiceId(id="worker")
    composition = WorkerComposition(
        hop_contribution_receiver=Mock(),
        model_pass_result_retriever=Mock(),
        worker_identity_query=DefaultWorkerIdentityQuery(worker_id),
        failure_receiver=Mock(),
    )

    with TestClient(build_http_app(composition)) as client:
        assert client.get(DEFAULT_HTTP_READINESS_PATH).status_code == 204
        assert client.get(WORKER_IDENTITY_PATH).json() == {"service_id": "worker"}
        for path in (PREPARE_SERVICE_PLAN_PATH, APPLY_SERVICE_PLAN_PATH):
            assert client.put(path.format(version_number=1)).status_code == 404

    worker = build_worker_instance()
    assert not worker.get_capabilities_by_type(CapabilityType.PLAN_ACTUATION)


@asynccontextmanager
async def _context[T](value: T) -> AsyncGenerator[T]:
    yield value


@pytest.mark.asyncio
async def test_composition_starts_without_eager_plan_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = DefaultCompositionRoot()
    redis = AsyncMock()
    artifact_store = Mock(spec=ArtifactStore)
    create_executor = AsyncMock()
    monkeypatch.setattr(
        composition_module.OnnxPartitionExecutorFactory, "create", create_executor
    )
    monkeypatch.setattr(root, "build_redis_connection_context", lambda: _context(redis))
    monkeypatch.setattr(
        root, "build_advertising_manager_context", lambda *_: _context(None)
    )
    monkeypatch.setattr(
        root, "build_artifact_store_context", lambda: _context(artifact_store)
    )
    monkeypatch.setattr(root, "start_profiling_lifecycles", AsyncMock())
    worker = build_worker_instance()

    async with root.compose(worker) as composition:
        assert (
            await composition.worker_identity_query.get_worker_identity()
            == worker.service_id
        )
        redis.hget.assert_not_awaited()
        artifact_store.download_artifact.assert_not_called()
        create_executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_composition_reuses_artifacts_across_store_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    (source_path / "model.onnx").write_bytes(b"model")
    entrypoint = PurePosixPath("model.onnx")
    bundle = ArtifactBundle(
        manifest=ArtifactManifest(
            entrypoint_ppp=entrypoint,
            files_info=(ArtifactFileInfo(file_ppp=entrypoint),),
        ),
        root_path=source_path,
    )
    authoritative = LocalArtifactStore(tmp_path / "authoritative")
    artifact_ref = ArtifactRef(value="model")
    await authoritative.upload_artifact(artifact_ref, bundle)
    download = Mock(wraps=authoritative.download_artifact)
    monkeypatch.setattr(authoritative, "download_artifact", download)
    client = Mock()
    monkeypatch.setattr(composition_module.boto3, "client", Mock(return_value=client))
    monkeypatch.setattr(
        composition_module, "S3ArtifactStore", Mock(return_value=authoritative)
    )
    cache_path = tmp_path / "cache"
    monkeypatch.setenv("WORKER_ARTIFACT_CACHE_DIR", str(cache_path))
    root = DefaultCompositionRoot()

    async with (
        root.build_artifact_store_context() as store,
        store.download_artifact(artifact_ref) as cached,
    ):
        cached_path = cached.root_path
        assert cached_path.is_relative_to(cache_path)
        assert cached.entrypoint_path.read_bytes() == b"model"

    download.side_effect = AssertionError(
        "Cached artifact must not be downloaded again"
    )
    async with (
        root.build_artifact_store_context() as store,
        store.download_artifact(artifact_ref) as cached,
    ):
        assert cached.root_path == cached_path
        assert cached.entrypoint_path.read_bytes() == b"model"

    assert download.call_count == 1
    assert client.close.call_count == 2


@pytest.mark.asyncio
async def test_artifact_context_closes_client_when_cache_initialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "cache-file"
    cache_path.write_text("not a directory")
    monkeypatch.setenv("WORKER_ARTIFACT_CACHE_DIR", str(cache_path))
    client = Mock()
    monkeypatch.setattr(composition_module.boto3, "client", Mock(return_value=client))

    with pytest.raises(OSError):
        async with DefaultCompositionRoot().build_artifact_store_context():
            pytest.fail("Cache initialization must fail")

    client.close.assert_called_once()
