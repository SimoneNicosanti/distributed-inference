from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest
from fastapi import APIRouter, UploadFile
from fastapi.routing import APIRoute

from distributed_inference.artifact_store.domain.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactFileInfo,
    ArtifactManifest,
)
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)
from distributed_inference.model_manager.adapters.inbound.http.model_manager_http_router import (
    build_model_manager_router,
)
from distributed_inference.model_manager.adapters.inbound.http.model_manager_http_schema import (
    DownloadSubModelRequest,
    GenerateSubModelRequest,
    GetProfiledModelVersionRequest,
    RegisterModelRequest,
)
from distributed_inference.model_manager.application.ports.inbound.model_manager import (
    ModelManager,
)
from distributed_inference.model_manager.domain.model_version import ModelVersion
from test.support.artifact_store.artifact_bundle_test_utils import (
    build_test_bundle,
    read_bundle_content,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model,
    build_model_version,
    build_model_version_id,
    build_profiled_model_version,
    build_sub_model,
    build_sub_model_id,
)


@asynccontextmanager
async def _async_context[T](value: T) -> AsyncGenerator[T]:
    yield value


def _endpoint(router: APIRouter, path: str, method: str) -> Callable[..., Any]:
    for route in router.routes:
        if (
            isinstance(route, APIRoute)
            and route.path == path
            and route.methods is not None
            and method in route.methods
        ):
            return route.endpoint
    raise AssertionError(f"Missing {method} route {path}")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_and_generate_routes_delegate_typed_requests() -> None:
    manager = MagicMock(spec=ModelManager)
    router = build_model_manager_router(manager)
    model = build_model()
    sub_model = build_sub_model()
    manager.register_model.return_value = model.model_id
    manager.generate_sub_model.return_value = sub_model

    register_response = await _endpoint(router, "/model-manager/models", "POST")(
        RegisterModelRequest(model=model)
    )
    generate_response = await _endpoint(router, "/model-manager/sub-models", "POST")(
        GenerateSubModelRequest(
            model_version_id=sub_model.sub_model_id.model_version_id,
            layers=["encoder.1", "encoder.0"],
        )
    )

    assert register_response.model_id == model.model_id
    manager.register_model.assert_awaited_once_with(model=model)
    assert generate_response.sub_model == sub_model
    manager.generate_sub_model.assert_awaited_once_with(
        model_version_id=sub_model.sub_model_id.model_version_id,
        layers=["encoder.1", "encoder.0"],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_route_decompresses_bundle_before_delegating() -> None:
    manager = MagicMock(spec=ModelManager)
    router = build_model_manager_router(manager)
    model_version = build_model_version()
    version_id = model_version.model_version_id
    model_path = PurePosixPath("model.onnx")
    manifest = ArtifactManifest(
        entrypoint_ppp=model_path,
        files_info=(ArtifactFileInfo(file_ppp=model_path),),
    )
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        archive.writestr(MANIFEST_FILE_NAME, manifest.model_dump_json())
        archive.writestr("model.onnx", b"onnx-model")
    archive_bytes.seek(0)
    upload = UploadFile(filename="model.zip", file=archive_bytes)
    received_contents: dict[PurePosixPath, bytes] = {}

    async def upload_model_version(
        *,
        model_version: ModelVersion,
        bundle: ReadableArtifactBundle,
    ) -> object:
        received_contents.update(await read_bundle_content(bundle))
        return version_id

    manager.upload_model_version.side_effect = upload_model_version

    response = await _endpoint(router, "/model-manager/model-versions", "POST")(
        model_version.model_dump_json(),
        upload,
    )

    assert response.model_version_id == version_id
    assert received_contents == {model_path: b"onnx-model"}
    assert (
        manager.upload_model_version.call_args.kwargs["model_version"] == model_version
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_route_streams_the_compressed_sub_model_bundle(
    tmp_path: Path,
) -> None:
    manager = MagicMock(spec=ModelManager)
    router = build_model_manager_router(manager)
    sub_model_id = build_sub_model_id()
    bundle = build_test_bundle(
        tmp_path / "sub-model",
        files={PurePosixPath("model.onnx"): b"sub-model-content"},
    )
    manager.download_sub_model.return_value = _async_context(bundle)

    response = await _endpoint(router, "/model-manager/sub-models/download", "POST")(
        DownloadSubModelRequest(sub_model_id=sub_model_id)
    )
    streamed = b"".join([chunk async for chunk in response.body_iterator])

    manager.download_sub_model.assert_called_once_with(sub_model_id)
    with ZipFile(BytesIO(streamed)) as archive:
        assert archive.read("model.onnx") == b"sub-model-content"
        assert (
            ArtifactManifest.model_validate_json(archive.read(MANIFEST_FILE_NAME))
            == bundle.manifest
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graph_route_returns_the_profiled_model_version() -> None:
    manager = MagicMock(spec=ModelManager)
    router = build_model_manager_router(manager)
    version_id = build_model_version_id()
    profiled_model_version = build_profiled_model_version(
        model_version=build_model_version(model_version_id=version_id)
    )
    manager.get_profiled_model_version.return_value = profiled_model_version

    response = await _endpoint(router, "/model-manager/model-versions/graph", "GET")(
        GetProfiledModelVersionRequest(model_version_id=version_id)
    )

    assert response.profiled_model_version is profiled_model_version
    manager.get_profiled_model_version.assert_awaited_once_with(version_id)
