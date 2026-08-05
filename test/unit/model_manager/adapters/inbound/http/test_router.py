from collections.abc import Callable
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4
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
from distributed_inference.domain.identifiers import (
    UserId,
)
from distributed_inference.model_manager.adapters.inbound.http.router import (
    build_model_manager_router,
)
from distributed_inference.model_manager.adapters.inbound.http.schema import (
    GenerateSubModelRequest,
    RegisterModelRequest,
)
from distributed_inference.model_manager.application.ports.inbound.model_manager import (
    ModelManager,
)
from distributed_inference.model_manager.domain.model import ModelId
from distributed_inference.model_manager.domain.model_version import ModelVersionId
from distributed_inference.model_manager.domain.model_version_graph import (
    ModelInfo,
    ModelType,
    TaskType,
)
from distributed_inference.model_manager.domain.sub_model import SubModelId
from test.support.artifact_store.artifact_bundle_test_utils import read_bundle_content


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


def _ids() -> tuple[UserId, ModelId, ModelVersionId]:
    user_id = UserId(id=uuid4())
    model_id = ModelId(owner_id=user_id, model_name="vision-model")
    version_id = ModelVersionId(model_id=model_id, version_tag=1)
    return user_id, model_id, version_id


def _model_info() -> ModelInfo:
    return ModelInfo(
        name="vision-model",
        accuracy=0.9,
        task=TaskType.CLASSIFICATION,
        type=ModelType.CNN,
        dynamic_shapes={},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_and_generate_routes_delegate_typed_requests() -> None:
    manager = MagicMock(spec=ModelManager)
    router = build_model_manager_router(manager)
    user_id, model_id, version_id = _ids()
    sub_model_id = SubModelId(
        model_version_id=version_id,
        layers=("encoder.0", "encoder.1"),
    )
    manager.register_model.return_value = model_id
    manager.generate_sub_model.return_value = sub_model_id

    register_response = await _endpoint(router, "/model-manager/models", "POST")(
        RegisterModelRequest(owner_id=user_id, model_name="vision-model")
    )
    generate_response = await _endpoint(router, "/model-manager/sub-models", "POST")(
        GenerateSubModelRequest(
            model_version_id=version_id,
            layers=["encoder.1", "encoder.0"],
        )
    )

    assert register_response.model_id == model_id
    manager.register_model.assert_awaited_once_with(
        owner_id=user_id,
        model_name="vision-model",
    )
    assert generate_response.sub_model_id == sub_model_id
    manager.generate_sub_model.assert_awaited_once_with(
        model_version_id=version_id,
        layers=["encoder.1", "encoder.0"],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upload_route_decompresses_bundle_before_delegating() -> None:
    manager = MagicMock(spec=ModelManager)
    router = build_model_manager_router(manager)
    _, model_id, version_id = _ids()
    model_info = _model_info()
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

    async def put_model_version(
        *,
        model_id: ModelId,
        model_info: ModelInfo,
        bundle: ReadableArtifactBundle,
    ) -> ModelVersionId:
        received_contents.update(await read_bundle_content(bundle))
        return version_id

    manager.put_model_version.side_effect = put_model_version

    response = await _endpoint(router, "/model-manager/model-versions", "POST")(
        model_id.model_dump_json(),
        model_info.model_dump_json(),
        upload,
    )

    assert response.model_version_id == version_id
    assert received_contents == {model_path: b"onnx-model"}
    manager.put_model_version.assert_awaited_once()
