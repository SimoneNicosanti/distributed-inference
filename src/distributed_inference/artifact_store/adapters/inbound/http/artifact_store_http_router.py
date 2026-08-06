import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter

from distributed_inference.artifact_processing import compression_utils
from distributed_inference.artifact_store.adapters.inbound.http.artifact_store_http_paths import (
    ARTIFACT_STORE_PATH_CHECK,
    ARTIFACT_STORE_PATH_DOWNLOAD,
    ARTIFACT_STORE_PATH_PREFIX,
    ARTIFACT_STORE_PATH_UPLOAD,
)
from distributed_inference.artifact_store.adapters.inbound.http.artifact_store_http_schema import (
    CheckArtifactExistenceRequest,
    CheckArtifactExistenceResponse,
    DownloadArtifactRequest,
    UploadArtifactResponse,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey

CHUNK_SIZE = 1024 * 1024


def build_http_artifact_store(artifact_store: ArtifactStore) -> APIRouter:
    router = APIRouter(
        prefix=ARTIFACT_STORE_PATH_PREFIX,
        tags=["artifact-store"],
    )

    @router.post(
        ARTIFACT_STORE_PATH_UPLOAD,
        response_model=UploadArtifactResponse,
    )
    async def upload_artifact(
        artifact_key_json: str = Form(),
        bundle_zip: UploadFile = File(),
    ) -> UploadArtifactResponse:
        artifact_key: ArtifactKey = TypeAdapter(ArtifactKey).validate_json(
            artifact_key_json
        )
        async with aiofiles.tempfile.NamedTemporaryFile() as zip_file:
            zip_file_path = Path(str(zip_file.name))
            while chunk := await asyncio.to_thread(bundle_zip.file.read, CHUNK_SIZE):
                await zip_file.write(chunk)
            async with compression_utils.decompress_artifact_bundle(
                zip_file_path
            ) as artifact_bundle:
                await artifact_store.put_artifact(artifact_key, artifact_bundle)

        return UploadArtifactResponse(
            artifact_key=artifact_key,
        )

    @router.post(ARTIFACT_STORE_PATH_DOWNLOAD)
    async def download_artifact(
        request: DownloadArtifactRequest,
    ) -> StreamingResponse:
        async def stream_artifact() -> AsyncIterator[bytes]:
            artifact_key = request.artifact_key
            async with (
                artifact_store.open_artifact(artifact_key) as artifact_bundle,
                compression_utils.compress_artifact_bundle(
                    artifact_bundle
                ) as zip_file_path,
                aiofiles.open(zip_file_path, "rb") as zip_file,
            ):
                while chunk := await zip_file.read(CHUNK_SIZE):
                    yield chunk

        return StreamingResponse(
            stream_artifact(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": ('attachment; filename="artifact.zip"')},
        )

    @router.post(
        ARTIFACT_STORE_PATH_CHECK, response_model=CheckArtifactExistenceResponse
    )
    async def check_artifact_existence(
        request: CheckArtifactExistenceRequest,
    ) -> CheckArtifactExistenceResponse:
        artifact_key = request.artifact_key

        exists = await artifact_store.check_artifact_existence(artifact_key)

        return CheckArtifactExistenceResponse(exists=exists)

    return router
