from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import override

import aiofiles
import httpx

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
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)
from distributed_inference.directory.application.ports.outbound.resolver import (
    ServiceResolver,
)
from distributed_inference.directory.domain.service_instance import (
    ServiceProtocol,
    ServiceType,
)

CHUNK_SIZE = 1024 * 1024


class HttpArtifactStore(ArtifactStore):
    def __init__(self, service_resolver: ServiceResolver):
        self._service_resolver = service_resolver
        self._client = httpx.AsyncClient()

        ## TODO: close self._client once this adapter is wired into an AsyncLifecycle

    async def _resolve_service_root(self) -> str:
        artifact_stores = (
            await self._service_resolver.resolve_service_by_type_and_protocol(
                service_type=ServiceType.ARTIFACT_STORE,
                service_protocol=ServiceProtocol.HTTP,
            )
        )
        ## TODO: We should handle the case where the artifact store is not available
        if not artifact_stores:
            raise ValueError("No artifact store available")

        endpoint = artifact_stores[0].service_endpoint.get_endpoint_string()
        return f"{endpoint}{ARTIFACT_STORE_PATH_PREFIX}"

    @override
    async def put_artifact(
        self,
        artifact_key: ArtifactKey,
        readable_bundle: ReadableArtifactBundle,
    ) -> None:
        service_root = await self._resolve_service_root()

        async with compression_utils.compress_artifact_bundle(
            readable_bundle
        ) as compressed_bundle_path:
            async with aiofiles.open(compressed_bundle_path, "rb") as bundle_file:
                bundle_bytes = await bundle_file.read()

            response = await self._client.post(
                f"{service_root}{ARTIFACT_STORE_PATH_UPLOAD}",
                data={"artifact_key_json": artifact_key.model_dump_json()},
                files={"bundle_zip": (compressed_bundle_path.name, bundle_bytes)},
            )
            response.raise_for_status()

    @override
    @asynccontextmanager
    async def open_artifact(
        self,
        artifact_key: ArtifactKey,
    ) -> AsyncGenerator[ReadableArtifactBundle]:
        service_root = await self._resolve_service_root()
        request = DownloadArtifactRequest(artifact_key=artifact_key)

        async with aiofiles.tempfile.NamedTemporaryFile() as zip_file:
            zip_file_path = Path(str(zip_file.name))

            async with self._client.stream(
                "POST",
                f"{service_root}{ARTIFACT_STORE_PATH_DOWNLOAD}",
                json=request.model_dump(mode="json"),
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(CHUNK_SIZE):
                    await zip_file.write(chunk)

            async with compression_utils.decompress_artifact_bundle(
                zip_file_path
            ) as artifact_bundle:
                yield artifact_bundle

    @override
    async def check_artifact_existence(
        self,
        artifact_key: ArtifactKey,
    ) -> bool:
        service_root = await self._resolve_service_root()
        request = CheckArtifactExistenceRequest(artifact_key=artifact_key)

        response = await self._client.post(
            f"{service_root}{ARTIFACT_STORE_PATH_CHECK}",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        return CheckArtifactExistenceResponse.model_validate_json(response.text).exists
