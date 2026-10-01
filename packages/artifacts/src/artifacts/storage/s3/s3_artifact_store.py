import asyncio
import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import override

import aiofiles

from artifacts.contracts.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactManifest,
)
from artifacts.contracts.artifact_ref import ArtifactRef
from artifacts.storage.artifact_store import ArtifactStore
from artifacts.storage.s3.s3_client import S3Client
from artifacts.workspace.artifact_bundle import ArtifactBundle


class S3ArtifactStore(ArtifactStore):
    def __init__(self, client: S3Client, bucket_name: str) -> None:
        self._client = client
        self._bucket_name = bucket_name

    @override
    async def put_artifact(
        self,
        artifact_ref: ArtifactRef,
        bundle: ArtifactBundle,
    ) -> None:
        for file_info in bundle.manifest.files_info:
            source_path = bundle.root_path.joinpath(*file_info.file_ppp.parts)
            await self._upload_file(
                source_path,
                self._build_object_key(artifact_ref, file_info.file_ppp),
            )

        with TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory, MANIFEST_FILE_NAME)
            async with aiofiles.open(manifest_path, "w") as manifest_file:
                await manifest_file.write(bundle.manifest.model_dump_json())
            await self._upload_file(
                manifest_path,
                self._build_object_key(
                    artifact_ref,
                    PurePosixPath(MANIFEST_FILE_NAME),
                ),
            )

    @override
    @asynccontextmanager
    async def download_artifact(
        self,
        artifact_ref: ArtifactRef,
    ) -> AsyncGenerator[ArtifactBundle]:
        with TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory)
            manifest_path = root_path.joinpath(MANIFEST_FILE_NAME)
            await self._download_file(
                self._build_object_key(
                    artifact_ref,
                    PurePosixPath(MANIFEST_FILE_NAME),
                ),
                manifest_path,
            )

            async with aiofiles.open(manifest_path) as manifest_file:
                manifest = ArtifactManifest.model_validate_json(
                    await manifest_file.read()
                )

            for file_info in manifest.files_info:
                destination_path = root_path.joinpath(*file_info.file_ppp.parts)
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                await self._download_file(
                    self._build_object_key(artifact_ref, file_info.file_ppp),
                    destination_path,
                )

            yield ArtifactBundle(manifest=manifest, root_path=root_path)

    @override
    async def check_artifact_existence(
        self,
        artifact_ref: ArtifactRef,
    ) -> bool:
        manifest_key = self._build_object_key(
            artifact_ref,
            PurePosixPath(MANIFEST_FILE_NAME),
        )
        try:
            await asyncio.to_thread(
                self._client.head_object,
                Bucket=self._bucket_name,
                Key=manifest_key,
            )
        except Exception as error:
            if self._is_missing_object_error(error):
                return False
            raise
        return True

    async def _upload_file(self, source_path: Path, key: str) -> None:
        await asyncio.to_thread(
            self._client.upload_file,
            str(source_path),
            self._bucket_name,
            key,
        )

    async def _download_file(self, key: str, destination_path: Path) -> None:
        await asyncio.to_thread(
            self._client.download_file,
            self._bucket_name,
            key,
            str(destination_path),
        )

    @staticmethod
    def _build_object_key(
        artifact_ref: ArtifactRef,
        file_ppp: PurePosixPath,
    ) -> str:
        ref_hash = hashlib.sha256(artifact_ref.value.encode("utf-8")).hexdigest()
        return f"{ref_hash}/{file_ppp.as_posix()}"

    @staticmethod
    def _is_missing_object_error(error: Exception) -> bool:
        response = getattr(error, "response", None)
        if not isinstance(response, dict):
            return False
        error_details = response.get("Error")
        if not isinstance(error_details, dict):
            return False
        return error_details.get("Code") in {"404", "NoSuchKey", "NotFound"}
