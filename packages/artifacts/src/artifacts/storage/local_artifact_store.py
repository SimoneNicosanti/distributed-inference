import asyncio
import hashlib
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import override

import aiofiles
import aiofiles.os
import aiofiles.ospath
import aiorwlock

from artifacts.contracts.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactManifest,
)
from artifacts.contracts.artifact_ref import ArtifactRef
from artifacts.storage.artifact_store import ArtifactStore
from artifacts.workspace.artifact_bundle import ArtifactBundle


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_path: Path) -> None:
        self._artifact_dir = base_path.joinpath("artifacts")
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[ArtifactRef, aiorwlock.RWLock] = {}

    @override
    async def put_artifact(
        self,
        artifact_ref: ArtifactRef,
        bundle: ArtifactBundle,
    ) -> None:
        artifact_lock = self._locks.setdefault(artifact_ref, aiorwlock.RWLock())

        async with artifact_lock.writer_lock:
            artifact_root_path = self._build_artifact_root_path(artifact_ref)
            await self._copy_artifact_files(
                bundle.root_path,
                artifact_root_path,
                bundle.manifest,
            )

            manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
            async with aiofiles.open(manifest_path, "w") as manifest_file:
                await manifest_file.write(bundle.manifest.model_dump_json())

    @override
    @asynccontextmanager
    async def download_artifact(
        self,
        artifact_ref: ArtifactRef,
    ) -> AsyncGenerator[ArtifactBundle]:
        artifact_lock = self._locks.setdefault(artifact_ref, aiorwlock.RWLock())

        async with artifact_lock.reader_lock:
            manifest = await self._build_manifest(artifact_ref)
            yield ArtifactBundle(
                manifest=manifest,
                root_path=self._build_artifact_root_path(artifact_ref),
            )

    @override
    async def check_artifact_existence(self, artifact_ref: ArtifactRef) -> bool:
        artifact_lock = self._locks.setdefault(artifact_ref, aiorwlock.RWLock())

        async with artifact_lock.reader_lock:
            artifact_root_path = self._build_artifact_root_path(artifact_ref)
            manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
            if not await aiofiles.ospath.isdir(
                artifact_root_path
            ) or not await aiofiles.ospath.isfile(manifest_path):
                return False

            try:
                manifest = await self._build_manifest(artifact_ref)
            except OSError, ValueError:
                return False

            return all(
                [
                    await aiofiles.ospath.isfile(
                        artifact_root_path.joinpath(*file_info.file_ppp.parts)
                    )
                    for file_info in manifest.files_info
                ]
            )

    def _build_artifact_root_path(self, artifact_ref: ArtifactRef) -> Path:
        ref_hash = hashlib.sha256(artifact_ref.value.encode("utf-8")).hexdigest()
        return self._artifact_dir.joinpath(ref_hash)

    async def _build_manifest(self, artifact_ref: ArtifactRef) -> ArtifactManifest:
        manifest_path = self._build_artifact_root_path(artifact_ref).joinpath(
            MANIFEST_FILE_NAME
        )
        async with aiofiles.open(manifest_path) as manifest_file:
            manifest_json = await manifest_file.read()
        return ArtifactManifest.model_validate_json(manifest_json)

    async def _copy_artifact_files(
        self,
        source_root: Path,
        destination_root: Path,
        manifest: ArtifactManifest,
    ) -> None:
        for file_info in manifest.files_info:
            source_path = source_root.joinpath(*file_info.file_ppp.parts)
            destination_path = destination_root.joinpath(*file_info.file_ppp.parts)
            await aiofiles.os.makedirs(destination_path.parent, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, source_path, destination_path)
