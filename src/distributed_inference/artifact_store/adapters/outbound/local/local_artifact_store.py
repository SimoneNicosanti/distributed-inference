import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import override

import aiofiles.os
import aiofiles.ospath
import aiorwlock

from distributed_inference.artifact_store.adapters.outbound.local.local_readable_artifact_bundle import (
    LocalReadableArtifactBundle,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ArtifactKey,
    ArtifactKind,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactManifest,
)
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)

CHUNK_SIZE = 1024 * 1024


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        base_path.mkdir(parents=True, exist_ok=True)

        self._locks: dict[ArtifactKey, aiorwlock.RWLock] = {}

        self._artifact_dir = base_path.joinpath("artifacts")
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        for kind in ArtifactKind:
            kind_directory = self._artifact_dir.joinpath(kind.value)
            kind_directory.mkdir(parents=True, exist_ok=True)

    @override
    async def put_artifact(
        self,
        artifact_key: ArtifactKey,
        readable_bundle: ReadableArtifactBundle,
    ) -> None:
        self._locks.setdefault(artifact_key, aiorwlock.RWLock())
        artifact_lock = self._locks[artifact_key]

        ## TODO: We should ensure consistency of the bundle
        ## for example when a bundle is changed with a new version
        async with artifact_lock.writer_lock:
            artifact_root_path = await self._build_artifact_root_path(artifact_key)
            manifest = readable_bundle.manifest
            for file_info in manifest.files_info:
                local_file_path = artifact_root_path.joinpath(*file_info.file_ppp.parts)
                await aiofiles.os.makedirs(local_file_path.parent, exist_ok=True)

                async with readable_bundle.open_file(
                    file_info.file_ppp
                ) as artifact_file_reader:
                    async with aiofiles.open(local_file_path, "wb") as local_file:
                        while chunk := await artifact_file_reader.read(CHUNK_SIZE):
                            await local_file.write(chunk)

            manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
            async with aiofiles.open(manifest_path, "w+") as manifest_file:
                await manifest_file.write(readable_bundle.manifest.model_dump_json())

    @override
    @asynccontextmanager
    async def open_artifact(
        self,
        artifact_key: ArtifactKey,
    ) -> AsyncGenerator[ReadableArtifactBundle]:
        self._locks.setdefault(artifact_key, aiorwlock.RWLock())
        artifact_lock = self._locks[artifact_key]

        async with artifact_lock.reader_lock:
            artifact_root_path = await self._build_artifact_root_path(artifact_key)

            manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
            async with aiofiles.open(manifest_path, "r") as manifest_file:
                manifest_json = await manifest_file.read()
            manifest = ArtifactManifest.model_validate_json(manifest_json)

            local_readable_bundle = LocalReadableArtifactBundle(
                manifest, artifact_root_path
            )

            yield local_readable_bundle

    @override
    async def check_artifact_existence(
        self,
        artifact_key: ArtifactKey,
    ) -> bool:

        self._locks.setdefault(artifact_key, aiorwlock.RWLock())
        artifact_lock = self._locks[artifact_key]

        async with artifact_lock.reader_lock:
            artifact_root_path = await self._build_artifact_root_path(artifact_key)
            manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
            if not await aiofiles.ospath.exists(
                artifact_root_path
            ) or not await aiofiles.ospath.exists(manifest_path):
                return False

            async with aiofiles.open(manifest_path, "r") as manifest_file:
                manifest_json = await manifest_file.read()

            manfest = ArtifactManifest.model_validate_json(manifest_json)
            for file_info in manfest.files_info:
                local_file_path = artifact_root_path.joinpath(*file_info.file_ppp.parts)
                if not await aiofiles.ospath.isfile(local_file_path):
                    return False

            return True

    async def _build_artifact_root_path(self, aritifact_key: ArtifactKey) -> Path:

        key_hash = await asyncio.to_thread(self.__hash_artifact_key, aritifact_key)

        artifact_base_path = self._artifact_dir.joinpath(
            aritifact_key.kind.value
        ).joinpath(str(key_hash))

        return artifact_base_path

    ## We should use a different hash, like md5
    def __hash_artifact_key(self, key: ArtifactKey) -> int:
        return hash(key)

    async def _build_entrypoint_path(self, aritifact_key: ArtifactKey) -> Path:
        key_hash = await asyncio.to_thread(self.__hash_artifact_key, aritifact_key)

        manifest = await self._build_manifest(aritifact_key)
        entrypoint_ppp = manifest.entrypoint_ppp
        return (
            self._artifact_dir.joinpath(aritifact_key.kind.value)
            .joinpath(str(key_hash))
            .joinpath(*entrypoint_ppp.parts)
        )

    async def _build_manifest(self, aritifact_key: ArtifactKey) -> ArtifactManifest:
        root_path = await self._build_artifact_root_path(aritifact_key)
        manifest_path = root_path.joinpath(MANIFEST_FILE_NAME)
        async with aiofiles.open(manifest_path, "r") as manifest_file:
            manifest_json = await manifest_file.read()
        return ArtifactManifest.model_validate_json(manifest_json)

    @asynccontextmanager
    async def get_artifact_manifest_root_path_entry_path(
        self, artifact_key: ArtifactKey
    ) -> AsyncGenerator[tuple[ArtifactManifest, Path, Path]]:
        self._locks.setdefault(artifact_key, aiorwlock.RWLock())
        artifact_lock = self._locks[artifact_key]

        async with artifact_lock.reader_lock:
            manifest = await self._build_manifest(artifact_key)
            artifact_root_path = await self._build_artifact_root_path(artifact_key)
            entrypoint_path = await self._build_entrypoint_path(artifact_key)

            yield manifest, artifact_root_path, entrypoint_path
