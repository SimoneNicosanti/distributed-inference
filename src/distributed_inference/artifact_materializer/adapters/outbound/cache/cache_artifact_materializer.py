import asyncio
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, override

import aiofiles
import aiofiles.os
import aiofiles.ospath
import aiorwlock

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey
from distributed_inference.artifact_store.domain.artifact_manifest import (
    MANIFEST_FILE_NAME,
    ArtifactManifest,
)
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)

CHUNK_SIZE = 1024 * 1024


class CachedArtifactMaterializer(ArtifactMaterializer):
    def __init__(self, cache_path: Path, artifact_store: ArtifactStore):
        self._artifact_store = artifact_store
        self._cache_locks: dict[ArtifactKey, aiorwlock.RWLock] = {}

        ## TODO: We should handle cache population at restart
        self._cache_path = cache_path
        self._cache_path.mkdir(parents=True, exist_ok=True)

        self._cache: dict[ArtifactKey, MaterializedArtifact] = {}

    @override
    @asynccontextmanager
    async def materialize_artifact(
        self,
        artifact_key: ArtifactKey,
    ) -> AsyncGenerator[MaterializedArtifact]:

        self._cache_locks.setdefault(artifact_key, aiorwlock.RWLock())
        artifact_lock = self._cache_locks[artifact_key]

        while True:
            async with artifact_lock.reader_lock:
                if await self._check_artifact_in_cache(artifact_key):
                    materialized_artifact = self._cache[artifact_key]
                    yield materialized_artifact
                    return

            async with artifact_lock.writer_lock:
                if not await self._check_artifact_in_cache(artifact_key):
                    async with self._artifact_store.open_artifact(
                        artifact_key
                    ) as opened_artifact:
                        await self._cache_artifact_locally(
                            artifact_key, opened_artifact
                        )

                    root_path = await self._build_artifact_root_path(artifact_key)
                    entrypoint_path = await self._build_entrypoint_path(artifact_key)
                    manifest = await self._build_manifest(artifact_key)
                    materialized_artifact = MaterializedArtifact(
                        manifest=manifest,
                        root_path=root_path,
                        entrypoint_path=entrypoint_path,
                    )

                    self._cache[artifact_key] = materialized_artifact

    async def _cache_artifact_locally(
        self,
        artifact_key: ArtifactKey,
        artifact_bundle: ReadableArtifactBundle,
    ) -> None:
        artifact_root_path = await self._build_artifact_root_path(artifact_key)

        for file_info in artifact_bundle.manifest.files_info:
            cache_file_path = artifact_root_path.joinpath(*file_info.file_ppp.parts)
            await aiofiles.os.makedirs(cache_file_path.parent, exist_ok=True)

            async with artifact_bundle.open_file(
                file_info.file_ppp
            ) as artifact_file_reader:
                async with aiofiles.open(cache_file_path, "wb") as local_file:
                    while chunk := await artifact_file_reader.read(CHUNK_SIZE):
                        await local_file.write(chunk)

        manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
        async with aiofiles.open(manifest_path, "w+") as manifest_file:
            await manifest_file.write(artifact_bundle.manifest.model_dump_json())

    async def _check_artifact_in_cache(self, artifact_key: ArtifactKey) -> bool:
        artifact_root_path = await self._build_artifact_root_path(artifact_key)

        if not await aiofiles.ospath.exists(artifact_root_path):
            return False

        manifest_path = artifact_root_path.joinpath(MANIFEST_FILE_NAME)
        if not await aiofiles.ospath.isfile(manifest_path):
            return False

        manifest = await self._build_manifest(artifact_key)

        for file_info in manifest.files_info:
            local_file_path = artifact_root_path.joinpath(*file_info.file_ppp.parts)
            if not await aiofiles.ospath.isfile(local_file_path):
                return False

        return True

    async def _build_artifact_root_path(self, aritifact_key: ArtifactKey) -> Path:

        key_hash = await asyncio.to_thread(self.__hash_artifact_key, aritifact_key)
        artifact_base_path = self._cache_path.joinpath(
            aritifact_key.kind.value
        ).joinpath(str(key_hash))

        return artifact_base_path

    async def _build_manifest(self, aritifact_key: ArtifactKey) -> ArtifactManifest:
        root_path = await self._build_artifact_root_path(aritifact_key)
        manifest_path = root_path.joinpath(MANIFEST_FILE_NAME)

        async with aiofiles.open(manifest_path, "r") as manifest_file:
            manifest_json = await manifest_file.read()
        return ArtifactManifest.model_validate_json(manifest_json)

    async def _build_entrypoint_path(self, aritifact_key: ArtifactKey) -> Path:
        root_path = await self._build_artifact_root_path(aritifact_key)
        manifest = await self._build_manifest(aritifact_key)
        entrypoint_ppp = manifest.entrypoint_ppp
        return root_path.joinpath(*entrypoint_ppp.parts)

    def __hash_artifact_key(self, key: ArtifactKey) -> str:
        md5_hash = hashlib.md5(key.model_dump_json().encode("utf-8")).hexdigest()
        return md5_hash
