from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import override

import aiorwlock

from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.storage.artifact_store import (
    ArtifactCheckError,
    ArtifactDownloadError,
    ArtifactStore,
    ArtifactUploadError,
)
from shared.artifact.artifact_ref import ArtifactRef


class CachedArtifactStore(ArtifactStore):
    """Artifact store decorator backed by an authoritative store and a cache."""

    def __init__(
        self,
        authoritative_store: ArtifactStore,
        cache_store: ArtifactStore,
    ) -> None:
        self._authoritative_store = authoritative_store
        self._cache_store = cache_store
        self._locks: dict[ArtifactRef, aiorwlock.RWLock] = {}

    @override
    async def upload_artifact(
        self,
        artifact_ref: ArtifactRef,
        bundle: ArtifactBundle,
    ) -> None:
        artifact_lock = self._locks.setdefault(artifact_ref, aiorwlock.RWLock())

        async with artifact_lock.writer_lock:
            await self._authoritative_store.upload_artifact(artifact_ref, bundle)
            await self._cache_store.upload_artifact(artifact_ref, bundle)

    @override
    @asynccontextmanager
    async def download_artifact(
        self,
        artifact_ref: ArtifactRef,
    ) -> AsyncGenerator[ArtifactBundle]:
        artifact_lock = self._locks.setdefault(artifact_ref, aiorwlock.RWLock())

        async with artifact_lock.reader_lock:
            try:
                cache_hit = await self._cache_store.check_artifact_existence(artifact_ref)
            except ArtifactCheckError as error:
                raise ArtifactDownloadError(
                    f"Unable to download artifact {artifact_ref}"
                ) from error

            if cache_hit:
                async with self._cache_store.download_artifact(artifact_ref) as bundle:
                    yield bundle
                return

        async with artifact_lock.writer_lock:
            try:
                if not await self._cache_store.check_artifact_existence(artifact_ref):
                    async with self._authoritative_store.download_artifact(
                        artifact_ref
                    ) as source:
                        await self._cache_store.upload_artifact(artifact_ref, source)
            except (ArtifactCheckError, ArtifactUploadError) as error:
                raise ArtifactDownloadError(
                    f"Unable to download artifact {artifact_ref}"
                ) from error

        async with (
            artifact_lock.reader_lock,
            self._cache_store.download_artifact(artifact_ref) as bundle,
        ):
            yield bundle

    @override
    async def check_artifact_existence(self, artifact_ref: ArtifactRef) -> bool:
        return await self._authoritative_store.check_artifact_existence(artifact_ref)
