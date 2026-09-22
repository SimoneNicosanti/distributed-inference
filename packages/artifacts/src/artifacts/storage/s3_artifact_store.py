from contextlib import AbstractAsyncContextManager
from typing import override

from artifacts.contracts.artifact_ref import ArtifactRef
from artifacts.storage.artifact_store import ArtifactStore
from artifacts.workspace.artifact_bundle import ArtifactBundle


class S3ArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        raise NotImplementedError("S3 artifact storage is not implemented")

    @override
    async def put_artifact(
        self,
        artifact_ref: ArtifactRef,
        bundle: ArtifactBundle,
    ) -> None:
        raise NotImplementedError

    @override
    def download_artifact(
        self,
        artifact_ref: ArtifactRef,
    ) -> AbstractAsyncContextManager[ArtifactBundle]:
        raise NotImplementedError

    @override
    async def check_artifact_existence(
        self,
        artifact_ref: ArtifactRef,
    ) -> bool:
        raise NotImplementedError
