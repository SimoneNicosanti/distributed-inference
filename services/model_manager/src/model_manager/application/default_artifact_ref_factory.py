import asyncio
import hashlib
from typing import override

from model_manager.application.abc.artifact_ref_factory import ArtifactRefFactory
from shared.artifact.artifact_ref import ArtifactRef


class DefaultArtifactRefFactory(ArtifactRefFactory):
    @override
    async def build_artifact_ref(self, value: str) -> ArtifactRef:
        sha256_value = await asyncio.to_thread(hashlib.sha256, value.encode("utf-8"))

        return ArtifactRef(value=sha256_value.hexdigest())
