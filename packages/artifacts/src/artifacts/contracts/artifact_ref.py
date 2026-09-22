from hashlib import md5

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    """Opaque identifier assigned by the artifact owner."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(min_length=1)


async def build_artifact_ref(value: str) -> ArtifactRef:
    return ArtifactRef(value=md5(value.encode("utf-8")).hexdigest())
