from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    """Opaque identifier assigned by the artifact owner."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(min_length=1)
