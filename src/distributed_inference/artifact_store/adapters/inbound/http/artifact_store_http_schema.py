from pydantic import BaseModel, ConfigDict

from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey


class UploadArtifactResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    artifact_key: ArtifactKey


class DownloadArtifactRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    artifact_key: ArtifactKey


class CheckArtifactExistenceRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    artifact_key: ArtifactKey


class CheckArtifactExistenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    exists: bool
