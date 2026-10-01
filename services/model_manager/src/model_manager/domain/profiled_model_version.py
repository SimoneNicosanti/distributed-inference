from pydantic import BaseModel, ConfigDict

from model_manager.domain.model_version_graph import ModelVersionGraph
from shared.model.model_version import ModelVersion, ModelVersionId


class ProfiledModelVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version: ModelVersion
    model_version_graph: ModelVersionGraph

    @property
    def model_version_id(self) -> ModelVersionId:
        return self.model_version.model_version_id
