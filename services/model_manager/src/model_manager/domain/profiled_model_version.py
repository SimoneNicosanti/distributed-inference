from pydantic import BaseModel, ConfigDict

from integration.model.model_version import ModelVersion, ModelVersionId
from model_manager.domain.model_version_graph import ModelVersionGraph


class ProfiledModelVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version: ModelVersion
    model_version_graph: ModelVersionGraph

    @property
    def model_version_id(self) -> ModelVersionId:
        return self.model_version.model_version_id
