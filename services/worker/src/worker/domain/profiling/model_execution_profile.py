from pydantic import BaseModel, ConfigDict

from shared.model.keys import LayerKey
from shared.model.model_version import ModelVersionId


class ModelExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVersionId
    execution_times: dict[LayerKey, float]
