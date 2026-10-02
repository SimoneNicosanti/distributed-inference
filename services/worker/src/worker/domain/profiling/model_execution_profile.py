from pydantic import BaseModel, ConfigDict

from shared.model.keys import LayerKey
from shared.model.model_version import ModelVersionId


class BackendLayerExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_time: float
    memory: float


class LayerExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_key: LayerKey

    cpu_execution_profile: BackendLayerExecutionProfile
    gpu_execution_profile: BackendLayerExecutionProfile | None


class ModelExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVersionId
    layer_profiles: dict[LayerKey, LayerExecutionProfile]
