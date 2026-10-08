from pydantic import BaseModel, ConfigDict

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class BackendLayerExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_time: float
    memory: float


class LayerExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_key: LayerKey

    cpu_execution_profile: BackendLayerExecutionProfile
    gpu_execution_profile: BackendLayerExecutionProfile | None


class ShapeExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape_point: ShapePoint
    layer_profiles: dict[LayerKey, LayerExecutionProfile]


class ModelExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVariantId
    shape_execution_profiles: tuple[ShapeExecutionProfile, ...]
